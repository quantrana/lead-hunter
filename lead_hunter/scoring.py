from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal
from urllib.parse import urlparse

from .models import OutreachDraft, ResearchProfile, RunConfig, ScoreResult


UNSUPPORTED_OUTREACH_CLAIMS = [
    "we helped",
    "we have helped",
    "hourglass helped",
    "guarantee",
    "guaranteed",
    "i sent",
    "this email was sent",
    "as a customer",
]


EvidenceGrounding = Literal["strong", "weak", "insufficient"]


@dataclass
class OutreachValidationResult:
    outreach: OutreachDraft
    evidence_grounding: EvidenceGrounding = "strong"
    hard_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.hard_errors


def enforce_save_threshold(score: ScoreResult, run_config: RunConfig) -> ScoreResult:
    should_save = False
    reason = score.rejection_reason
    if score.fit_tier in {"A", "B"} and score.total_score >= run_config.min_score:
        should_save = True
    if score.fit_tier == "C" and run_config.allow_c_tier and score.total_score >= 60:
        should_save = True
    if score.fit_tier == "Reject":
        should_save = False
        reason = reason or "Rejected by scoring rubric"
    if not should_save and not reason:
        reason = f"Score {score.total_score} did not meet save threshold {run_config.min_score}"
    return score.model_copy(update={"should_save": should_save, "rejection_reason": reason})


def fallback_score_profile(profile: ResearchProfile, run_config: RunConfig) -> ScoreResult:
    text = " ".join(
        [
            profile.location or "",
            profile.industry or "",
            profile.company_size_signal or "",
            profile.growth_signal or "",
            profile.hiring_signal or "",
            " ".join(profile.automation_pain_signals),
            " ".join(item.quote_or_summary for item in profile.evidence_items),
        ]
    ).lower()
    score = 0
    if any(term.lower() in text for term in [run_config.target_country, run_config.target_city_preference, "australia"]):
        score += 14
    if profile.automation_pain_signals:
        score += min(20, 10 + len(profile.automation_pain_signals) * 4)
    if profile.growth_signal or profile.company_size_signal:
        score += 12
    if any(term in text for term in ["manual", "spreadsheet", "queue", "reporting", "invoice", "onboarding", "compliance"]):
        score += 14
    if profile.likely_departments_with_pain:
        score += 15
    if profile.website or profile.source_urls:
        score += 4
    score += min(10, len(profile.evidence_items) * 4)
    if "ai consultancy" in text or "automation consultancy" in text:
        score -= 20
    if not profile.evidence_items:
        score -= 25
    if "australia" not in text and "melbourne" not in text:
        score -= 20
    score = max(0, min(100, score))
    if score >= 80:
        tier = "A"
    elif score >= 70:
        tier = "B"
    elif score >= 60:
        tier = "C"
    else:
        tier = "Reject"
    agent_type = choose_agent_type(text)
    result = ScoreResult(
        total_score=score,
        fit_tier=tier,
        recommended_agent_type=agent_type,
        pain_summary="; ".join(profile.automation_pain_signals[:3]) or "No clear operational pain found.",
        why_hourglass=f"{agent_type} appears relevant if the public evidence reflects a repeatable operational workflow.",
        strongest_evidence=profile.evidence_items[0].quote_or_summary if profile.evidence_items else "No usable evidence.",
        risks_or_uncertainties="Evidence is limited and should be reviewed by a human.",
        confidence=min(0.95, max(profile.confidence, 0.25 if profile.evidence_items else 0.1)),
        should_save=tier in {"A", "B"},
        rejection_reason=None if tier in {"A", "B"} else "Insufficient score or evidence quality.",
        reasoning_summary="Fallback rubric applied deterministically from extracted public evidence.",
    )
    return enforce_save_threshold(result, run_config)


def choose_agent_type(text: str):
    if re.search(r"invoice|accounts receivable|payment|finance", text):
        return "Invoice Chasing Agent"
    if re.search(r"support|ticket|queue|customer service", text):
        return "Support Agent"
    if re.search(r"onboarding|implementation", text):
        return "Onboarding Agent"
    if re.search(r"report|dashboard|spreadsheet|data", text):
        return "Reporting Agent"
    if re.search(r"document|pdf|extract|forms", text):
        return "Document Extraction Agent"
    if re.search(r"compliance|audit|risk", text):
        return "Compliance Workflow Agent"
    if re.search(r"recruit|talent|hiring", text):
        return "Recruiting Ops Agent"
    if re.search(r"sales|crm|revops", text):
        return "Sales Ops Agent"
    if re.search(r"success|retention", text):
        return "Customer Success Agent"
    return "Internal Ops Router"


def validate_outreach(profile: ResearchProfile, outreach: OutreachDraft) -> OutreachValidationResult:
    outreach = outreach.model_copy(update={"outreach_pitch": truncate_pitch_at_sentence_boundary(outreach.outreach_pitch)})
    errors: list[str] = []
    warnings: list[str] = []
    if _is_insufficient_evidence(outreach):
        replacement = insufficient_evidence_outreach(profile)
        return OutreachValidationResult(
            outreach=replacement,
            evidence_grounding="insufficient",
            warnings=["Model returned INSUFFICIENT_EVIDENCE; lead kept for review with no user-facing pitch token."],
        )
    company = profile.company_name.lower()
    combined = f"{outreach.outreach_subject}\n{outreach.outreach_pitch}".lower()
    if company not in combined:
        errors.append("Outreach must contain the company name.")
    grounding = evidence_grounding_status(profile, outreach)
    if grounding != "strong":
        warnings.append("Outreach evidence grounding is weak; demote tier by one and keep model pitch for human review.")
    for claim in UNSUPPORTED_OUTREACH_CLAIMS:
        if claim in combined:
            errors.append(f"Outreach contains unsupported or unsafe claim: {claim}")
    if "send" in combined and "email" in combined:
        errors.append("Outreach must not imply automated sending.")
    return OutreachValidationResult(outreach=outreach, evidence_grounding=grounding, hard_errors=errors, warnings=warnings)


def prepare_outreach_for_save(
    profile: ResearchProfile,
    score: ScoreResult,
    outreach: OutreachDraft,
) -> tuple[OutreachDraft, ScoreResult, OutreachValidationResult]:
    validation = validate_outreach(profile, outreach)
    validated_outreach = validation.outreach.model_copy(update={"evidence_grounding": validation.evidence_grounding})
    validation.outreach = validated_outreach
    final_score = score
    if validation.evidence_grounding in {"weak", "insufficient"}:
        final_score = demote_score_tier(score)
    return validated_outreach, final_score, validation


def evidence_grounding_status(profile: ResearchProfile, outreach: OutreachDraft) -> EvidenceGrounding:
    if _is_insufficient_evidence(outreach):
        return "insufficient"
    combined = f"{outreach.outreach_subject}\n{outreach.outreach_pitch}"
    if _contains_evidence_url(profile, combined):
        return "strong"
    if _contains_verbatim_evidence_phrase(profile, combined):
        return "strong"
    if _contains_resolved_domain(profile, combined):
        return "strong"
    return "weak"


def demote_score_tier(score: ScoreResult) -> ScoreResult:
    tier_order = {"A": "B", "B": "C", "C": "Reject", "Reject": "Reject"}
    next_tier = tier_order[score.fit_tier]
    should_save = False if next_tier == "Reject" else score.should_save
    reason = score.rejection_reason
    if next_tier == "Reject":
        reason = reason or "Evidence grounding was insufficient for a final-tier lead."
    return score.model_copy(
        update={
            "fit_tier": next_tier,
            "should_save": should_save,
            "rejection_reason": reason,
            "risks_or_uncertainties": _append_unique_sentence(
                score.risks_or_uncertainties,
                "Outreach evidence grounding was weak and the tier was demoted for human review.",
            ),
        }
    )


def truncate_pitch_at_sentence_boundary(text: str, limit: int = 800) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[:limit].rstrip()
    matches = list(re.finditer(r"[.!?](?:\s|$)", clipped))
    if matches:
        return clipped[: matches[-1].end()].strip()
    word_boundary = clipped.rfind(" ")
    if word_boundary > 0:
        return clipped[:word_boundary].rstrip()
    return clipped


def insufficient_evidence_outreach(profile: ResearchProfile) -> OutreachDraft:
    return OutreachDraft(
        outreach_subject=f"AI agent idea for {profile.company_name}",
        outreach_pitch=(
            f"{profile.company_name} needs human review before outreach. "
            "The model could not generate a pitch grounded in specific public evidence without becoming generic."
        ),
        suggested_next_step="Human review required before writing or sending any outreach.",
        reasoning_summary="Model returned INSUFFICIENT_EVIDENCE; placeholder copy prevents exposing the token as a pitch.",
    )


def fallback_outreach(profile: ResearchProfile, score: ScoreResult) -> OutreachDraft:
    evidence_text = profile.evidence_items[0].quote_or_summary if profile.evidence_items else score.strongest_evidence
    signal = _short_evidence_anchor(evidence_text or score.strongest_evidence or score.pain_summary)
    pain = _short_sentence(score.pain_summary or "internal operations", 18)
    return OutreachDraft(
        outreach_subject=f"AI agent idea for {profile.company_name}",
        outreach_pitch=(
            f"Hey {profile.company_name},\n\n"
            f"I noticed {signal}. It looks like there may be a workflow around {pain} where an AI agent could help prepare next actions for review.\n\n"
            f"A practical first build could be a {score.recommended_agent_type}, keeping human approval in place.\n\n"
            "Worth a quick look?"
        ),
        suggested_next_step="Human review, then decide whether this company is worth contacting.",
        reasoning_summary="Fallback outreach generated from strongest evidence and score summary.",
    )


def _short_evidence_anchor(text: str, max_words: int = 20) -> str:
    return _short_sentence(text, max_words).rstrip(".")


def _short_sentence(text: str, max_words: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."


def _is_insufficient_evidence(outreach: OutreachDraft) -> bool:
    return (outreach.outreach_pitch or "").strip().upper() == "INSUFFICIENT_EVIDENCE"


def _contains_evidence_url(profile: ResearchProfile, text: str) -> bool:
    normalized_text = _normalize_urlish_text(text)
    urls = list(profile.source_urls) + [item.url for item in profile.evidence_items]
    for url in urls:
        normalized = _normalize_urlish_text(url)
        if normalized and normalized in normalized_text:
            return True
    return False


def _contains_resolved_domain(profile: ResearchProfile, text: str) -> bool:
    domain = _domain_from_value(profile.domain or profile.website)
    if not domain:
        return False
    return domain.lower() in text.lower()


def _contains_verbatim_evidence_phrase(profile: ResearchProfile, text: str) -> bool:
    text_words = _normalized_words(text)
    if len(text_words) < 4:
        return False
    text_phrases = {" ".join(text_words[index : index + 4]) for index in range(len(text_words) - 3)}
    for item in profile.evidence_items:
        words = _normalized_words(item.quote_or_summary)
        if len(words) < 4:
            continue
        for index in range(len(words) - 3):
            if " ".join(words[index : index + 4]) in text_phrases:
                return True
    return False


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _normalize_urlish_text(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text.rstrip("/")


def _domain_from_value(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = (parsed.netloc or parsed.path).lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _append_unique_sentence(text: str, sentence: str) -> str:
    if sentence in (text or ""):
        return text
    return f"{(text or '').rstrip()} {sentence}".strip()
