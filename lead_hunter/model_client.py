from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .critic import CRITIC_DEFAULT_MODEL, CRITIC_MAX_OUTPUT_TOKENS, CRITIC_MODEL_ENV_VAR, CRITIC_SYSTEM_PROMPT, CritiqueResult, _load_pitch_style, _load_rubric
from .models import CandidateSignal, EvidenceItem, OutreachDraft, ResearchProfile, RunConfig, ScoreResult
from .planner import (
    PLANNER_MAX_OUTPUT_TOKENS,
    PLANNER_MODEL,
    PLANNER_MODEL_ENV_VAR,
    NextAction,
    RunState,
    heuristic_next_action,
    planner_system_prompt,
    planner_user_payload,
)
from .scoring import enforce_save_threshold, fallback_outreach, fallback_score_profile
from .utils import compact_text, domain_from_url

T = TypeVar("T", bound=BaseModel)


class ModelClientError(RuntimeError):
    pass


class BaseModelClient:
    def research_profile(
        self,
        candidate: CandidateSignal,
        pages: list[dict[str, str]],
        run_config: RunConfig,
    ) -> ResearchProfile:
        raise NotImplementedError

    def score_profile(self, profile: ResearchProfile, run_config: RunConfig) -> ScoreResult:
        raise NotImplementedError

    def generate_outreach(self, profile: ResearchProfile, score: ScoreResult) -> OutreachDraft:
        raise NotImplementedError

    def decide_next_action(self, state: RunState, config: Any) -> NextAction:
        raise NotImplementedError

    def critic_review(self, recent_leads: list[dict[str, Any]], current_rubric: str, current_pitch_style: str) -> CritiqueResult:
        raise NotImplementedError


class OpenAIModelClient(BaseModelClient):
    def __init__(self, model_env_var: str, temperature: float, max_output_tokens: int, budget=None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ModelClientError("OPENAI_API_KEY is required for production runs. Use --test-mode for fixture smoke tests.")
        self.model = os.getenv(model_env_var)
        if not self.model:
            raise ModelClientError(f"{model_env_var} is required and must contain the OpenAI model name.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency install issue
            raise ModelClientError("openai package is not installed. Run: pip install -r requirements.txt") from exc
        self.client = OpenAI(api_key=api_key)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.budget = budget

    def research_profile(
        self,
        candidate: CandidateSignal,
        pages: list[dict[str, str]],
        run_config: RunConfig,
    ) -> ResearchProfile:
        prompt = {
            "task": "Extract a conservative public-evidence company research profile for Lead Hunter.",
            "rules": [
                "Do not invent facts or URLs.",
                "Use low confidence if evidence is weak.",
                "Prefer company-level operational pain over personal data.",
                "Australia relevance matters.",
                "Use company-owned homepage/about/careers pages as primary evidence.",
                "Use the original discovery snippet only as supplementary evidence.",
                "Verify Australia relevance from ABN mentions, AU phone numbers, AU addresses, or office/location copy where available.",
                "Infer industry from homepage tagline and about page, not from the job title alone.",
                "Use careers evidence to identify scale signals and hiring signals.",
                "Look for ops, data, RevOps, support, finance, onboarding, compliance, reporting, or manual workflow pain.",
            ],
            "target_country": run_config.target_country,
            "target_city_preference": run_config.target_city_preference,
            "candidate": candidate.model_dump(),
            "evidence_inputs": pages,
        }
        profile = self._structured_call(
            ResearchProfile,
            "lead_hunter_research_profile",
            "You extract structured public business research for an AI consultancy prospecting agent. Company-owned pages are primary evidence; discovery snippets are supplementary.",
            json.dumps(prompt, ensure_ascii=True),
        )
        return profile.model_copy(update={"trace_id": candidate.trace_id})

    def score_profile(self, profile: ResearchProfile, run_config: RunConfig) -> ScoreResult:
        rubric_text = _load_rubric()
        prompt = {
            "task": "Score this company as a fit for Hourglass, a Melbourne AI consultancy building custom AI agents.",
            "rubric_md": rubric_text,
            "profile": profile.model_dump(),
        }
        score = self._structured_call(
            ScoreResult,
            "lead_hunter_score",
            "You score leads conservatively from evidence. Use the live rubric.md content exactly as the scoring guide. Save only strong, supported leads.",
            json.dumps(prompt, ensure_ascii=True),
        )
        return enforce_save_threshold(score, run_config)

    def generate_outreach(self, profile: ResearchProfile, score: ScoreResult) -> OutreachDraft:
        pitch_style_text = _load_pitch_style()
        prompt = {
            "task": "Generate concise evidence-grounded outreach copy. Do not send anything.",
            "pitch_style_md": pitch_style_text,
            "grounding_directives": [
                "You will be given research evidence as a structured list of evidence_items, each with a url and a quote_or_summary.",
                "Your pitch's OPENER (first sentence) MUST reference a specific concrete detail from one of the evidence_items -- a named workflow, a specific role being hired, a specific product line, or a specific operational scale signal. Generic openers like 'I noticed your team is growing' are not allowed. Anchor the opener to something specific that came from the research.",
                "Your pitch's BODY MUST include either a verbatim 4-word-or-longer phrase from one of the evidence quotes, OR a direct URL from evidence_urls.",
                "If you can't ground your pitch in something specific, set outreach_pitch exactly to INSUFFICIENT_EVIDENCE and do not add any other pitch text. The agent will handle this case.",
            ],
            "evidence_urls": [item.url for item in profile.evidence_items],
            "evidence_items": [item.model_dump() for item in profile.evidence_items],
            "subject_format": f"AI agent idea for {profile.company_name}",
            "profile": profile.model_dump(),
            "score": score.model_dump(),
        }
        return self._structured_call(
            OutreachDraft,
            "lead_hunter_outreach",
            "You write short, evidence-backed outreach drafts for human review only. Follow the live pitch_style.md content exactly; if it includes a special final-word rule, obey it. The opener and body must be grounded in supplied evidence, or outreach_pitch must be exactly INSUFFICIENT_EVIDENCE.",
            json.dumps(prompt, ensure_ascii=True),
        )

    def decide_next_action(self, state: RunState, config: Any) -> NextAction:
        model = os.getenv(PLANNER_MODEL_ENV_VAR, PLANNER_MODEL)
        system = planner_system_prompt()
        user = planner_user_payload(state, config)
        last_error: Exception | None = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else "\nReturn only valid JSON matching the schema. No prose."
            try:
                response = self.client.responses.parse(
                    model=model,
                    instructions=system,
                    input=user + suffix,
                    temperature=0,
                    max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS,
                    text_format=NextAction,
                )
                parsed = getattr(response, "output_parsed", None)
                self._record_usage(response, system, user, parsed)
                if parsed is not None:
                    return parsed
                text = getattr(response, "output_text", None) or _extract_response_text(response)
                return NextAction.model_validate_json(text)
            except Exception as exc:  # pragma: no cover - live API dependent
                last_error = exc
        raise ModelClientError(f"OpenAI planner structured output failed: {last_error}")

    def critic_review(self, recent_leads: list[dict[str, Any]], current_rubric: str, current_pitch_style: str) -> CritiqueResult:
        model = os.getenv(CRITIC_MODEL_ENV_VAR) or os.getenv("OPENAI_MODEL") or CRITIC_DEFAULT_MODEL
        prompt = {
            "recent_leads": recent_leads[-10:],
            "current_rubric_md": current_rubric,
            "current_pitch_style_md": current_pitch_style,
        }
        last_error: Exception | None = None
        user = json.dumps(prompt, ensure_ascii=True)
        for attempt in range(2):
            suffix = "" if attempt == 0 else "\nReturn only valid JSON matching the schema. No prose."
            try:
                response = self.client.responses.parse(
                    model=model,
                    instructions=CRITIC_SYSTEM_PROMPT,
                    input=user + suffix,
                    temperature=0.2,
                    max_output_tokens=CRITIC_MAX_OUTPUT_TOKENS,
                    text_format=CritiqueResult,
                )
                parsed = getattr(response, "output_parsed", None)
                self._record_usage(response, CRITIC_SYSTEM_PROMPT, user, parsed)
                if parsed is not None:
                    return parsed
                text = getattr(response, "output_text", None) or _extract_response_text(response)
                return CritiqueResult.model_validate_json(text)
            except Exception as exc:  # pragma: no cover - live API dependent
                last_error = exc
        raise ModelClientError(f"OpenAI critic structured output failed: {last_error}")

    def _structured_call(self, model_cls: type[T], name: str, system: str, user: str) -> T:
        last_error: Exception | None = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else "\nReturn only valid JSON matching the schema. No prose."
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=system,
                    input=user + suffix,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    text_format=model_cls,
                )
                parsed = getattr(response, "output_parsed", None)
                self._record_usage(response, system, user, parsed)
                if parsed is not None:
                    return parsed
                text = getattr(response, "output_text", None) or _extract_response_text(response)
                return model_cls.model_validate_json(text)
            except Exception as exc:  # pragma: no cover - live API dependent
                last_error = exc
        raise ModelClientError(f"OpenAI structured output failed for {name}: {last_error}")

    def _record_usage(self, response: Any, system: str, user: str, parsed: Any) -> None:
        if not self.budget:
            return
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        if input_tokens is None:
            input_tokens = max(1, len(system + user) // 4)
        if output_tokens is None:
            output_text = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else getattr(response, "output_text", "")
            output_tokens = max(1, len(output_text or "") // 4)
        self.budget.record_model_call(input_tokens, output_tokens)


class FakeModelClient(BaseModelClient):
    def __init__(self, fixture_path: str | Path | None = None, budget=None) -> None:
        self.fixture_data: dict[str, Any] = {}
        self.budget = budget
        if fixture_path and Path(fixture_path).exists():
            self.fixture_data = json.loads(Path(fixture_path).read_text(encoding="utf-8"))

    def research_profile(
        self,
        candidate: CandidateSignal,
        pages: list[dict[str, str]],
        run_config: RunConfig,
    ) -> ResearchProfile:
        self._record_fake_call(candidate.model_dump_json() + json.dumps(pages, ensure_ascii=True))
        if "Invalid JSON" in candidate.company_name:
            raise ModelClientError("Fake invalid JSON response after retry")
        if candidate.company_name in self.fixture_data.get("research_profiles", {}):
            data = self.fixture_data["research_profiles"][candidate.company_name]
            return ResearchProfile.model_validate(data).model_copy(update={"trace_id": candidate.trace_id})
        combined = " ".join([candidate.signal_text] + [page.get("text", "") for page in pages])
        lower = combined.lower()
        evidence = [
            EvidenceItem(
                url=candidate.source_url,
                title=f"{candidate.company_name} public signal",
                quote_or_summary=compact_text(candidate.signal_text, 240),
                signal_type="source_signal",
                why_it_matters="This public signal suggests operational workflow complexity.",
            )
        ]
        for page in pages:
            signal_type = page.get("signal_type", "")
            if signal_type in {"homepage", "about", "careers"}:
                evidence.append(
                    EvidenceItem(
                        url=page.get("url", candidate.website or candidate.source_url),
                        title=page.get("title") or signal_type.title(),
                        quote_or_summary=compact_text(page.get("quote_or_summary") or page.get("text", ""), 240),
                        signal_type=signal_type,
                        why_it_matters=f"The company-owned {signal_type} page provides primary evidence for qualification.",
                    )
                )
        if candidate.website and len(evidence) == 1:
            evidence.append(
                EvidenceItem(
                    url=candidate.website,
                    title=f"{candidate.company_name} website",
                    quote_or_summary=compact_text(combined, 240),
                    signal_type="company_page",
                    why_it_matters="The company page provides public context for fit assessment.",
                )
            )
        weak = "weak" in lower or "hobby" in lower
        profile = ResearchProfile(
            company_name=candidate.company_name,
            website=candidate.website,
            domain=candidate.domain or domain_from_url(candidate.website),
            location=candidate.detected_location or ("Melbourne, Australia" if "melbourne" in lower else "Australia"),
            industry=candidate.detected_industry or ("Logistics" if "logistics" in lower else "Business services"),
            company_size_signal="Multiple operational teams or open roles" if not weak else "No scale signal found",
            growth_signal="Hiring and expansion signal" if "hiring" in lower or "growth" in lower else None,
            hiring_signal="Public role or source mentions operations hiring" if "hiring" in lower or "operations" in lower else None,
            automation_pain_signals=[
                "manual reporting across operations",
                "support queue triage",
            ]
            if not weak
            else [],
            likely_tools_or_systems=["CRM", "spreadsheets"] if not weak else [],
            likely_departments_with_pain=["Operations", "Customer Support"] if not weak else [],
            evidence_items=evidence if not weak else evidence[:1],
            source_urls=[candidate.source_url] + ([candidate.website] if candidate.website else []),
            confidence=0.82 if not weak else 0.25,
            disqualifiers=[] if not weak else ["Weak evidence"],
            notes="Fake model fixture profile for deterministic tests.",
            trace_id=candidate.trace_id,
        )
        return profile

    def score_profile(self, profile: ResearchProfile, run_config: RunConfig) -> ScoreResult:
        self._record_fake_call(profile.model_dump_json())
        if profile.company_name in self.fixture_data.get("scores", {}):
            score = ScoreResult.model_validate(self.fixture_data["scores"][profile.company_name])
            return enforce_save_threshold(score, run_config)
        return fallback_score_profile(profile, run_config)

    def generate_outreach(self, profile: ResearchProfile, score: ScoreResult) -> OutreachDraft:
        self._record_fake_call(profile.model_dump_json() + score.model_dump_json())
        if profile.company_name in self.fixture_data.get("outreach", {}):
            return OutreachDraft.model_validate(self.fixture_data["outreach"][profile.company_name])
        return fallback_outreach(profile, score)

    def decide_next_action(self, state: RunState, config: Any) -> NextAction:
        self._record_fake_call(state.model_dump_json())
        if state.budget_remaining.stop_recommended:
            return NextAction(
                action="stop",
                reasoning=f"Budget guard recommends stopping: {state.budget_remaining.stop_reason}.",
            )
        return heuristic_next_action(state)

    def critic_review(self, recent_leads: list[dict[str, Any]], current_rubric: str, current_pitch_style: str) -> CritiqueResult:
        self._record_fake_call(json.dumps({"leads": recent_leads, "rubric": current_rubric, "style": current_pitch_style}, ensure_ascii=True))
        critic_fixture = self.fixture_data.get("critic")
        if critic_fixture:
            return CritiqueResult.model_validate(critic_fixture)
        return CritiqueResult(
            verdict="keep",
            file_changed=None,
            reasoning="Fake critic reviewed the recent leads and found no repeated pitch or scoring pattern that justified a rewrite.",
            new_content=None,
        )

    def _record_fake_call(self, prompt: str) -> None:
        if self.budget:
            self.budget.record_model_call(max(1, len(prompt) // 4), 150)


def _extract_response_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    return str(response)


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for key, value in list(node.items()):
                node[key] = visit(value)
        elif isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(json.loads(json.dumps(schema)))
