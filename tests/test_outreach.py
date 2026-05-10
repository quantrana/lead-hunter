from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent import LeadHunterAgent, save_qualified_lead
from lead_hunter.models import CandidateSignal, EvidenceItem, OutreachDraft, ResearchProfile, ScoreResult
from lead_hunter.scoring import prepare_outreach_for_save, validate_outreach


FIXTURE_CONFIG = Path("tests/fixtures/test_config.yaml")
OUTPUT_DIR = Path("tests/fixtures/tmp_outputs")


@pytest.fixture(autouse=True)
def clean_outputs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def _profile() -> ResearchProfile:
    return ResearchProfile(
        company_name="Bega Group",
        website="https://begagroup.com.au",
        domain="begagroup.com.au",
        automation_pain_signals=["manufacturing ops scheduling"],
        evidence_items=[
            EvidenceItem(
                url="https://begagroup.com.au/about",
                title="About Bega",
                quote_or_summary="Bega Group operates large scale manufacturing sites across Australia with compliance reporting workflows.",
                signal_type="about",
                why_it_matters="Scale and compliance create repeatable operational work.",
            )
        ],
        source_urls=["https://begagroup.com.au/about"],
        confidence=0.85,
        trace_id="lh_test",
    )


def _score() -> ScoreResult:
    return ScoreResult(
        total_score=74,
        fit_tier="B",
        recommended_agent_type="Internal Ops Router",
        pain_summary="manufacturing ops scheduling and compliance reporting",
        why_hourglass="Hourglass could build a focused internal ops agent.",
        strongest_evidence="Bega Group operates large scale manufacturing sites across Australia with compliance reporting workflows.",
        risks_or_uncertainties="Needs human review.",
        confidence=0.82,
        should_save=True,
    )


def test_validate_outreach_passes_with_url_in_pitch():
    outreach = OutreachDraft(
        outreach_subject="AI agent idea for Bega Group",
        outreach_pitch="Bega Group's page at https://begagroup.com.au/about points to manufacturing ops scheduling. An Internal Ops Router could help prepare exception summaries for review. Worth a quick look?",
    )
    result = validate_outreach(_profile(), outreach)
    assert result.ok
    assert result.evidence_grounding == "strong"


def test_validate_outreach_passes_with_verbatim_quote():
    outreach = OutreachDraft(
        outreach_subject="AI agent idea for Bega Group",
        outreach_pitch="Bega Group operates large scale manufacturing sites across Australia, which suggests compliance reporting workflows. An Internal Ops Router could help prepare review packs. Worth 20 minutes?",
    )
    result = validate_outreach(_profile(), outreach)
    assert result.ok
    assert result.evidence_grounding == "strong"


def test_validate_outreach_demotes_tier_on_weak_grounding():
    outreach = OutreachDraft(
        outreach_subject="AI agent idea for Bega Group",
        outreach_pitch="Bega Group may have a workflow where an Internal Ops Router could help the team prepare better handoffs. Worth a quick look?",
    )
    final_outreach, final_score, result = prepare_outreach_for_save(_profile(), _score(), outreach)
    assert result.ok
    assert result.evidence_grounding == "weak"
    assert final_score.fit_tier == "C"
    assert final_outreach.outreach_pitch == outreach.outreach_pitch


def test_validate_outreach_handles_insufficient_evidence_token():
    agent = LeadHunterAgent(FIXTURE_CONFIG, test_mode=True)
    agent.init_state(reset=True)
    candidate = CandidateSignal(
        company_name="Bega Group",
        website="https://begagroup.com.au",
        domain="begagroup.com.au",
        source_url="https://example.test/source",
        source_type="manual_company_seed",
        signal_text="Manual seed",
        signal_reason="Test",
        trace_id="lh_test_insufficient",
    )
    outreach = OutreachDraft(
        outreach_subject="AI agent idea for Bega Group",
        outreach_pitch="INSUFFICIENT_EVIDENCE",
    )
    final_outreach, final_score, result = prepare_outreach_for_save(_profile(), _score(), outreach)
    saved = save_qualified_lead(agent, candidate, _profile(), final_score, final_outreach)
    lead = agent.storage.get_leads()[0]
    assert saved is True
    assert result.evidence_grounding == "insufficient"
    assert final_score.fit_tier == "C"
    assert "INSUFFICIENT_EVIDENCE" not in lead["outreach_pitch"]
    assert lead["evidence_grounding"] == "insufficient"


def test_pitch_truncation_at_sentence_boundary():
    long_pitch = (
        "Bega Group operates large scale manufacturing sites across Australia with compliance reporting workflows. "
        + "This sentence contains useful but repetitive context about scheduling, reporting, approval queues, and review packs. " * 20
        + "Final sentence should be removed because it is too far beyond the cap."
    )
    outreach = OutreachDraft(
        outreach_subject="AI agent idea for Bega Group",
        outreach_pitch=long_pitch,
    )
    result = validate_outreach(_profile(), outreach)
    assert len(result.outreach.outreach_pitch) <= 800
    assert result.outreach.outreach_pitch.endswith(".")
    assert "Final sentence should be removed" not in result.outreach.outreach_pitch
