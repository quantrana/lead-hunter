from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from agent import LeadHunterAgent, main, research_candidate
from lead_hunter.config import load_config
from lead_hunter.export import CSV_COLUMNS
from lead_hunter.models import CandidateSignal, EvidenceItem, ResearchProfile, ScoreResult
from lead_hunter.research import resolve_domain
from lead_hunter.scoring import enforce_save_threshold, validate_outreach
from lead_hunter.storage import Storage
from lead_hunter.utils import domain_from_url, normalize_company_name


FIXTURE_CONFIG = Path("tests/fixtures/test_config.yaml")
OUTPUT_DIR = Path("tests/fixtures/tmp_outputs")


@pytest.fixture(autouse=True)
def clean_outputs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def test_config_parsing():
    config = load_config(FIXTURE_CONFIG)
    assert config.model.provider == "openai"
    assert config.run.output_dir == str(OUTPUT_DIR)
    assert {source.type for source in config.sources} >= {"rss", "webpage", "manual_company_seed"}


def test_url_and_candidate_normalization():
    assert domain_from_url("https://www.Example.com:443/path") == "example.com"
    assert normalize_company_name("The Example Pty Ltd") == "example"


def test_research_profile_validation():
    profile = ResearchProfile(
        company_name="Aussie Logistics Co",
        evidence_items=[
            EvidenceItem(
                url="https://example.test",
                title="Evidence",
                quote_or_summary="Public evidence mentions manual reporting.",
                signal_type="workflow",
                why_it_matters="Manual reporting can be automated with review.",
            )
        ],
        confidence=0.8,
    )
    assert profile.evidence_items[0].signal_type == "workflow"


def test_score_threshold_logic():
    config = load_config(FIXTURE_CONFIG)
    score = ScoreResult(
        total_score=65,
        fit_tier="C",
        recommended_agent_type="Reporting Agent",
        pain_summary="Some reporting pain.",
        why_hourglass="Maybe relevant.",
        strongest_evidence="Evidence",
        risks_or_uncertainties="Weak.",
        confidence=0.5,
        should_save=True,
    )
    enforced = enforce_save_threshold(score, config.run)
    assert enforced.should_save is False
    assert enforced.rejection_reason


def test_sqlite_persistence_and_domain_dedupe():
    storage = Storage(OUTPUT_DIR / "lead_hunter.sqlite")
    storage.init_db(reset=True)
    agent = LeadHunterAgent(FIXTURE_CONFIG, test_mode=True)
    agent.init_state(reset=True)
    agent.run_once()
    first_count = agent.storage.get_summary().saved_leads
    agent.run_once()
    second_count = agent.storage.get_summary().saved_leads
    assert first_count == 1
    assert second_count == 1


def test_agent_once_generates_csv_html_logs_and_rejects_weak_leads():
    exit_code = main(["once", "--config", str(FIXTURE_CONFIG), "--test-mode"])
    assert exit_code == 0
    assert (OUTPUT_DIR / "leads.csv").exists()
    assert (OUTPUT_DIR / "leads.html").exists()
    assert (OUTPUT_DIR / "run_log.jsonl").exists()
    assert (OUTPUT_DIR / "errors.jsonl").exists()
    csv_text = (OUTPUT_DIR / "leads.csv").read_text(encoding="utf-8")
    html_text = (OUTPUT_DIR / "leads.html").read_text(encoding="utf-8")
    assert "Aussie Logistics Co" in csv_text
    assert "Tiny Hobby" not in csv_text
    assert "Lead Hunter does not send emails" in html_text
    assert "Internal Ops Router" in html_text
    assert all(column in csv_text.splitlines()[0] for column in CSV_COLUMNS)


def test_status_export_and_render_commands():
    assert main(["once", "--config", str(FIXTURE_CONFIG), "--test-mode"]) == 0
    assert main(["status", "--config", str(FIXTURE_CONFIG), "--test-mode"]) == 0
    assert main(["export", "--config", str(FIXTURE_CONFIG), "--test-mode"]) == 0
    assert main(["render", "--config", str(FIXTURE_CONFIG), "--test-mode"]) == 0


def test_production_requires_openai_key(monkeypatch):
    monkeypatch.setenv("LEAD_HUNTER_SKIP_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    assert main(["once", "--config", str(FIXTURE_CONFIG)]) == 2


def test_invalid_model_json_handling_logged_not_crashing():
    assert main(["once", "--config", str(FIXTURE_CONFIG), "--test-mode"]) == 0
    errors = (OUTPUT_DIR / "errors.jsonl").read_text(encoding="utf-8")
    assert "Invalid JSON Pty" in errors
    assert "Fake invalid JSON response after retry" in errors


def test_outreach_pitch_validation_grounded():
    agent = LeadHunterAgent(FIXTURE_CONFIG, test_mode=True)
    agent.init_state(reset=True)
    agent.run_once()
    lead = agent.storage.get_leads()[0]
    assert "Aussie Logistics Co" in lead["outreach_subject"]
    assert "operations automation" in lead["outreach_pitch"]
    assert "we helped" not in lead["outreach_pitch"].lower()
    assert "send" not in lead["outreach_pitch"].lower() or "email" not in lead["outreach_pitch"].lower()


def test_trace_logging_complete():
    assert main(["once", "--config", str(FIXTURE_CONFIG), "--test-mode"]) == 0
    log_text = (OUTPUT_DIR / "run_log.jsonl").read_text(encoding="utf-8")
    for step in ["source_selected", "candidates_discovered", "candidate_researched", "candidate_scored", "outreach_generated", "lead_saved", "dashboard_updated"]:
        assert step in log_text


def test_resolve_domain_strips_suffixes_and_slugifies(monkeypatch):
    class FakeHeadResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.headers = {}

    def fake_head(url, **kwargs):
        if "omggroup.com.au" in url:
            return FakeHeadResponse(200)
        raise OSError("not found")

    monkeypatch.setattr("lead_hunter.research.httpx.head", fake_head)
    assert resolve_domain("OMG Group Pty Ltd") == "omggroup.com.au"


def test_research_candidate_fetches_multiple_pages(monkeypatch):
    class FakeGetResponse:
        def __init__(self, url: str, title: str, body: str):
            self.status_code = 200
            self.url = url
            self.headers = {"content-type": "text/html"}
            self.text = f"<html><head><title>{title}</title></head><body>{body}</body></html>"

    def fake_get(url, **kwargs):
        if url.endswith("/"):
            return FakeGetResponse(url, "OMG Group", "OMG Group is an Australian operations business with customer support and reporting workflows.")
        if url.endswith("/about"):
            return FakeGetResponse(url, "About OMG Group", "About OMG Group. Melbourne team supporting national operations, finance, and onboarding.")
        if url.endswith("/careers"):
            return FakeGetResponse(url, "Careers", "Careers include operations, automation, reporting, and customer success roles.")
        return FakeGetResponse(url, "Other", "Other")

    monkeypatch.setattr("lead_hunter.research.httpx.get", fake_get)
    monkeypatch.setattr("lead_hunter.research.time.sleep", lambda seconds: None)
    agent = LeadHunterAgent(FIXTURE_CONFIG, test_mode=True)
    agent.init_state(reset=True)
    candidate = CandidateSignal(
        company_name="OMG Group Pty Ltd",
        website="https://omggroup.com.au",
        domain="omggroup.com.au",
        source_url="https://seek.example/omg",
        source_type="job_board_url",
        signal_text="Operations Manager at OMG Group Pty Ltd mentions reporting and customer operations.",
        signal_reason="Test job listing",
    )
    profile = research_candidate(agent, candidate)
    signal_types = {item.signal_type for item in profile.evidence_items}
    assert len(profile.evidence_items) >= 4
    assert {"homepage", "about", "careers"}.issubset(signal_types)


def test_research_falls_back_when_domain_unresolvable(monkeypatch):
    def fake_head(url, **kwargs):
        raise OSError("not found")

    monkeypatch.setattr("lead_hunter.research.httpx.head", fake_head)
    agent = LeadHunterAgent(FIXTURE_CONFIG, test_mode=True)
    agent.init_state(reset=True)
    candidate = CandidateSignal(
        company_name="No Domain Pty Ltd",
        source_url="https://seek.example/no-domain",
        source_type="job_board_url",
        signal_text="Operations role mentions workflow improvement.",
        signal_reason="Test job listing",
    )
    profile = research_candidate(agent, candidate)
    assert profile.confidence <= 0.3
    assert "no_resolvable_domain" in profile.disqualifiers
