from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SourceType = Literal[
    "rss",
    "webpage",
    "job_board_url",
    "ats_jobs_url",
    "search_url",
    "company_list_url",
    "manual_company_seed",
]

FitTier = Literal["A", "B", "C", "Reject"]

RecommendedAgentType = Literal[
    "Email Triage Agent",
    "Invoice Chasing Agent",
    "Sales Ops Agent",
    "Support Agent",
    "Onboarding Agent",
    "Reporting Agent",
    "Document Extraction Agent",
    "Order Processing Agent",
    "Knowledge Brain",
    "Compliance Workflow Agent",
    "Recruiting Ops Agent",
    "Customer Success Agent",
    "Internal Ops Router",
    "Other",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class RunConfig(BaseModel):
    max_hours: float = 36
    max_leads: int = 50
    min_score: int = 70
    allow_c_tier: bool = False
    crawl_delay_seconds: float = 8
    request_timeout_seconds: float = 20
    max_candidates_per_source: int = 25
    inner_page_delay_seconds: float = 1.5
    max_spend_usd: float = 25.0
    stall_limit: int = 30
    output_dir: str = "outputs"
    target_country: str = "Australia"
    target_city_preference: str = "Melbourne"
    user_agent: str = "LeadHunter/1.0 research bot for public business information"


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: Literal["openai"] = "openai"
    model_env_var: str = "OPENAI_MODEL"
    temperature: float = 0.2
    max_output_tokens: int = 2000


class ManualCompanySeed(BaseModel):
    name: str
    website: str | None = None
    reason: str = "Manual seed for testing only"


class SourceConfig(BaseModel):
    id: str
    type: SourceType
    url: str | None = None
    priority: int = 100
    tags: list[str] = Field(default_factory=list)
    companies: list[ManualCompanySeed] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_payload(self) -> "SourceConfig":
        if self.type == "manual_company_seed":
            if not self.companies:
                raise ValueError("manual_company_seed source requires companies")
        elif not self.url:
            raise ValueError(f"{self.type} source requires url")
        return self


class AppConfig(BaseModel):
    run: RunConfig
    model: ModelConfig
    sources: list[SourceConfig]

    @field_validator("sources")
    @classmethod
    def sources_are_present(cls, sources: list[SourceConfig]) -> list[SourceConfig]:
        if not sources:
            raise ValueError("at least one source is required")
        return sources


class FetchedPage(BaseModel):
    url: str
    final_url: str
    title: str = ""
    text: str = ""
    html: str = ""
    links: list[str] = Field(default_factory=list)
    fetched_at: str = Field(default_factory=now_utc)
    status_code: int | None = None
    error: str | None = None


class CandidateSignal(BaseModel):
    company_name: str
    website: str | None = None
    domain: str | None = None
    source_url: str
    source_type: SourceType
    signal_text: str
    signal_reason: str
    detected_location: str | None = None
    detected_industry: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class EvidenceItem(BaseModel):
    url: str
    title: str = ""
    quote_or_summary: str
    signal_type: str
    why_it_matters: str

    @field_validator("quote_or_summary", "signal_type", "why_it_matters")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("evidence text fields must not be empty")
        return value.strip()


class ResearchProfile(BaseModel):
    company_name: str
    website: str | None = None
    domain: str | None = None
    location: str | None = None
    industry: str | None = None
    company_size_signal: str | None = None
    growth_signal: str | None = None
    hiring_signal: str | None = None
    automation_pain_signals: list[str] = Field(default_factory=list)
    likely_tools_or_systems: list[str] = Field(default_factory=list)
    likely_departments_with_pain: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    disqualifiers: list[str] = Field(default_factory=list)
    notes: str | None = None
    trace_id: str | None = None


class ScoreResult(BaseModel):
    total_score: int = Field(ge=0, le=100)
    fit_tier: FitTier
    recommended_agent_type: RecommendedAgentType
    pain_summary: str
    why_hourglass: str
    strongest_evidence: str
    risks_or_uncertainties: str
    confidence: float = Field(ge=0.0, le=1.0)
    should_save: bool
    rejection_reason: str | None = None
    reasoning_summary: str = ""

    @model_validator(mode="after")
    def reject_consistency(self) -> "ScoreResult":
        if self.fit_tier == "Reject" and self.should_save:
            raise ValueError("Reject tier cannot be saved")
        if not self.should_save and self.fit_tier != "Reject" and not self.rejection_reason:
            self.rejection_reason = "Below configured save threshold"
        return self


class OutreachDraft(BaseModel):
    outreach_subject: str
    outreach_pitch: str
    suggested_next_step: str = "Human review, then decide whether to contact the company."
    reasoning_summary: str = ""
    evidence_grounding: Literal["strong", "weak", "insufficient"] = "strong"


class QualifiedLead(BaseModel):
    total_score: int
    fit_tier: FitTier
    company_name: str
    website: str | None = None
    domain: str | None = None
    location: str | None = None
    industry: str | None = None
    company_size_signal: str | None = None
    source_type: str
    source_url: str
    evidence_urls: list[str] = Field(default_factory=list)
    pain_signals: list[str] = Field(default_factory=list)
    recommended_agent_type: RecommendedAgentType
    why_hourglass: str
    outreach_subject: str
    outreach_pitch: str
    suggested_next_step: str
    confidence: float = Field(ge=0.0, le=1.0)
    risks_or_uncertainties: str
    evidence_grounding: Literal["strong", "weak", "insufficient"] = "strong"
    last_checked_at: str = Field(default_factory=now_utc)
    agent_trace_id: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    strongest_evidence: str = ""


class TraceEvent(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    trace_id: str
    timestamp: str = Field(default_factory=now_utc)
    step: str
    tool_called: str
    input_summary: str
    output_summary: str
    model_reasoning_summary: str = ""
    confidence: float | None = None
    errors: list[str] = Field(default_factory=list)


class ErrorEvent(BaseModel):
    timestamp: str = Field(default_factory=now_utc)
    source: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    total_candidates: int = 0
    researched_candidates: int = 0
    saved_leads: int = 0
    a_tier_leads: int = 0
    b_tier_leads: int = 0
    c_tier_leads: int = 0
    rejected_candidates: int = 0
    average_score: float = 0.0
    first_trace_at: str | None = None
    last_trace_at: str | None = None
    run_duration: str = "0m"
