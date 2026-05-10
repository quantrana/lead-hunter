from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from lead_hunter.budget import Budget
from lead_hunter.config import load_config, sorted_sources
from lead_hunter.critic import _load_pitch_style, _load_rubric, apply_critique_result, run_critic
from lead_hunter.export import export_csv
from lead_hunter.fetch import Fetcher
from lead_hunter.model_client import BaseModelClient, FakeModelClient, ModelClientError, OpenAIModelClient
from lead_hunter.models import (
    AppConfig,
    CandidateSignal,
    ErrorEvent,
    OutreachDraft,
    QualifiedLead,
    ResearchProfile,
    ScoreResult,
    SourceConfig,
    TraceEvent,
    now_utc,
)
from lead_hunter.planner import (
    BudgetRemaining,
    NextAction,
    RunState,
    SourceRunState,
    decide_next_action as planner_decide_next_action,
    read_learnings_summary,
)
from lead_hunter.render import render_dashboard
from lead_hunter.research import discovery_evidence_page, fetch_company_pages, page_dict_to_evidence, resolve_domain
from lead_hunter.scoring import fallback_outreach, fallback_score_profile, prepare_outreach_for_save
from lead_hunter.storage import Storage
from lead_hunter.traces import TraceLogger
from lead_hunter.utils import (
    compact_text,
    domain_from_url,
    find_signal_terms,
    has_australia_hint,
    normalize_domain,
    normalize_company_name,
    trace_id,
)


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LeadHunterAgent:
    def __init__(self, config_path: str | Path, test_mode: bool = False) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.test_mode = test_mode
        self.output_dir = Path(self.config.run.output_dir)
        self.db_path = self.output_dir / "lead_hunter.sqlite"
        self.csv_path = self.output_dir / "leads.csv"
        self.html_path = self.output_dir / "leads.html"
        self.run_log_path = self.output_dir / "run_log.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.learnings_path = self.output_dir / "learnings.jsonl"
        self.budget = Budget(model_name=os.getenv(self.config.model.model_env_var, "fake" if test_mode else ""))
        self.storage = Storage(self.db_path)
        self.fetcher = Fetcher(self.config.run, test_mode=test_mode, budget=self.budget)
        self.trace_logger = TraceLogger(self.storage, self.run_log_path, self.errors_path)
        self._model_client: BaseModelClient | None = None
        self.stop_requested = False

    def _build_model_client(self) -> BaseModelClient:
        if self.test_mode:
            fixture = Path("tests/fixtures/fake_model_outputs.json")
            return FakeModelClient(fixture if fixture.exists() else None, budget=self.budget)
        return OpenAIModelClient(
            model_env_var=self.config.model.model_env_var,
            temperature=self.config.model.temperature,
            max_output_tokens=self.config.model.max_output_tokens,
            budget=self.budget,
        )

    @property
    def model_client(self) -> BaseModelClient:
        if self._model_client is None:
            self._model_client = self._build_model_client()
        return self._model_client

    def init_state(self, reset: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if reset:
            for path in [self.csv_path, self.html_path, self.run_log_path, self.errors_path, self.learnings_path]:
                if path.exists():
                    path.unlink()
        self.storage.init_db(reset=reset)
        self.storage.upsert_sources(self.config.sources)
        if not reset:
            state = self.storage.load_budget_state()
            if state:
                self.budget = Budget.from_dict(state)
                self.fetcher.budget = self.budget
                self._model_client = None
        self.budget.model_name = self.budget.model_name or os.getenv(self.config.model.model_env_var, "fake" if self.test_mode else "")
        self.budget.sync_leads_saved(self.storage.get_summary().saved_leads)
        self.storage.save_budget(self.budget)
        self.learnings_path.parent.mkdir(parents=True, exist_ok=True)
        self.learnings_path.touch(exist_ok=True)
        self.export_outputs()

    def trace(
        self,
        trace_id_value: str,
        step: str,
        tool_called: str,
        input_summary: str,
        output_summary: str,
        model_reasoning_summary: str = "",
        confidence: float | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.trace_logger.trace(
            TraceEvent(
                trace_id=trace_id_value,
                step=step,
                tool_called=tool_called,
                input_summary=input_summary,
                output_summary=output_summary,
                model_reasoning_summary=model_reasoning_summary,
                confidence=confidence,
                errors=errors or [],
            )
        )

    def error(self, source: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.trace_logger.error(ErrorEvent(source=source, message=message, context=context or {}))

    def export_outputs(self) -> None:
        leads = self.storage.get_leads()
        export_csv(leads, self.csv_path)
        render_dashboard(
            leads,
            self.storage.get_summary(),
            self.html_path,
            config_name=str(self.config_path),
            output_dir=str(self.output_dir),
        )

    def choose_sources(self) -> list[SourceConfig]:
        return sorted_sources(self.config)

    def run_once(self) -> None:
        self.init_state(reset=False)
        self.stop_requested = False
        _ = self.model_client
        max_decisions = max(20, len(self.choose_sources()) * max(4, min(self.config.run.max_candidates_per_source, 5)) * 3)
        self._run_planner_loop(max_decisions=max_decisions)
        self.export_outputs()

    def run_continuous(self, hours: float | None = None, max_leads: int | None = None) -> None:
        self.init_state(reset=False)
        self.stop_requested = False
        _ = self.model_client
        self._run_planner_loop(max_hours=hours, max_leads=max_leads, sleep_between_actions=True)

    def _run_planner_loop(
        self,
        max_decisions: int | None = None,
        max_hours: float | None = None,
        max_leads: int | None = None,
        sleep_between_actions: bool = False,
    ) -> None:
        decisions = 0
        while not self.stop_requested:
            should_stop, reason = self._budget_should_stop(max_hours=max_hours, max_leads=max_leads)
            if should_stop:
                self._stop_for_budget(reason)
                break
            if max_decisions is not None and decisions >= max_decisions:
                break
            state = self.build_run_state(max_hours=max_hours, max_leads=max_leads)
            action = planner_decide_next_action(state, self.config, self.model_client)
            self._trace_action_decision(action, state)
            self.execute_action(action)
            decisions += 1
            self.storage.save_budget(self.budget)
            if sleep_between_actions and not self.stop_requested:
                time.sleep(max(0.0, self.config.run.crawl_delay_seconds))

    def build_run_state(self, max_hours: float | None = None, max_leads: int | None = None) -> RunState:
        self.budget.sync_leads_saved(self.storage.get_summary().saved_leads)
        should_stop, stop_reason = self._budget_should_stop(max_hours=max_hours, max_leads=max_leads)
        spend_cap = max(0.0, float(self.config.run.max_spend_usd))
        spend = self.budget.estimated_cost_usd
        spend_remaining = max(0.0, spend_cap - spend)
        hours_cap = max_hours if max_hours is not None else self.config.run.max_hours
        hours_remaining = max(0.0, float(hours_cap) - self.budget.elapsed_hours)
        target_leads = max_leads if max_leads is not None else self.config.run.max_leads
        leads_remaining = max(0, int(target_leads) - self.budget.leads_saved)
        candidate_stats = self.storage.candidate_source_stats()
        sources: list[SourceRunState] = []
        for source in self.choose_sources():
            source_candidate_stats = candidate_stats.get(source.id, {})
            source_trace_stats = self.storage.source_trace_stats(source.id, since=self.budget.started_at)
            sources.append(
                SourceRunState(
                    source_id=source.id,
                    last_used_at=source_trace_stats.get("last_used_at"),
                    candidates_yielded=int(source_candidate_stats.get("candidates_yielded", 0)),
                    leads_yielded=int(source_candidate_stats.get("leads_yielded", 0)),
                    last_error=source_trace_stats.get("last_error"),
                    times_touched=int(source_trace_stats.get("times_touched", 0)),
                    zero_candidate_runs=int(source_trace_stats.get("zero_candidate_runs", 0)),
                )
            )
        candidate_ids_pending_research = self.storage.get_candidate_ids_by_status("discovered")
        research_profile_ids_pending_score = self.storage.get_pending_research_profile_ids()
        critic_state = self.storage.get_critic_state()
        return RunState(
            sources=sources,
            candidates_discovered_pending_research=len(candidate_ids_pending_research),
            research_profiles_pending_score=len(research_profile_ids_pending_score),
            candidate_ids_pending_research=[str(candidate_id) for candidate_id in candidate_ids_pending_research],
            research_profile_ids_pending_score=[str(profile_id) for profile_id in research_profile_ids_pending_score],
            recent_rejection_reasons=self.storage.recent_rejection_reasons(5),
            leads_saved_count=self.budget.leads_saved,
            leads_target=int(target_leads),
            last_critic_lead_count=int(critic_state.get("last_critic_lead_count") or 0),
            budget_remaining=BudgetRemaining(
                spend_usd=spend,
                spend_remaining_usd=spend_remaining,
                spend_remaining_ratio=(spend_remaining / spend_cap) if spend_cap else 0.0,
                hours_elapsed=self.budget.elapsed_hours,
                hours_remaining=hours_remaining,
                hours_remaining_ratio=(hours_remaining / float(hours_cap)) if hours_cap else 0.0,
                leads_remaining=leads_remaining,
                leads_remaining_ratio=(leads_remaining / int(target_leads)) if target_leads else 0.0,
                iterations_since_last_lead=self.budget.consecutive_no_lead_iterations,
                model_calls=self.budget.model_calls,
                web_fetches=self.budget.web_fetches,
                stop_recommended=should_stop,
                stop_reason=stop_reason,
            ),
            learnings_summary=read_learnings_summary(self.learnings_path),
        )

    def _trace_action_decision(self, action: NextAction, state: RunState) -> None:
        self.trace(
            trace_id(),
            "next_action_decided",
            "decide_next_action",
            f"candidates_pending_research={state.candidates_discovered_pending_research}; profiles_pending_score={state.research_profiles_pending_score}; leads={state.leads_saved_count}/{state.leads_target}",
            f"Planner chose {action.action}"
            + (f" source={action.source_id}" if action.source_id else "")
            + (f" candidate={action.candidate_id}" if action.candidate_id else "")
            + (f" research_profile={action.research_profile_id}" if action.research_profile_id else ""),
            action.reasoning,
            confidence=0.85,
        )

    def execute_action(self, action: NextAction) -> None:
        lead_saved = False
        if action.action == "stop":
            self.stop_requested = True
            self.trace(
                trace_id(),
                "agent_stopped",
                "decide_next_action",
                "Planner returned stop.",
                "Run stopped cleanly.",
                action.reasoning,
                confidence=1.0,
            )
            return
        if action.action == "discover":
            self._execute_discover_action(action)
        elif action.action == "research":
            self._execute_research_action(action)
        elif action.action == "score_and_save":
            lead_saved = self._execute_score_and_save_action(action)
        elif action.action == "reflect":
            self.run_critic_reflection(force=False, planner_reasoning=action.reasoning)
        self._record_iteration_and_check_stop(lead_saved)

    def critic_due(self) -> bool:
        lead_count = self.storage.get_summary().saved_leads
        critic_state = self.storage.get_critic_state()
        last_count = int(critic_state.get("last_critic_lead_count") or 0)
        return lead_count >= 5 and lead_count % 5 == 0 and lead_count >= last_count + 2

    def run_critic_reflection(self, force: bool = False, planner_reasoning: str = "") -> dict[str, Any] | None:
        lead_count = self.storage.get_summary().saved_leads
        if not force and not self.critic_due():
            self.trace(
                trace_id(),
                "reflect_skipped",
                "run_critic",
                f"leads_saved={lead_count}",
                "Critic trigger conditions were not met.",
                planner_reasoning,
                confidence=0.8,
            )
            return None
        recent_leads = self.storage.get_recent_leads(10)
        if not recent_leads:
            self.trace(
                trace_id(),
                "reflect_skipped",
                "run_critic",
                "No saved leads",
                "Critic skipped because there were no saved leads to review.",
                planner_reasoning,
                confidence=0.8,
            )
            return None
        current_rubric = _load_rubric()
        current_pitch_style = _load_pitch_style()
        result = run_critic(recent_leads, current_rubric, current_pitch_style, self.model_client)
        entry = apply_critique_result(
            result,
            trigger_lead_count=lead_count,
            recent_leads=recent_leads,
            learnings_path=self.learnings_path,
        )
        self.storage.save_critic_state(lead_count, entry["timestamp"])
        self.trace(
            trace_id(),
            "reflect",
            "run_critic",
            f"leads_saved={lead_count}; leads_reviewed={len(recent_leads)}",
            f"Critic verdict={result.verdict}; file_changed={result.file_changed}",
            result.reasoning,
            confidence=1.0,
        )
        return entry

    def _execute_discover_action(self, action: NextAction) -> None:
        source = next((item for item in self.choose_sources() if item.id == action.source_id), None)
        if not source:
            self.error("planner", "Planner selected unknown source.", {"source_id": action.source_id})
            self.trace(
                trace_id(),
                "lead_rejected",
                "decide_next_action",
                str(action.source_id),
                "Planner selected an unknown source; no action taken.",
                action.reasoning,
                errors=["unknown_source"],
            )
            return
        candidates = discover_candidates(self, source)
        for candidate in candidates:
            candidate_id, is_new = self.storage.save_candidate(candidate)
            if not is_new:
                self.trace(
                    candidate.trace_id or trace_id(),
                    "candidate_deduped",
                    "discover_candidates",
                    candidate.company_name,
                    "Candidate already seen; skipped duplicate.",
                    confidence=0.9,
                )
            else:
                self.storage.update_candidate_status(candidate_id, "discovered")

    def _execute_research_action(self, action: NextAction) -> None:
        candidate_id = _coerce_candidate_id(action.candidate_id)
        if candidate_id is None:
            self.trace(
                trace_id(),
                "lead_rejected",
                "research_candidate",
                "No pending candidate",
                "Planner requested research but no discovered candidate was pending.",
                action.reasoning,
                confidence=0.5,
            )
            return
        status = self.storage.get_candidate_status(candidate_id)
        if status != "discovered":
            self.trace(
                trace_id(),
                "lead_rejected",
                "research_candidate",
                str(candidate_id),
                f"Planner requested research for candidate outside DISCOVERED state: {status}.",
                action.reasoning,
                confidence=0.5,
            )
            return
        candidate = self.storage.get_candidate(candidate_id)
        if not candidate:
            self.error("planner", "Planner selected missing candidate.", {"candidate_id": candidate_id})
            return
        try:
            profile = research_candidate(self, candidate)
            research_profile_id = self.storage.save_research_profile(candidate_id, profile)
            if profile.confidence < 0.4 and "no_resolvable_domain" in profile.disqualifiers:
                self.storage.update_candidate_status(candidate_id, "disqualified")
                self.trace(
                    candidate.trace_id or trace_id(),
                    "lead_rejected",
                    "decide_next_action",
                    candidate.company_name,
                    "Planner disqualified low-confidence research with no resolvable domain before scoring.",
                    action.reasoning,
                    profile.confidence,
                )
                return
            self.storage.update_candidate_status(candidate_id, "researched")
            self.trace(
                candidate.trace_id or trace_id(),
                "research_profile_ready",
                "research_candidate",
                candidate.company_name,
                f"Research profile {research_profile_id} is ready for score_and_save.",
                action.reasoning,
                profile.confidence,
            )
        except Exception as exc:
            self.storage.update_candidate_status(candidate_id, "error")
            self.error("candidate_research", str(exc), {"company": candidate.company_name, "candidate_id": candidate_id})
            self.trace(
                candidate.trace_id or trace_id(),
                "lead_rejected",
                "research_candidate",
                candidate.company_name,
                "Candidate skipped because research failed.",
                errors=[str(exc)],
            )

    def _execute_score_and_save_action(self, action: NextAction) -> bool:
        research_profile_id = _coerce_candidate_id(action.research_profile_id)
        if research_profile_id is None:
            self.trace(
                trace_id(),
                "lead_rejected",
                "score_candidate",
                "No researched candidate",
                "Planner requested scoring but did not provide a research_profile_id.",
                action.reasoning,
                confidence=0.5,
            )
            return False
        bundle = self.storage.get_research_profile_bundle(research_profile_id)
        if not bundle:
            self.trace(
                trace_id(),
                "lead_rejected",
                "score_candidate",
                str(research_profile_id),
                "Planner selected a missing or already-processed research profile.",
                action.reasoning,
                confidence=0.5,
            )
            return False
        candidate_id, candidate, profile = bundle
        if profile.confidence < 0.4 and "no_resolvable_domain" in profile.disqualifiers:
            self.storage.update_candidate_status(candidate_id, "disqualified")
            self.trace(
                profile.trace_id or candidate.trace_id or trace_id(),
                "lead_rejected",
                "decide_next_action",
                profile.company_name,
                "Planner disqualified low-confidence research with no resolvable domain before scoring.",
                action.reasoning,
                profile.confidence,
            )
            return False
        score = score_candidate(self, profile)
        self.storage.save_score(candidate_id, candidate.trace_id or "", score)
        if score.should_save:
            outreach, final_score = generate_grounded_outreach(self, profile, score)
            saved = save_qualified_lead(self, candidate, profile, final_score, outreach)
            self.storage.update_candidate_status(candidate_id, "saved" if saved else "duplicate_lead")
            if saved:
                self.budget.record_lead_saved()
            return saved
        self.storage.update_candidate_status(candidate_id, "rejected")
        self.trace(
            candidate.trace_id or "",
            "lead_rejected",
            "score_candidate",
            profile.company_name,
            score.rejection_reason or "Rejected",
            score.reasoning_summary,
            score.confidence,
        )
        return False

    def _process_candidates(self, candidates: list[CandidateSignal]) -> None:
        for candidate in candidates:
            leads_before = self.storage.get_summary().saved_leads
            candidate_id, is_new = self.storage.save_candidate(candidate)
            if not is_new:
                self.trace(
                    candidate.trace_id or trace_id(),
                    "candidate_deduped",
                    "discover_candidates",
                    candidate.company_name,
                    "Candidate already seen; skipped duplicate.",
                    confidence=0.9,
                )
                self._record_iteration_and_check_stop(False)
                if self.stop_requested:
                    return
                continue
            if self.storage.candidate_already_processed(candidate_id):
                continue
            try:
                profile = research_candidate(self, candidate)
                self.storage.save_research_profile(candidate_id, profile)
                score = score_candidate(self, profile)
                self.storage.save_score(candidate_id, candidate.trace_id or "", score)
                if score.should_save:
                    outreach, final_score = generate_grounded_outreach(self, profile, score)
                    saved = save_qualified_lead(self, candidate, profile, final_score, outreach)
                    self.storage.update_candidate_status(candidate_id, "saved" if saved else "duplicate_lead")
                    if saved:
                        self.budget.record_lead_saved()
                else:
                    self.storage.update_candidate_status(candidate_id, "rejected")
                    self.trace(
                        candidate.trace_id or "",
                        "lead_rejected",
                        "score_candidate",
                        profile.company_name,
                        score.rejection_reason or "Rejected",
                        score.reasoning_summary,
                        score.confidence,
                    )
            except Exception as exc:
                self.storage.update_candidate_status(candidate_id, "error")
                self.error("candidate_processing", str(exc), {"company": candidate.company_name})
                self.trace(
                    candidate.trace_id or trace_id(),
                    "lead_rejected",
                    "candidate_processing",
                    candidate.company_name,
                    "Candidate skipped because processing failed.",
                    errors=[str(exc)],
                )
            leads_after = self.storage.get_summary().saved_leads
            self._record_iteration_and_check_stop(leads_after > leads_before)
            if self.stop_requested:
                return
            time.sleep(max(0.0, self.config.run.crawl_delay_seconds))

    def _record_iteration_and_check_stop(self, lead_saved: bool) -> None:
        self.budget.record_iteration(lead_saved=lead_saved)
        self.budget.sync_leads_saved(self.storage.get_summary().saved_leads)
        self.storage.save_budget(self.budget)
        should_stop, reason = self._budget_should_stop()
        if should_stop:
            self._stop_for_budget(reason)

    def _budget_should_stop(self, max_hours: float | None = None, max_leads: int | None = None) -> tuple[bool, str]:
        should_stop, reason = self.budget.should_stop(self.config)
        if should_stop:
            return should_stop, reason
        if max_hours is not None and self.budget.elapsed_hours >= max_hours:
            return True, "time_cap"
        if max_leads is not None and self.budget.leads_saved >= max_leads:
            return True, "lead_cap"
        return False, ""

    def _stop_for_budget(self, reason: str) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.storage.save_budget(self.budget)
        self.export_outputs()
        self.trace(
            trace_id(),
            "budget_stop",
            "budget",
            "Budget guard",
            f"Stopping cleanly: {reason}",
            f"Spend=${self.budget.estimated_cost_usd:.4f}; leads={self.budget.leads_saved}; elapsed_hours={self.budget.elapsed_hours:.2f}; no_lead_iterations={self.budget.consecutive_no_lead_iterations}",
            confidence=1.0,
        )
        print(f"Budget guard stopping cleanly: {reason}")


def _coerce_candidate_id(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_candidates(agent: LeadHunterAgent, source: SourceConfig) -> list[CandidateSignal]:
    run_trace_id = trace_id()
    agent.trace(
        run_trace_id,
        "source_selected",
        "discover_candidates",
        source.id,
        f"Selected {source.type} source with priority {source.priority}.",
        confidence=1.0,
    )
    try:
        if source.type == "manual_company_seed":
            candidates = _discover_manual_seed(source)
        elif source.type == "rss":
            candidates = _discover_rss(agent, source)
        else:
            candidates = _discover_webpage(agent, source)
        for candidate in candidates:
            if not candidate.trace_id:
                candidate.trace_id = trace_id()
        before_filter = len(candidates)
        candidates = [candidate for candidate in candidates if _is_likely_company_name(candidate.company_name)]
        filtered = before_filter - len(candidates)
        candidates = candidates[: agent.config.run.max_candidates_per_source]
        agent.trace(
            run_trace_id,
            "candidates_discovered",
            "discover_candidates",
            source.id,
            f"Discovered {len(candidates)} candidate signal(s); filtered {filtered} non-company item(s).",
            "Source parsed with conservative public-signal extraction.",
            confidence=0.75 if candidates else 0.2,
        )
        return candidates
    except Exception as exc:
        agent.error("discover_candidates", str(exc), {"source": source.id})
        agent.trace(
            run_trace_id,
            "candidates_discovered",
            "discover_candidates",
            source.id,
            "Discovery failed; continuing with other sources.",
            errors=[str(exc)],
        )
        return []


def research_candidate(agent: LeadHunterAgent, candidate: CandidateSignal) -> ResearchProfile:
    pages: list[dict[str, str]] = [discovery_evidence_page(candidate)]
    initial_domain = normalize_domain(candidate.domain or candidate.website)
    fixture_placeholder_domain = agent.test_mode and (
        (candidate.website or "").startswith(("fixture://", "file://"))
        or bool(initial_domain and (initial_domain.endswith(".example") or "." not in initial_domain))
    )
    resolved_domain = initial_domain
    if agent.test_mode and (
        (candidate.website or "").startswith(("fixture://", "file://"))
        or (resolved_domain and (resolved_domain.endswith(".example") or "." not in resolved_domain))
    ):
        resolved_domain = None
    if not resolved_domain:
        if agent.test_mode:
            agent.storage.save_cached_domain(candidate.company_name, None)
        else:
            cache_hit, cached_domain = agent.storage.get_cached_domain(candidate.company_name)
            if cache_hit:
                resolved_domain = cached_domain
            else:
                resolved_domain = resolve_domain(candidate.company_name, budget=agent.budget)
                agent.storage.save_cached_domain(candidate.company_name, resolved_domain)
    company_pages: list[dict[str, str]] = []
    if resolved_domain:
        company_pages = fetch_company_pages(
            resolved_domain,
            agent.config.run.user_agent,
            agent.config.run.inner_page_delay_seconds,
            budget=agent.budget,
        )
        pages.extend(company_pages)
    model_candidate = candidate.model_copy(
        update={
            "website": candidate.website or (f"https://{resolved_domain}" if resolved_domain else None),
            "domain": resolved_domain,
        }
    )
    try:
        profile = agent.model_client.research_profile(model_candidate, pages, agent.config.run)
    except ModelClientError as exc:
        agent.error("research_candidate", str(exc), {"company": candidate.company_name})
        raise
    disqualifiers = list(profile.disqualifiers)
    confidence = profile.confidence
    if not resolved_domain and not fixture_placeholder_domain:
        if "no_resolvable_domain" not in disqualifiers:
            disqualifiers.append("no_resolvable_domain")
        confidence = min(confidence, 0.3)
    elif not company_pages:
        if "no_company_pages_fetched" not in disqualifiers:
            disqualifiers.append("no_company_pages_fetched")
        confidence = min(confidence, 0.45)
    if resolved_domain:
        existing_urls = {item.url for item in profile.evidence_items}
        for page in company_pages:
            if page["url"] not in existing_urls:
                profile.evidence_items.append(page_dict_to_evidence(page))
                existing_urls.add(page["url"])
    profile = profile.model_copy(
        update={
            "company_name": profile.company_name or candidate.company_name,
            "website": profile.website or candidate.website or (f"https://{resolved_domain}" if resolved_domain else None),
            "domain": normalize_domain(profile.domain or resolved_domain or candidate.domain or candidate.website),
            "source_urls": list(dict.fromkeys(profile.source_urls + [page["url"] for page in pages])),
            "confidence": confidence,
            "disqualifiers": disqualifiers,
            "trace_id": candidate.trace_id,
        }
    )
    agent.trace(
        candidate.trace_id or "",
        "candidate_researched",
        "research_candidate",
        candidate.company_name,
        f"Extracted {len(profile.evidence_items)} evidence item(s) with confidence {profile.confidence}.",
        f"Research profile used {len(company_pages)} company-owned page(s) plus discovery evidence.",
        profile.confidence,
    )
    agent.trace(
        candidate.trace_id or "",
        "evidence_extracted",
        "research_candidate",
        candidate.company_name,
        "; ".join(item.signal_type for item in profile.evidence_items) or "No evidence items.",
        confidence=profile.confidence,
    )
    return profile


def score_candidate(agent: LeadHunterAgent, research_profile: ResearchProfile) -> ScoreResult:
    try:
        score = agent.model_client.score_profile(research_profile, agent.config.run)
    except ModelClientError as exc:
        agent.error("score_candidate", str(exc), {"company": research_profile.company_name})
        score = fallback_score_profile(research_profile, agent.config.run)
    agent.trace(
        research_profile.trace_id or "",
        "candidate_scored",
        "score_candidate",
        research_profile.company_name,
        f"Score {score.total_score}, tier {score.fit_tier}, should_save={score.should_save}.",
        score.reasoning_summary,
        score.confidence,
    )
    return score


def save_qualified_lead(
    agent: LeadHunterAgent,
    candidate: CandidateSignal,
    profile: ResearchProfile,
    score: ScoreResult,
    outreach: OutreachDraft,
) -> bool:
    lead = QualifiedLead(
        total_score=score.total_score,
        fit_tier=score.fit_tier,
        company_name=profile.company_name,
        website=profile.website,
        domain=normalize_domain(profile.domain or profile.website),
        location=profile.location,
        industry=profile.industry,
        company_size_signal=profile.company_size_signal,
        source_type=candidate.source_type,
        source_url=candidate.source_url,
        evidence_urls=[item.url for item in profile.evidence_items],
        pain_signals=profile.automation_pain_signals,
        recommended_agent_type=score.recommended_agent_type,
        why_hourglass=score.why_hourglass,
        outreach_subject=outreach.outreach_subject,
        outreach_pitch=outreach.outreach_pitch,
        suggested_next_step=outreach.suggested_next_step,
        confidence=score.confidence,
        risks_or_uncertainties=score.risks_or_uncertainties,
        evidence_grounding=getattr(outreach, "evidence_grounding", "strong"),
        agent_trace_id=candidate.trace_id or trace_id(),
        evidence_items=profile.evidence_items,
        strongest_evidence=score.strongest_evidence,
    )
    saved = agent.storage.save_lead(lead)
    agent.export_outputs()
    agent.trace(
        lead.agent_trace_id,
        "lead_saved" if saved else "candidate_deduped",
        "save_qualified_lead",
        lead.company_name,
        "Lead saved and exports updated." if saved else "Lead already existed; exports refreshed.",
        "SQLite, CSV, and HTML are updated without sending outreach.",
        lead.confidence,
    )
    agent.trace(
        lead.agent_trace_id,
        "dashboard_updated",
        "save_qualified_lead",
        lead.company_name,
        "Dashboard and CSV rendered from SQLite state.",
        confidence=1.0,
    )
    return saved


def generate_grounded_outreach(agent: LeadHunterAgent, profile: ResearchProfile, score: ScoreResult) -> tuple[OutreachDraft, ScoreResult]:
    model_failed = False
    model_error = ""
    try:
        outreach = agent.model_client.generate_outreach(profile, score)
    except ModelClientError as exc:
        model_failed = True
        model_error = str(exc)
        agent.error("generate_outreach", str(exc), {"company": profile.company_name})
        outreach = fallback_outreach(profile, score)
    outreach, final_score, validation = prepare_outreach_for_save(profile, score, outreach)
    if validation.hard_errors and not model_failed:
        raise ValueError("; ".join(validation.hard_errors))
    outreach = outreach.model_copy(update={"evidence_grounding": validation.evidence_grounding})
    if validation.warnings:
        agent.trace(
            profile.trace_id or "",
            "outreach_grounding_warning",
            "validate_outreach",
            profile.company_name,
            f"Evidence grounding marked {validation.evidence_grounding}; tier {score.fit_tier}->{final_score.fit_tier}.",
            "; ".join(validation.warnings),
            final_score.confidence,
        )
    agent.trace(
        profile.trace_id or "",
        "outreach_generated",
        "score_candidate",
        profile.company_name,
        outreach.outreach_subject,
        outreach.reasoning_summary,
        final_score.confidence,
        errors=[model_error] if model_failed else [],
    )
    return outreach, final_score


def _discover_manual_seed(source: SourceConfig) -> list[CandidateSignal]:
    candidates = []
    for company in source.companies:
        candidates.append(
            CandidateSignal(
                company_name=company.name,
                website=company.website,
                domain=domain_from_url(company.website),
                source_url=company.website or f"manual://{normalize_company_name(company.name)}",
                source_type=source.type,
                signal_text=company.reason,
                signal_reason="Manual company seed marked for testing/demo source.",
                raw_metadata={"source_id": source.id, "manual_seed": True, "tags": source.tags},
                trace_id=trace_id(),
            )
        )
    return candidates


def _discover_rss(agent: LeadHunterAgent, source: SourceConfig) -> list[CandidateSignal]:
    assert source.url
    raw, error = agent.fetcher.fetch_raw_text(source.url)
    if error:
        agent.error("rss_fetch", error, {"source": source.id, "url": source.url})
        return []
    entries = agent.fetcher.parse_rss(raw, source.url)
    candidates: list[CandidateSignal] = []
    for entry in entries:
        text = compact_text(f"{entry.get('title', '')}. {entry.get('summary', '')}", 1200)
        if not _looks_like_signal(text):
            continue
        company_name = _company_from_rss_entry(entry)
        if not company_name:
            continue
        candidates.append(
            CandidateSignal(
                company_name=company_name,
                website=None,
                domain=None,
                source_url=entry.get("link") or source.url,
                source_type=source.type,
                signal_text=text,
                signal_reason=_signal_reason(text),
                detected_location="Australia" if has_australia_hint(text) else None,
                raw_metadata={"source_id": source.id, "published": entry.get("published", ""), "tags": source.tags},
                trace_id=trace_id(),
            )
        )
    return candidates[:25]


def _discover_webpage(agent: LeadHunterAgent, source: SourceConfig) -> list[CandidateSignal]:
    assert source.url
    page = agent.fetcher.fetch(source.url)
    if page.error:
        agent.error("page_fetch", page.error, {"source": source.id, "url": source.url})
        return []
    candidates: list[CandidateSignal] = []
    if source.type in {"job_board_url", "ats_jobs_url"}:
        candidates.extend(_extract_job_board_candidates(page, source))
        if candidates:
            return candidates[:25]
    text = compact_text(f"{page.title}. {page.text}", 1800)
    if _looks_like_signal(text):
        candidates.append(
            CandidateSignal(
                company_name=_infer_company_name(page.title, page.final_url),
                website=_root_url(page.final_url),
                domain=domain_from_url(page.final_url),
                source_url=page.final_url,
                source_type=source.type,
                signal_text=text,
                signal_reason=_signal_reason(text),
                detected_location="Australia" if has_australia_hint(text) else None,
                raw_metadata={"source_id": source.id, "tags": source.tags, "title": page.title},
                trace_id=trace_id(),
            )
        )
    if source.type in {"company_list_url", "search_url"}:
        for link in _companyish_links(page.links):
            domain = domain_from_url(link)
            if not domain:
                continue
            candidates.append(
                CandidateSignal(
                    company_name=_name_from_domain(domain),
                    website=_root_url(link),
                    domain=domain,
                    source_url=page.final_url,
                    source_type=source.type,
                    signal_text=compact_text(f"Linked from {page.title or page.final_url}. {text}", 1000),
                    signal_reason="Company-like public link found on a relevant source page.",
                    detected_location="Australia" if domain.endswith(".au") or has_australia_hint(text) else None,
                    raw_metadata={"source_id": source.id, "linked_url": link, "tags": source.tags},
                    trace_id=trace_id(),
                )
            )
    deduped: dict[str, CandidateSignal] = {}
    for candidate in candidates:
        key = candidate.domain or normalize_company_name(candidate.company_name)
        deduped[key] = candidate
    return list(deduped.values())[:25]


def _extract_job_board_candidates(page, source: SourceConfig) -> list[CandidateSignal]:
    html = page.html or ""
    soup = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")
    candidates: list[CandidateSignal] = []
    if _is_seek_url(page.final_url):
        candidates.extend(_extract_seek_candidates(soup, page, source))
        return _dedupe_candidate_signals(candidates)
    candidates.extend(_extract_selector_job_candidates(soup, page, source))
    candidates.extend(_extract_text_job_candidates(page, source))
    return _dedupe_candidate_signals(candidates)


def _dedupe_candidate_signals(candidates: list[CandidateSignal]) -> list[CandidateSignal]:
    deduped: dict[str, CandidateSignal] = {}
    for candidate in candidates:
        if not _is_likely_company_name(candidate.company_name):
            continue
        key = f"{normalize_company_name(candidate.company_name)}:{candidate.source_url}"
        deduped[key] = candidate
    return list(deduped.values())


def _extract_seek_candidates(soup: BeautifulSoup, page, source: SourceConfig) -> list[CandidateSignal]:
    candidates: list[CandidateSignal] = []
    company_links = soup.select('[data-automation="jobCompany"][data-type="company"], [data-automation="jobCompany"]')
    for link in company_links[:30]:
        company = link.get_text(" ", strip=True)
        if not company:
            label = link.get("aria-label") or link.get("title") or ""
            company = re.sub(r"^Jobs at\s+", "", label, flags=re.IGNORECASE).strip()
        card = _seek_job_container(link)
        card_text = compact_text(card.get_text(" ", strip=True) if card else page.text, 1200)
        if "this is a" not in card_text.lower() or " job" not in card_text.lower():
            continue
        title_node = link.find_previous(attrs={"data-automation": "jobTitle"}) or (card.select_one('[data-automation="jobTitle"]') if card else None)
        title = title_node.get_text(" ", strip=True) if title_node else ""
        if not title or len(title) > 90 or "classification" in title.lower() or "listed" in title.lower():
            title = _seek_title_from_text(card_text, company)
        location = _extract_location_from_text(card_text)
        industry = _extract_industry_from_text(card_text)
        candidates.append(
            CandidateSignal(
                company_name=company,
                website=None,
                domain=None,
                source_url=page.final_url,
                source_type=source.type,
                signal_text=compact_text(f"{title}. {card_text}", 1200),
                signal_reason="SEEK listing names a specific hiring company and operational role.",
                detected_location=location,
                detected_industry=industry,
                raw_metadata={"source_id": source.id, "tags": source.tags, "job_title": title, "extractor": "seek_jobCompany"},
                trace_id=trace_id(),
            )
        )
    return candidates


def _seek_job_container(link):
    parent = link
    for _ in range(12):
        parent = parent.parent
        if parent is None:
            return link.find_parent("div")
        text = parent.get_text(" ", strip=True)
        if "This is a" in text and (" job" in text.lower()) and len(text) > 120:
            return parent
    return link.find_parent("div")


def _extract_selector_job_candidates(soup: BeautifulSoup, page, source: SourceConfig) -> list[CandidateSignal]:
    candidates: list[CandidateSignal] = []
    selectors = [
        (".posting", ".posting-title h5, .posting-title a", ".company, .posting-company"),
        (".opening", "a, h3", ".company, .department"),
        ("[data-ui='job'], [data-ui='job-card']", "a, h3", "[data-ui='company']"),
        ("article[data-automation='normalJob'], article[data-automation='job-card']", "[data-automation='jobTitle']", "[data-automation='jobCompany']"),
    ]
    board_company = _company_from_ats_board_url(page.final_url)
    for card_selector, title_selector, company_selector in selectors:
        for card in soup.select(card_selector)[:30]:
            company_node = card.select_one(company_selector)
            title_node = card.select_one(title_selector)
            company = company_node.get_text(" ", strip=True) if company_node else board_company
            title = title_node.get_text(" ", strip=True) if title_node else ""
            if not company:
                continue
            text = compact_text(card.get_text(" ", strip=True), 1200)
            candidates.append(
                CandidateSignal(
                    company_name=company,
                    website=_root_url(page.final_url),
                    domain=domain_from_url(page.final_url),
                    source_url=page.final_url,
                    source_type=source.type,
                    signal_text=compact_text(f"{title}. {text}", 1200),
                    signal_reason="Public job board listing names a specific hiring company.",
                    detected_location=_extract_location_from_text(text),
                    detected_industry=_extract_industry_from_text(text),
                    raw_metadata={"source_id": source.id, "tags": source.tags, "job_title": title, "extractor": card_selector},
                    trace_id=trace_id(),
                )
            )
    return candidates


def _extract_text_job_candidates(page, source: SourceConfig) -> list[CandidateSignal]:
    candidates: list[CandidateSignal] = []
    text = page.text
    pattern = re.compile(
        r"(?P<title>[A-Z][A-Za-z0-9&/,.()' +#-]{3,90})\s+at\s+(?P<company>[A-Z][A-Za-z0-9&.'’ PtyLtdLimitedGroupCompanyCo-]{2,80})\s+This is a",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        title = compact_text(match.group("title"), 120)
        company = compact_text(match.group("company"), 100)
        start = max(0, match.start() - 250)
        end = min(len(text), match.end() + 800)
        snippet = compact_text(text[start:end], 1200)
        candidates.append(
            CandidateSignal(
                company_name=company,
                website=None,
                domain=None,
                source_url=page.final_url,
                source_type=source.type,
                signal_text=snippet,
                signal_reason="Job listing text names a specific hiring company.",
                detected_location=_extract_location_from_text(snippet),
                detected_industry=_extract_industry_from_text(snippet),
                raw_metadata={"source_id": source.id, "tags": source.tags, "job_title": title, "extractor": "text_at_company"},
                trace_id=trace_id(),
            )
        )
    return candidates[:30]


def _looks_like_signal(text: str) -> bool:
    if not text.strip():
        return False
    terms = find_signal_terms(text)
    return bool(terms) or has_australia_hint(text)


def _signal_reason(text: str) -> str:
    terms = find_signal_terms(text)
    if terms:
        return f"Public source mentions {', '.join(terms[:5])}."
    if has_australia_hint(text):
        return "Public source has Australia relevance."
    return "Public source contains company context."


def _infer_company_name(title: str, url: str) -> str:
    cleaned = re.sub(r"\s+", " ", title or "").strip()
    patterns = [
        r"\bat\s+([A-Z][A-Za-z0-9&.' -]{2,60})",
        r"^([A-Z][A-Za-z0-9&.' -]{2,60})\s+(?:is|has|raises|hiring|seeks|launches)\b",
        r"^([A-Z][A-Za-z0-9&.' -]{2,60})\s*[-|:]",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return _clean_company_name(match.group(1))
    if cleaned:
        first = re.split(r"\s[-|:]\s", cleaned)[0]
        if 2 < len(first) < 80:
            return _clean_company_name(first)
    domain = domain_from_url(url)
    return _name_from_domain(domain or "Unknown Company")


def _company_from_rss_entry(entry: dict[str, Any]) -> str | None:
    for key in ("author", "dc_creator", "publisher"):
        value = compact_text(str(entry.get(key, "") or ""), 100)
        if _is_likely_company_name(value):
            return _clean_company_name(value)
    source = entry.get("source") or {}
    if isinstance(source, dict):
        source_title = compact_text(str(source.get("title", "") or ""), 100)
        if _is_likely_company_name(source_title):
            return _clean_company_name(source_title)
    link = entry.get("link", "")
    if str(link).startswith("fixture://"):
        title_company = _infer_company_name(entry.get("title", ""), link)
        return title_company if _is_likely_company_name(title_company) else None
    domain = domain_from_url(link)
    if domain and not any(domain.endswith(blocked) for blocked in ["news.google.com", "google.com", "feedburner.com"]) and domain != "tests":
        name = _name_from_domain(domain)
        if _is_likely_company_name(name):
            return name
    return None


NON_COMPANY_PATTERNS = [
    r"^how to\b",
    r"^\d+\s+(?:tips|ways|strategies|keys|steps)\b",
    r"\bmarket overview\b",
    r"^guide to\b",
    r"^top\s+\d+\b",
    r"\bwhy you\b",
    r"\bwhat is\b",
    r"\bbudget\s+\d{4}\b",
    r"\bnewsletter\b",
    r"\bpodcast\b",
    r"\bcase study\b",
]


def _is_likely_company_name(name: str | None) -> bool:
    if not name:
        return False
    cleaned = re.sub(r"\s+", " ", name).strip()
    if len(cleaned) < 2 or len(cleaned) > 90:
        return False
    lower = cleaned.lower().strip(" -:|")
    if lower in {"seek", "unknown company", "private advertiser", "au", "hk", "id", "my", "nz", "ph", "sg", "th"}:
        return False
    if any(re.search(pattern, lower) for pattern in NON_COMPANY_PATTERNS):
        return False
    if lower.endswith((" jobs", " job", " careers")):
        return False
    if not re.search(r"[a-zA-Z]", cleaned):
        return False
    return True


def _title_near_company(text: str) -> str:
    match = re.search(r"(?:Listed .*? )?([A-Z][A-Za-z0-9&/,.()' +#-]{3,90})\s+at\s+", text)
    return compact_text(match.group(1), 100) if match else ""


def _seek_title_from_text(text: str, company: str) -> str:
    marker = f" at {company}"
    if marker in text:
        before = text.split(marker, 1)[0]
        before = re.split(r"\bListed\s+[^.]{0,60}?\s+ago\s+", before)[-1]
        before = re.split(r"\bclassification:\s*", before, flags=re.IGNORECASE)[0]
        title = before.strip(" -:|")
        if 2 < len(title) <= 90:
            return compact_text(title, 90)
    return _title_near_company(text)


def _extract_location_from_text(text: str) -> str | None:
    match = re.search(
        r"\b(?:[A-Z][A-Za-z .'-]{1,35},\s*)?(?:All\s+)?(Melbourne|Sydney|Brisbane|Perth|Adelaide|Canberra|Gold Coast|Hobart|Darwin)\s+(VIC|NSW|QLD|WA|SA|ACT|TAS|NT)\b",
        text,
    )
    if match:
        location = match.group(0)
        location = re.sub(r"^.*?\b(?:job|role)\s+", "", location, flags=re.IGNORECASE)
        return location.strip()
    if has_australia_hint(text):
        return "Australia"
    return None


def _extract_industry_from_text(text: str) -> str | None:
    match = re.search(r"classification:\s*([^()]+)\(", text, re.IGNORECASE)
    if match:
        return compact_text(match.group(1), 80)
    for industry in ["Manufacturing, Transport & Logistics", "Administration & Office Support", "Engineering", "Healthcare", "Accounting", "Information & Communication Technology"]:
        if industry.lower() in text.lower():
            return industry
    return None


def _is_seek_url(url: str) -> bool:
    domain = domain_from_url(url) or ""
    return domain.endswith("seek.com") or domain.endswith("seek.com.au")


def _company_from_ats_board_url(url: str) -> str | None:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    if "greenhouse.io" in domain or "lever.co" in domain or "workable.com" in domain:
        return re.sub(r"[-_]+", " ", parts[0]).title()
    return None


def _clean_company_name(value: str) -> str:
    value = re.sub(r"\b(is|has|are|was|were)$", "", value.strip(), flags=re.IGNORECASE)
    return value.strip(" -:|")[:80] or "Unknown Company"


def _name_from_domain(domain: str) -> str:
    stem = domain.split(".")[0]
    return re.sub(r"[-_]+", " ", stem).title()


def _root_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme == "file":
        return url
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_public_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https", "file", "fixture"}


def _companyish_links(links: list[str]) -> list[str]:
    blocked = {"facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com", "youtube.com"}
    result = []
    for link in links:
        domain = domain_from_url(link)
        if not domain or any(domain.endswith(item) for item in blocked):
            continue
        result.append(link)
    return result[:20]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lead Hunter autonomous prospecting agent")
    parser.add_argument("command", choices=["init", "once", "run", "render", "status", "export", "reset"])
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--test-mode", action="store_true", help="Use deterministic fixtures and fake model client")
    parser.add_argument("--hours", type=float, default=None, help="Override max run hours")
    parser.add_argument("--max-leads", type=int, default=None, help="Override max saved leads")
    parser.add_argument("--force-critic", action="store_true", help="After a once/run command, force a Critic review even below the normal trigger")
    return parser


def main(argv: list[str] | None = None) -> int:
    if os.getenv("LEAD_HUNTER_SKIP_DOTENV") != "1":
        load_dotenv_if_present()
    args = build_parser().parse_args(argv)
    if args.test_mode:
        os.environ.setdefault("LEAD_HUNTER_ENV", "test")
    try:
        if args.command == "reset":
            config = load_config(args.config)
            output_dir = Path(config.run.output_dir)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            agent = LeadHunterAgent(args.config, test_mode=args.test_mode)
            agent.init_state(reset=True)
            print(f"Reset state in {agent.output_dir}")
            return 0
        agent = LeadHunterAgent(args.config, test_mode=args.test_mode)
        if args.command == "init":
            agent.init_state(reset=True)
            print(f"Initialized Lead Hunter state in {agent.output_dir}")
        elif args.command == "once":
            agent.run_once()
            if args.force_critic:
                entry = agent.run_critic_reflection(force=True, planner_reasoning="Forced by --force-critic")
                if entry:
                    print(f"Forced critic verdict: {entry['verdict']}")
            print(f"Completed one Lead Hunter cycle. Leads saved: {agent.storage.get_summary().saved_leads}")
        elif args.command == "run":
            agent.run_continuous(hours=args.hours, max_leads=args.max_leads)
            if args.force_critic:
                entry = agent.run_critic_reflection(force=True, planner_reasoning="Forced by --force-critic")
                if entry:
                    print(f"Forced critic verdict: {entry['verdict']}")
            print(f"Run complete or max leads reached. Leads saved: {agent.storage.get_summary().saved_leads}")
        elif args.command == "render":
            agent.init_state(reset=False)
            agent.export_outputs()
            print(f"Rendered dashboard: {agent.html_path}")
        elif args.command == "status":
            agent.init_state(reset=False)
            print("\n".join(agent.storage.status_lines() + agent.budget.status_lines()))
        elif args.command == "export":
            agent.init_state(reset=False)
            agent.export_outputs()
            print(f"Exported CSV: {agent.csv_path}")
        return 0
    except ModelClientError as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Lead Hunter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
