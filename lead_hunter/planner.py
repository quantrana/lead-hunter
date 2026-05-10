from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import AppConfig
from .utils import compact_text, safe_json_loads


PLANNER_MODEL = "gpt-5.4-mini"
PLANNER_MODEL_ENV_VAR = "OPENAI_MODEL_PLANNER"
PLANNER_MAX_OUTPUT_TOKENS = 300


class SourceRunState(BaseModel):
    source_id: str
    last_used_at: str | None = None
    candidates_yielded: int = 0
    leads_yielded: int = 0
    last_error: str | None = None
    times_touched: int = 0
    zero_candidate_runs: int = 0


class BudgetRemaining(BaseModel):
    spend_usd: float = 0.0
    spend_remaining_usd: float = 0.0
    spend_remaining_ratio: float = 1.0
    hours_elapsed: float = 0.0
    hours_remaining: float = 0.0
    hours_remaining_ratio: float = 1.0
    leads_remaining: int = 0
    leads_remaining_ratio: float = 1.0
    iterations_since_last_lead: int = 0
    model_calls: int = 0
    web_fetches: int = 0
    stop_recommended: bool = False
    stop_reason: str = ""


class RunState(BaseModel):
    sources: list[SourceRunState] = Field(default_factory=list)
    candidates_discovered_pending_research: int = 0
    research_profiles_pending_score: int = 0
    candidate_ids_pending_research: list[str] = Field(default_factory=list)
    research_profile_ids_pending_score: list[str] = Field(default_factory=list)
    recent_rejection_reasons: list[str] = Field(default_factory=list)
    leads_saved_count: int = 0
    leads_target: int = 0
    last_critic_lead_count: int = 0
    budget_remaining: BudgetRemaining
    learnings_summary: list[str] = Field(default_factory=list)


class NextAction(BaseModel):
    action: Literal["discover", "research", "score_and_save", "reflect", "stop"]
    source_id: str | None = None
    candidate_id: str | None = None
    research_profile_id: str | None = None
    reasoning: str


def planner_system_prompt() -> str:
    return (
        "You are Lead Hunter's planner. Choose exactly one next action for an autonomous "
        "prospecting agent. Pipeline stages flow: source -> discovered candidate -> researched profile -> saved lead. "
        "Available actions: discover from one configured source, research one discovered candidate, "
        "score_and_save one researched profile, reflect on saved leads, "
        "or stop. Return only the structured NextAction.\n"
        "State fields:\n"
        "- candidates_discovered_pending_research is the count of candidates in DISCOVERED state.\n"
        "- research_profiles_pending_score is the count of profiles in RESEARCHED state.\n"
        "- candidate_ids_pending_research lists candidate IDs that can be researched.\n"
        "- research_profile_ids_pending_score lists research profile IDs that can be scored and saved.\n"
        "Action mapping:\n"
        "- discover: use when no pipeline work is pending and at least one useful source remains. Include source_id.\n"
        "- research: use when candidates_discovered_pending_research > 0. Include candidate_id from candidate_ids_pending_research.\n"
        "- score_and_save: use when research_profiles_pending_score > 0. Include research_profile_id from research_profile_ids_pending_score.\n"
        "- reflect: use when leads_saved is a multiple of 5 and critic feedback is due.\n"
        "- stop: use when complete, exhausted, or stuck.\n"
        "If candidates_discovered_pending_research > 0, the next action for ONE of them is 'research'. "
        "If research_profiles_pending_score > 0, the next action for ONE of them is 'score_and_save'. "
        "You must include candidate_id for research or research_profile_id for score_and_save. Pick the first ID from the corresponding list in state if you have no other preference.\n"
        "Guardrails:\n"
        "- You may not loop forever. If pending work is empty AND no source has been touched in the last 10 minutes, prefer 'reflect' or 'stop'.\n"
        "- Do not call discover on a source that returned 0 candidates twice in a row this run.\n"
        "- Prefer research/score over discover once 30+ candidates exist.\n"
        "- If a source has been touched 3+ times in this run, deprioritize it. Prefer untouched sources or sources with the highest leads_yielded so far.\n"
        "- If a candidate's research returned confidence < 0.4 (no resolvable domain), do not waste a score_and_save action on it; mark it disqualified at the planner level and move on.\n"
        "- Trigger reflect when leads_saved is a multiple of 5 AND leads_saved >= last_critic_lead_count + 2. Otherwise prefer pipeline progress (research/score_and_save/discover) over reflection.\n"
        "- When budget remaining < 30% (spend, hours, OR leads), prefer score_and_save and reflect over discover. Do not start new sources late in the run.\n"
        "- If you cannot identify a useful next action, candidates_discovered_pending_research is 0, research_profiles_pending_score is 0, and all sources have been touched, return action='stop'. Stopping early on completion is correct."
    )


def decide_next_action(state: RunState, config: AppConfig, model_client: Any | None = None) -> NextAction:
    if state.budget_remaining.stop_recommended:
        return NextAction(
            action="stop",
            reasoning=f"Budget guard recommends stopping: {state.budget_remaining.stop_reason}.",
        )
    if state.leads_saved_count >= state.leads_target > 0:
        return NextAction(action="stop", reasoning="The saved lead target has been reached.")
    if model_client and hasattr(model_client, "decide_next_action"):
        action = model_client.decide_next_action(state, config)
    else:
        action = heuristic_next_action(state)
    return normalize_next_action(action, state)


def heuristic_next_action(state: RunState) -> NextAction:
    if state.research_profiles_pending_score > 0:
        return NextAction(
            action="score_and_save",
            research_profile_id=state.research_profile_ids_pending_score[0] if state.research_profile_ids_pending_score else None,
            reasoning="A researched candidate is waiting, so scoring and saving is the highest-leverage next step.",
        )
    if state.candidates_discovered_pending_research > 0:
        return NextAction(
            action="research",
            candidate_id=state.candidate_ids_pending_research[0] if state.candidate_ids_pending_research else None,
            reasoning="There are discovered candidates waiting for public evidence research.",
        )
    if should_reflect(state):
        return NextAction(
            action="reflect",
            reasoning="Saved lead count is at a critic trigger point and enough new leads have accumulated since the last critic run.",
        )

    available_sources = [source for source in state.sources if source.zero_candidate_runs < 2]
    if not available_sources:
        return NextAction(action="stop", reasoning="No pending work remains and all useful sources are exhausted.")

    untouched = [source for source in available_sources if source.times_touched == 0]
    if untouched:
        selected = sorted(untouched, key=lambda source: (-source.leads_yielded, source.source_id))[0]
        return NextAction(
            action="discover",
            source_id=selected.source_id,
            reasoning="Untouched sources should be sampled before revisiting sources that have already run.",
        )

    if all(source.times_touched > 0 for source in state.sources):
        return NextAction(
            action="stop",
            reasoning="No pending work remains and every configured source has already been touched.",
        )

    underused = [source for source in available_sources if source.times_touched < 3]
    if underused:
        selected = sorted(underused, key=lambda source: (source.times_touched, -source.leads_yielded, source.source_id))[0]
        return NextAction(
            action="discover",
            source_id=selected.source_id,
            reasoning="This source has not been overused and has the best balance of freshness and prior yield.",
        )

    productive = [source for source in available_sources if source.leads_yielded > 0]
    if productive:
        selected = sorted(productive, key=lambda source: (-source.leads_yielded, source.times_touched, source.source_id))[0]
        return NextAction(
            action="discover",
            source_id=selected.source_id,
            reasoning="All sources have been touched, so the planner is revisiting the source with the strongest lead yield.",
        )

    return NextAction(action="stop", reasoning="No pending work remains and every source has already been touched.")


def normalize_next_action(action: NextAction, state: RunState) -> NextAction:
    if state.research_profiles_pending_score > 0:
        profile_id = action.research_profile_id or (state.research_profile_ids_pending_score[0] if state.research_profile_ids_pending_score else None)
        if action.action != "score_and_save" or not profile_id:
            return NextAction(
                action="score_and_save",
                research_profile_id=profile_id,
                reasoning=f"Planner guard selected score_and_save because researched profile {profile_id} is pending; scoring researched profiles takes precedence over further research or discovery.",
            )
        return action.model_copy(update={"research_profile_id": profile_id})
    if state.candidates_discovered_pending_research > 0:
        candidate_id = action.candidate_id or (state.candidate_ids_pending_research[0] if state.candidate_ids_pending_research else None)
        if action.action != "research" or not candidate_id:
            return NextAction(
                action="research",
                candidate_id=candidate_id,
                reasoning=f"Planner guard selected research because discovered candidate {candidate_id} is pending; research must happen before scoring or new discovery.",
            )
        return action.model_copy(update={"candidate_id": candidate_id})
    if should_reflect(state) and action.action != "reflect":
        return NextAction(
            action="reflect",
            reasoning=f"Planner guard selected reflect because leads_saved={state.leads_saved_count} and last_critic_lead_count={state.last_critic_lead_count}.",
        )
    return action


def should_reflect(state: RunState) -> bool:
    return (
        state.leads_saved_count >= 5
        and state.leads_saved_count % 5 == 0
        and state.leads_saved_count >= state.last_critic_lead_count + 2
    )


def read_learnings_summary(path: str | Path = "learnings.jsonl", limit: int = 3) -> list[str]:
    learnings_path = Path(path)
    if not learnings_path.exists():
        return []
    lines = [line for line in learnings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries: list[str] = []
    for line in lines[-limit:]:
        payload = safe_json_loads(line, {})
        if isinstance(payload, dict):
            reasoning = payload.get("reasoning") or payload.get("summary") or payload.get("message")
            if reasoning:
                summaries.append(compact_text(str(reasoning), 240))
                continue
        summaries.append(compact_text(line, 240))
    return summaries


def planner_user_payload(state: RunState, config: AppConfig) -> str:
    payload = {
        "run_state": state.model_dump(),
        "run_settings": {
            "max_leads": config.run.max_leads,
            "min_score": config.run.min_score,
            "allow_c_tier": config.run.allow_c_tier,
            "target_country": config.run.target_country,
            "target_city_preference": config.run.target_city_preference,
        },
    }
    return json.dumps(payload, ensure_ascii=True)
