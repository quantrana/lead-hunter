from __future__ import annotations

from pathlib import Path

from agent import LeadHunterAgent
from lead_hunter.budget import Budget
from lead_hunter.config import load_config
from lead_hunter.model_client import FakeModelClient
from lead_hunter.planner import BudgetRemaining, RunState, SourceRunState, decide_next_action


FIXTURE_CONFIG = Path("tests/fixtures/test_config.yaml")


def test_planner_chooses_stop_when_budget_exhausted():
    config = load_config(FIXTURE_CONFIG)
    budget = Budget(model_name="gpt-5.4")
    budget.record_model_call(input_tokens=1_000_000, output_tokens=1_000_000)
    should_stop, reason = budget.should_stop(config)
    state = RunState(
        sources=[SourceRunState(source_id="fixture_rss_jobs")],
        candidates_discovered_pending_research=0,
        research_profiles_pending_score=0,
        leads_saved_count=0,
        leads_target=config.run.max_leads,
        budget_remaining=BudgetRemaining(
            spend_usd=budget.estimated_cost_usd,
            stop_recommended=True,
            stop_reason=reason or "spend_cap",
        ),
    )
    action = decide_next_action(state, config, FakeModelClient())
    assert action.action == "stop"
    assert "spend_cap" in action.reasoning


def test_planner_deprioritizes_overused_sources():
    config = load_config(FIXTURE_CONFIG)
    state = RunState(
        sources=[
            SourceRunState(source_id="overused_source", candidates_yielded=12, leads_yielded=2, times_touched=4),
            SourceRunState(source_id="untouched_source", candidates_yielded=0, leads_yielded=0, times_touched=0),
        ],
        candidates_discovered_pending_research=0,
        research_profiles_pending_score=0,
        leads_saved_count=0,
        leads_target=config.run.max_leads,
        budget_remaining=BudgetRemaining(),
    )
    action = decide_next_action(state, config, FakeModelClient())
    assert action.action == "discover"
    assert action.source_id == "untouched_source"


def test_planner_scores_researched_profile_before_more_research():
    config = load_config(FIXTURE_CONFIG)
    state = RunState(
        sources=[SourceRunState(source_id="fixture_rss_jobs")],
        candidates_discovered_pending_research=2,
        research_profiles_pending_score=1,
        candidate_ids_pending_research=["10", "11"],
        research_profile_ids_pending_score=["99"],
        leads_saved_count=0,
        leads_target=config.run.max_leads,
        budget_remaining=BudgetRemaining(),
    )
    action = decide_next_action(state, config, FakeModelClient())
    assert action.action == "score_and_save"
    assert action.research_profile_id == "99"


def test_planner_research_action_includes_candidate_id():
    config = load_config(FIXTURE_CONFIG)
    state = RunState(
        sources=[SourceRunState(source_id="fixture_rss_jobs")],
        candidates_discovered_pending_research=2,
        research_profiles_pending_score=0,
        candidate_ids_pending_research=["10", "11"],
        leads_saved_count=0,
        leads_target=config.run.max_leads,
        budget_remaining=BudgetRemaining(),
    )
    action = decide_next_action(state, config, FakeModelClient())
    assert action.action == "research"
    assert action.candidate_id == "10"


def test_planner_triggers_reflect_at_lead_multiple_of_5():
    config = load_config(FIXTURE_CONFIG)
    state = RunState(
        sources=[SourceRunState(source_id="fixture_rss_jobs", times_touched=1)],
        candidates_discovered_pending_research=0,
        research_profiles_pending_score=0,
        leads_saved_count=5,
        leads_target=50,
        last_critic_lead_count=0,
        budget_remaining=BudgetRemaining(),
    )
    action = decide_next_action(state, config, FakeModelClient())
    assert action.action == "reflect"


def test_agent_builds_planner_state_from_storage():
    agent = LeadHunterAgent(FIXTURE_CONFIG, test_mode=True)
    agent.init_state(reset=True)
    state = agent.build_run_state()
    assert [source.source_id for source in state.sources] == [
        "fixture_rss_jobs",
        "fixture_company_page",
        "fixture_manual_seed",
    ]
