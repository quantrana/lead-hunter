from __future__ import annotations

from lead_hunter.budget import Budget
from lead_hunter.models import RunConfig


def test_budget_stops_on_spend_cap():
    budget = Budget(model_name="gpt-5.4")
    budget.record_model_call(input_tokens=1_000_000, output_tokens=1_000_000)
    should_stop, reason = budget.should_stop(RunConfig(max_spend_usd=0.01))
    assert should_stop is True
    assert reason == "spend_cap"
