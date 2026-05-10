from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


MODEL_PRICES_USD_PER_MILLION = {
    "gpt-5.4": {"input": 1.25, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


@dataclass
class Budget:
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    web_fetches: int = 0
    leads_saved: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    consecutive_no_lead_iterations: int = 0
    iterations: int = 0

    @property
    def elapsed_hours(self) -> float:
        try:
            started = datetime.fromisoformat(self.started_at)
        except ValueError:
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - started).total_seconds() / 3600)

    @property
    def estimated_cost_usd(self) -> float:
        prices = self._prices()
        input_cost = self.input_tokens / 1_000_000 * prices["input"]
        output_cost = self.output_tokens / 1_000_000 * prices["output"]
        return round(input_cost + output_cost, 6)

    def record_model_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.model_calls += 1
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))

    def record_web_fetch(self, count: int = 1) -> None:
        self.web_fetches += max(0, int(count))

    def record_lead_saved(self) -> None:
        self.leads_saved += 1
        self.consecutive_no_lead_iterations = 0

    def record_iteration(self, lead_saved: bool = False) -> None:
        self.iterations += 1
        if lead_saved:
            self.consecutive_no_lead_iterations = 0
        else:
            self.consecutive_no_lead_iterations += 1

    def sync_leads_saved(self, leads_saved: int) -> None:
        self.leads_saved = max(self.leads_saved, int(leads_saved))

    def should_stop(self, config: Any) -> tuple[bool, str]:
        run_config = getattr(config, "run", config)
        if self.estimated_cost_usd >= float(getattr(run_config, "max_spend_usd", 25.0)):
            return True, "spend_cap"
        if self.elapsed_hours >= float(getattr(run_config, "max_hours", 36)):
            return True, "time_cap"
        if self.leads_saved >= int(getattr(run_config, "max_leads", 50)):
            return True, "lead_cap"
        if self.consecutive_no_lead_iterations >= int(getattr(run_config, "stall_limit", 30)):
            return True, "stall_limit"
        return False, ""

    def status_lines(self) -> list[str]:
        return [
            f"Budget estimated spend: ${self.estimated_cost_usd:.4f}",
            f"Budget model calls: {self.model_calls}",
            f"Budget tokens: input={self.input_tokens}, output={self.output_tokens}",
            f"Budget web fetches: {self.web_fetches}",
            f"Budget leads saved: {self.leads_saved}",
            f"Budget elapsed hours: {self.elapsed_hours:.2f}",
            f"Budget iterations since last lead: {self.consecutive_no_lead_iterations}",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_calls": self.model_calls,
            "web_fetches": self.web_fetches,
            "leads_saved": self.leads_saved,
            "started_at": self.started_at,
            "consecutive_no_lead_iterations": self.consecutive_no_lead_iterations,
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Budget":
        return cls(
            model_name=str(data.get("model_name") or ""),
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            model_calls=int(data.get("model_calls") or 0),
            web_fetches=int(data.get("web_fetches") or 0),
            leads_saved=int(data.get("leads_saved") or 0),
            started_at=str(data.get("started_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
            consecutive_no_lead_iterations=int(data.get("consecutive_no_lead_iterations") or 0),
            iterations=int(data.get("iterations") or 0),
        )

    def _prices(self) -> dict[str, float]:
        normalized = self.model_name.lower()
        for model_name in sorted(MODEL_PRICES_USD_PER_MILLION, key=len, reverse=True):
            if normalized.startswith(model_name):
                return MODEL_PRICES_USD_PER_MILLION[model_name]
        return MODEL_PRICES_USD_PER_MILLION["gpt-5.4-mini"]
