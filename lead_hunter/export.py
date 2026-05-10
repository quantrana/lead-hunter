from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "rank",
    "total_score",
    "fit_tier",
    "company_name",
    "website",
    "domain",
    "location",
    "industry",
    "company_size_signal",
    "source_type",
    "source_url",
    "evidence_urls",
    "pain_signals",
    "recommended_agent_type",
    "why_hourglass",
    "outreach_subject",
    "outreach_pitch",
    "suggested_next_step",
    "confidence",
    "evidence_grounding",
    "risks_or_uncertainties",
    "last_checked_at",
    "agent_trace_id",
]


def export_csv(leads: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(leads, key=lambda lead: (-int(lead.get("total_score", 0)), lead.get("company_name", "")))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, lead in enumerate(ranked, start=1):
            row = {column: lead.get(column, "") for column in CSV_COLUMNS}
            row["rank"] = index
            row["evidence_urls"] = " | ".join(lead.get("evidence_urls") or [])
            row["pain_signals"] = " | ".join(lead.get("pain_signals") or [])
            writer.writerow(row)
