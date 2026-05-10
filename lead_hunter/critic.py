from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .models import now_utc
from .utils import append_jsonl, compact_text


REPO_ROOT = Path(__file__).resolve().parent.parent
RUBRIC_PATH = REPO_ROOT / "rubric.md"
PITCH_STYLE_PATH = REPO_ROOT / "pitch_style.md"

CRITIC_MODEL_ENV_VAR = "OPENAI_MODEL_CRITIC"
CRITIC_DEFAULT_MODEL = "gpt-5.4"
CRITIC_MAX_OUTPUT_TOKENS = 1500

CRITIC_SYSTEM_PROMPT = """You are an adversarial reviewer of an autonomous lead-generation agent's recent output. Your job is to find ONE concrete pattern of weakness in either the scoring rubric or the pitch style, propose a tightened version of ONE file, and explain your reasoning in 2-4 sentences.
You are NOT adversarial-by-default. Most pitches do not need rewriting. Return verdict="keep" UNLESS:

3+ recent pitches share a near-identical opener phrase (>=4 words verbatim across pitches)
2+ recent saved leads have score >= 70 but the pitch is unmistakably generic ("your operations" rather than a named workflow, no evidence URL or quote)
Scoring shows clear miscalibration (e.g., A-tier leads with confidence < 0.6, suggesting rubric is too permissive, or B/A-tier leads with no specific operational pain identified)
Pitches consistently fail to recommend a specific agent type from the allowed list

When you do propose a change, change ONE thing. Do not rewrite both rubric.md and pitch_style.md in the same call. Set file_changed to either "rubric.md" or "pitch_style.md", and provide the FULL new file content in new_content. The other file is untouched.
When proposing changes, preserve the file's structure and the "Notes" section. Append a brief Notes line indicating what version this is and what changed.
If you return verdict="keep", set file_changed=null and new_content=null. Provide reasoning explaining what you reviewed and why it was acceptable. This is a valid and important outcome -- do not invent problems.

This conservatism is critical. We want 1-3 meaningful Critic-driven rewrites over a 24-hour run, not 20 superficial ones. A Critic that says "keep" 60% of the time is doing its job correctly."""


class CritiqueResult(BaseModel):
    verdict: Literal["keep", "update"]
    file_changed: Literal["rubric.md", "pitch_style.md"] | None = None
    reasoning: str
    new_content: str | None = None


def _load_rubric(path: Path = RUBRIC_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _load_pitch_style(path: Path = PITCH_STYLE_PATH) -> str:
    return path.read_text(encoding="utf-8")


def run_critic(
    recent_leads: list[dict[str, Any]],
    current_rubric: str,
    current_pitch_style: str,
    model_client: Any,
) -> CritiqueResult:
    if not hasattr(model_client, "critic_review"):
        raise TypeError("model_client must implement critic_review")
    trimmed_leads = [_critic_lead_view(lead) for lead in recent_leads[-10:]]
    return model_client.critic_review(trimmed_leads, current_rubric, current_pitch_style)


def apply_critique_result(
    result: CritiqueResult,
    trigger_lead_count: int,
    recent_leads: list[dict[str, Any]],
    learnings_path: str | Path,
    rubric_path: str | Path = RUBRIC_PATH,
    pitch_style_path: str | Path = PITCH_STYLE_PATH,
) -> dict[str, Any]:
    learnings_path = Path(learnings_path)
    rubric_path = Path(rubric_path)
    pitch_style_path = Path(pitch_style_path)
    file_changed = result.file_changed if result.verdict == "update" else None
    diff_payload: dict[str, str] | None = None

    if result.verdict == "update":
        if file_changed not in {"rubric.md", "pitch_style.md"}:
            raise ValueError("Critic update must specify file_changed as rubric.md or pitch_style.md")
        if not result.new_content or not result.new_content.strip():
            raise ValueError("Critic update must include full new_content")
        target_path = rubric_path if file_changed == "rubric.md" else pitch_style_path
        before = target_path.read_text(encoding="utf-8")
        after = result.new_content
        diff_payload = _short_diff_payload(before, after)
        target_path.write_text(after, encoding="utf-8")

    entry = {
        "timestamp": now_utc().replace("+00:00", "Z"),
        "trigger_lead_count": trigger_lead_count,
        "verdict": result.verdict,
        "file_changed": file_changed,
        "reasoning": result.reasoning,
        "diff": diff_payload,
        "leads_reviewed": [_lead_id(lead) for lead in recent_leads[-10:]],
    }
    append_jsonl(learnings_path, entry)
    return entry


def _critic_lead_view(lead: dict[str, Any]) -> dict[str, Any]:
    return {
        "lead_id": _lead_id(lead),
        "company_name": lead.get("company_name"),
        "score": lead.get("total_score"),
        "tier": lead.get("fit_tier"),
        "outreach_pitch": lead.get("outreach_pitch"),
        "why_hourglass": lead.get("why_hourglass"),
    }


def _lead_id(lead: dict[str, Any]) -> str:
    return str(lead.get("lead_id") or lead.get("agent_trace_id") or lead.get("company_name") or "unknown_lead")


def _short_diff_payload(before: str, after: str) -> dict[str, str]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_start = max(0, i1 - 2)
        before_end = min(len(before_lines), max(i2, i1 + 1) + 2)
        after_start = max(0, j1 - 2)
        after_end = min(len(after_lines), max(j2, j1 + 1) + 2)
        return {
            "before_excerpt": compact_text("\n".join(before_lines[before_start:before_end]), 1200),
            "after_excerpt": compact_text("\n".join(after_lines[after_start:after_end]), 1200),
        }
    return {"before_excerpt": "", "after_excerpt": ""}
