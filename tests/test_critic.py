from __future__ import annotations

import json
from pathlib import Path

from lead_hunter.critic import CritiqueResult, apply_critique_result, run_critic


class StaticCriticClient:
    def __init__(self, result: CritiqueResult) -> None:
        self.result = result

    def critic_review(self, recent_leads, current_rubric, current_pitch_style):
        return self.result


def _sample_leads() -> list[dict]:
    return [
        {
            "lead_id": "lead_1",
            "company_name": "Bega Group",
            "total_score": 81,
            "fit_tier": "A",
            "outreach_pitch": "Bega's manufacturing footprint suggests compliance reporting workflows could be reviewed by an Internal Ops Router.",
            "why_hourglass": "Multi-site manufacturing creates repeatable operations workflows.",
        },
        {
            "lead_id": "lead_2",
            "company_name": "CH Racking",
            "total_score": 74,
            "fit_tier": "B",
            "outreach_pitch": "Saw your team is hiring around warehouse coordination; an Internal Ops Router could prepare exception summaries.",
            "why_hourglass": "Warehouse coordination is a named operational workflow.",
        },
        {
            "lead_id": "lead_3",
            "company_name": "Kingz Container Crew",
            "total_score": 70,
            "fit_tier": "B",
            "outreach_pitch": "With your recent automation hire, container crew scheduling looks like a candidate for routing and review.",
            "why_hourglass": "Scheduling and exception handling are agent-shaped workflows.",
        },
    ]


def test_critic_returns_keep_when_pitches_are_diverse(tmp_path: Path):
    rubric_path = tmp_path / "rubric.md"
    pitch_path = tmp_path / "pitch_style.md"
    learnings_path = tmp_path / "learnings.jsonl"
    rubric_path.write_text("rubric v0", encoding="utf-8")
    pitch_path.write_text("pitch style v0", encoding="utf-8")
    result = run_critic(
        _sample_leads(),
        "rubric v0",
        "pitch style v0",
        StaticCriticClient(
            CritiqueResult(
                verdict="keep",
                file_changed=None,
                reasoning="The pitches use varied openers and name specific workflows, so no rewrite is justified.",
                new_content=None,
            )
        ),
    )
    entry = apply_critique_result(result, 3, _sample_leads(), learnings_path, rubric_path, pitch_path)

    assert rubric_path.read_text(encoding="utf-8") == "rubric v0"
    assert pitch_path.read_text(encoding="utf-8") == "pitch style v0"
    assert entry["verdict"] == "keep"
    logged = json.loads(learnings_path.read_text(encoding="utf-8").strip())
    assert logged["verdict"] == "keep"
    assert logged["diff"] is None


def test_critic_rewrites_pitch_style_when_pattern_detected(tmp_path: Path):
    rubric_path = tmp_path / "rubric.md"
    pitch_path = tmp_path / "pitch_style.md"
    learnings_path = tmp_path / "learnings.jsonl"
    rubric_path.write_text("rubric v0", encoding="utf-8")
    pitch_path.write_text("pitch style v0\n- Old opener guidance", encoding="utf-8")
    weak_leads = [
        {
            "lead_id": "lead_1",
            "company_name": "A",
            "total_score": 72,
            "fit_tier": "B",
            "outreach_pitch": "I noticed you are hiring and your operations could use AI.",
            "why_hourglass": "Potential fit.",
        },
        {
            "lead_id": "lead_2",
            "company_name": "B",
            "total_score": 74,
            "fit_tier": "B",
            "outreach_pitch": "I noticed you are hiring and your operations could use AI.",
            "why_hourglass": "Potential fit.",
        },
        {
            "lead_id": "lead_3",
            "company_name": "C",
            "total_score": 75,
            "fit_tier": "B",
            "outreach_pitch": "I noticed you are hiring and your operations could use AI.",
            "why_hourglass": "Potential fit.",
        },
    ]
    new_style = "pitch style v1\n- Require opener variety and a named workflow.\nNotes:\nv1 - tightened opener variety."
    result = run_critic(
        weak_leads,
        "rubric v0",
        "pitch style v0",
        StaticCriticClient(
            CritiqueResult(
                verdict="update",
                file_changed="pitch_style.md",
                reasoning="Three recent pitches share the same opener and describe generic operations, so pitch style needs one targeted tightening.",
                new_content=new_style,
            )
        ),
    )
    entry = apply_critique_result(result, 3, weak_leads, learnings_path, rubric_path, pitch_path)

    assert pitch_path.read_text(encoding="utf-8") == new_style
    assert rubric_path.read_text(encoding="utf-8") == "rubric v0"
    assert entry["verdict"] == "update"
    assert entry["file_changed"] == "pitch_style.md"
    assert entry["diff"]["before_excerpt"]
    assert entry["diff"]["after_excerpt"]


def test_critic_only_changes_one_file_per_call(tmp_path: Path):
    rubric_path = tmp_path / "rubric.md"
    pitch_path = tmp_path / "pitch_style.md"
    learnings_path = tmp_path / "learnings.jsonl"
    rubric_path.write_text("rubric should stay", encoding="utf-8")
    pitch_path.write_text("pitch style old", encoding="utf-8")
    result = CritiqueResult(
        verdict="update",
        file_changed="pitch_style.md",
        reasoning="Only the pitch style file was selected for this one-change update.",
        new_content="pitch style new\nNotes:\nv1 - one file changed.",
    )

    apply_critique_result(result, 5, _sample_leads(), learnings_path, rubric_path, pitch_path)

    assert rubric_path.read_text(encoding="utf-8") == "rubric should stay"
    assert pitch_path.read_text(encoding="utf-8").startswith("pitch style new")
