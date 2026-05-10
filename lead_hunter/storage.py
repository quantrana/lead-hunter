from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .models import CandidateSignal, ErrorEvent, QualifiedLead, ResearchProfile, RunSummary, ScoreResult, TraceEvent
from .utils import normalize_company_name, normalize_domain, safe_json_dumps, safe_json_loads


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self, reset: bool = False) -> None:
        if reset and self.db_path.exists():
            self.db_path.unlink()
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    url TEXT,
                    priority INTEGER DEFAULT 100,
                    tags_json TEXT DEFAULT '[]',
                    last_selected_at TEXT
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    normalized_company_name TEXT NOT NULL,
                    website TEXT,
                    domain TEXT,
                    source_url TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    signal_text TEXT NOT NULL,
                    signal_reason TEXT NOT NULL,
                    detected_location TEXT,
                    detected_industry TEXT,
                    raw_metadata_json TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'discovered',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(normalized_company_name, source_url)
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_domain ON candidates(domain);
                CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);

                CREATE TABLE IF NOT EXISTS research_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    trace_id TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                );

                CREATE TABLE IF NOT EXISTS evidence_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    trace_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    quote_or_summary TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    why_it_matters TEXT NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                );

                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    trace_id TEXT NOT NULL,
                    score_json TEXT NOT NULL,
                    total_score INTEGER NOT NULL,
                    fit_tier TEXT NOT NULL,
                    should_save INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                );

                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    normalized_company_name TEXT NOT NULL,
                    domain TEXT,
                    lead_json TEXT NOT NULL,
                    total_score INTEGER NOT NULL,
                    fit_tier TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    industry TEXT,
                    recommended_agent_type TEXT,
                    confidence REAL,
                    last_checked_at TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_leads_domain
                    ON leads(domain)
                    WHERE domain IS NOT NULL AND domain != '';

                CREATE UNIQUE INDEX IF NOT EXISTS ux_leads_normalized_name
                    ON leads(normalized_company_name);

                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    step TEXT NOT NULL,
                    tool_called TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    output_summary TEXT NOT NULL,
                    model_reasoning_summary TEXT,
                    confidence REAL,
                    errors_json TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS domain_cache (
                    normalized_company_name TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    resolved_domain TEXT,
                    resolved_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS budget_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS critic_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_critic_lead_count INTEGER NOT NULL DEFAULT 0,
                    last_critic_run_at TEXT
                );
                """
            )

    def upsert_sources(self, sources: list[Any]) -> None:
        with self.connect() as conn:
            for source in sources:
                conn.execute(
                    """
                    INSERT INTO sources (id, type, url, priority, tags_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        type=excluded.type,
                        url=excluded.url,
                        priority=excluded.priority,
                        tags_json=excluded.tags_json
                    """,
                    (source.id, source.type, source.url, source.priority, safe_json_dumps(source.tags)),
                )

    def save_candidate(self, candidate: CandidateSignal) -> tuple[int, bool]:
        normalized_name = normalize_company_name(candidate.company_name)
        domain = normalize_domain(candidate.domain or candidate.website)
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM candidates
                WHERE normalized_company_name = ? AND source_url = ?
                """,
                (normalized_name, candidate.source_url),
            ).fetchone()
            if existing:
                return int(existing["id"]), False
            cur = conn.execute(
                """
                INSERT INTO candidates (
                    trace_id, company_name, normalized_company_name, website, domain,
                    source_url, source_type, signal_text, signal_reason,
                    detected_location, detected_industry, raw_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.trace_id or "",
                    candidate.company_name,
                    normalized_name,
                    candidate.website,
                    domain,
                    candidate.source_url,
                    candidate.source_type,
                    candidate.signal_text,
                    candidate.signal_reason,
                    candidate.detected_location,
                    candidate.detected_industry,
                    safe_json_dumps(candidate.raw_metadata),
                ),
            )
            return int(cur.lastrowid), True

    def candidate_already_processed(self, candidate_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            return bool(row and row["status"] in {"saved", "rejected"})

    def update_candidate_status(self, candidate_id: int, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE candidates SET status = ? WHERE id = ?", (status, candidate_id))

    def get_candidate(self, candidate_id: int) -> CandidateSignal | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row:
            return None
        return CandidateSignal(
            company_name=row["company_name"],
            website=row["website"],
            domain=row["domain"],
            source_url=row["source_url"],
            source_type=row["source_type"],
            signal_text=row["signal_text"],
            signal_reason=row["signal_reason"],
            detected_location=row["detected_location"],
            detected_industry=row["detected_industry"],
            raw_metadata=safe_json_loads(row["raw_metadata_json"], {}),
            trace_id=row["trace_id"],
        )

    def get_next_candidate_id_by_status(self, status: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM candidates WHERE status = ? ORDER BY created_at ASC, id ASC LIMIT 1",
                (status,),
            ).fetchone()
        return int(row["id"]) if row else None

    def get_candidate_status(self, candidate_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        return str(row["status"]) if row else None

    def get_candidate_ids_by_status(self, status: str, limit: int = 25) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM candidates
                WHERE status = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def count_candidates_by_status(self, status: str) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM candidates WHERE status = ?", (status,)).fetchone()
        return int(row["n"] if row else 0)

    def get_pending_research_profile_ids(self, limit: int = 25) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT rp.id
                FROM research_profiles rp
                JOIN candidates c ON c.id = rp.candidate_id
                WHERE c.status = 'researched'
                  AND NOT EXISTS (
                    SELECT 1 FROM scores s WHERE s.candidate_id = rp.candidate_id
                  )
                ORDER BY rp.created_at ASC, rp.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def get_latest_research_profile(self, candidate_id: int) -> ResearchProfile | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT profile_json FROM research_profiles
                WHERE candidate_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (candidate_id,),
            ).fetchone()
        if not row:
            return None
        return ResearchProfile.model_validate(safe_json_loads(row["profile_json"], {}))

    def get_research_profile_bundle(self, research_profile_id: int) -> tuple[int, CandidateSignal, ResearchProfile] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    rp.candidate_id,
                    rp.profile_json,
                    c.trace_id,
                    c.company_name,
                    c.website,
                    c.domain,
                    c.source_url,
                    c.source_type,
                    c.signal_text,
                    c.signal_reason,
                    c.detected_location,
                    c.detected_industry,
                    c.raw_metadata_json,
                    c.status
                FROM research_profiles rp
                JOIN candidates c ON c.id = rp.candidate_id
                WHERE rp.id = ?
                """,
                (research_profile_id,),
            ).fetchone()
        if not row or row["status"] != "researched":
            return None
        candidate = CandidateSignal(
            company_name=row["company_name"],
            website=row["website"],
            domain=row["domain"],
            source_url=row["source_url"],
            source_type=row["source_type"],
            signal_text=row["signal_text"],
            signal_reason=row["signal_reason"],
            detected_location=row["detected_location"],
            detected_industry=row["detected_industry"],
            raw_metadata=safe_json_loads(row["raw_metadata_json"], {}),
            trace_id=row["trace_id"],
        )
        profile = ResearchProfile.model_validate(safe_json_loads(row["profile_json"], {}))
        return int(row["candidate_id"]), candidate, profile

    def recent_rejection_reasons(self, limit: int = 5) -> list[str]:
        reasons: list[str] = []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT score_json FROM scores
                WHERE should_save = 0
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        for row in rows:
            payload = safe_json_loads(row["score_json"], {})
            reason = payload.get("rejection_reason") or payload.get("risks_or_uncertainties")
            if reason:
                reasons.append(str(reason))
        return reasons[:limit]

    def candidate_source_stats(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {}
        with self.connect() as conn:
            rows = conn.execute("SELECT status, raw_metadata_json FROM candidates").fetchall()
        for row in rows:
            metadata = safe_json_loads(row["raw_metadata_json"], {})
            source_id = str(metadata.get("source_id") or "")
            if not source_id:
                continue
            source_stats = stats.setdefault(source_id, {"candidates_yielded": 0, "leads_yielded": 0})
            source_stats["candidates_yielded"] += 1
            if row["status"] == "saved":
                source_stats["leads_yielded"] += 1
        return stats

    def source_trace_stats(self, source_id: str, since: str | None = None) -> dict[str, Any]:
        params: list[Any] = [source_id]
        since_clause = ""
        if since:
            since_clause = "AND timestamp >= ?"
            params.append(since)
        with self.connect() as conn:
            selected = conn.execute(
                f"""
                SELECT timestamp FROM traces
                WHERE step = 'source_selected'
                  AND input_summary = ?
                  {since_clause}
                ORDER BY timestamp DESC
                """,
                params,
            ).fetchall()
            discovered = conn.execute(
                f"""
                SELECT output_summary FROM traces
                WHERE step = 'candidates_discovered'
                  AND input_summary = ?
                  {since_clause}
                ORDER BY timestamp DESC, id DESC
                LIMIT 2
                """,
                params,
            ).fetchall()
            errors = conn.execute(
                """
                SELECT message FROM errors
                WHERE context_json LIKE ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (f'%"{source_id}"%',),
            ).fetchone()
        zero_runs = 0
        for row in discovered:
            if str(row["output_summary"]).lower().startswith("discovered 0"):
                zero_runs += 1
            else:
                break
        return {
            "last_used_at": selected[0]["timestamp"] if selected else None,
            "times_touched": len(selected),
            "zero_candidate_runs": zero_runs,
            "last_error": errors["message"] if errors else None,
        }

    def get_cached_domain(self, company_name: str) -> tuple[bool, str | None]:
        normalized_name = normalize_company_name(company_name)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT resolved_domain FROM domain_cache WHERE normalized_company_name = ?",
                (normalized_name,),
            ).fetchone()
        if row is None:
            return False, None
        return True, row["resolved_domain"]

    def save_cached_domain(self, company_name: str, domain: str | None) -> None:
        normalized_name = normalize_company_name(company_name)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO domain_cache (normalized_company_name, company_name, resolved_domain)
                VALUES (?, ?, ?)
                ON CONFLICT(normalized_company_name) DO UPDATE SET
                    company_name=excluded.company_name,
                    resolved_domain=excluded.resolved_domain,
                    resolved_at=CURRENT_TIMESTAMP
                """,
                (normalized_name, company_name, domain),
            )

    def save_research_profile(self, candidate_id: int, profile: ResearchProfile) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO research_profiles (candidate_id, trace_id, profile_json, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, profile.trace_id or "", profile.model_dump_json(), profile.confidence),
            )
            conn.execute("DELETE FROM evidence_items WHERE candidate_id = ?", (candidate_id,))
            for evidence in profile.evidence_items:
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        candidate_id, trace_id, url, title, quote_or_summary, signal_type, why_it_matters
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        profile.trace_id or "",
                        evidence.url,
                        evidence.title,
                        evidence.quote_or_summary,
                        evidence.signal_type,
                        evidence.why_it_matters,
                    ),
                )
            return int(cur.lastrowid)

    def save_score(self, candidate_id: int, trace_id: str, score: ScoreResult) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO scores (candidate_id, trace_id, score_json, total_score, fit_tier, should_save)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (candidate_id, trace_id, score.model_dump_json(), score.total_score, score.fit_tier, int(score.should_save)),
            )

    def save_lead(self, lead: QualifiedLead) -> bool:
        normalized_name = normalize_company_name(lead.company_name)
        domain = normalize_domain(lead.domain or lead.website)
        with self.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO leads (
                        trace_id, normalized_company_name, domain, lead_json, total_score,
                        fit_tier, company_name, industry, recommended_agent_type,
                        confidence, last_checked_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lead.agent_trace_id,
                        normalized_name,
                        domain,
                        lead.model_dump_json(),
                        lead.total_score,
                        lead.fit_tier,
                        lead.company_name,
                        lead.industry,
                        lead.recommended_agent_type,
                        lead.confidence,
                        lead.last_checked_at,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def save_trace(self, event: TraceEvent) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO traces (
                    trace_id, timestamp, step, tool_called, input_summary,
                    output_summary, model_reasoning_summary, confidence, errors_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.trace_id,
                    event.timestamp,
                    event.step,
                    event.tool_called,
                    event.input_summary,
                    event.output_summary,
                    event.model_reasoning_summary,
                    event.confidence,
                    safe_json_dumps(event.errors),
                ),
            )

    def save_error(self, event: ErrorEvent) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO errors (timestamp, source, message, context_json) VALUES (?, ?, ?, ?)",
                (event.timestamp, event.source, event.message, safe_json_dumps(event.context)),
            )

    def get_leads(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT lead_json FROM leads
                ORDER BY total_score DESC,
                    CASE fit_tier WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 ELSE 4 END,
                    company_name ASC
                """
            ).fetchall()
        return [safe_json_loads(row["lead_json"], {}) for row in rows]

    def get_recent_leads(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, trace_id, lead_json FROM leads
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        leads: list[dict[str, Any]] = []
        for row in reversed(rows):
            lead = safe_json_loads(row["lead_json"], {})
            lead["lead_id"] = str(row["id"])
            lead.setdefault("agent_trace_id", row["trace_id"])
            leads.append(lead)
        return leads

    def get_summary(self) -> RunSummary:
        with self.connect() as conn:
            total_candidates = conn.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
            researched = conn.execute("SELECT COUNT(DISTINCT candidate_id) AS n FROM research_profiles").fetchone()["n"]
            saved = conn.execute("SELECT COUNT(*) AS n FROM leads").fetchone()["n"]
            a_tier = conn.execute("SELECT COUNT(*) AS n FROM leads WHERE fit_tier = 'A'").fetchone()["n"]
            b_tier = conn.execute("SELECT COUNT(*) AS n FROM leads WHERE fit_tier = 'B'").fetchone()["n"]
            c_tier = conn.execute("SELECT COUNT(*) AS n FROM leads WHERE fit_tier = 'C'").fetchone()["n"]
            rejected = conn.execute("SELECT COUNT(*) AS n FROM candidates WHERE status IN ('rejected', 'disqualified')").fetchone()["n"]
            avg = conn.execute("SELECT AVG(total_score) AS n FROM leads").fetchone()["n"] or 0.0
            times = conn.execute("SELECT MIN(timestamp) AS first_at, MAX(timestamp) AS last_at FROM traces").fetchone()
        return RunSummary(
            total_candidates=total_candidates,
            researched_candidates=researched,
            saved_leads=saved,
            a_tier_leads=a_tier,
            b_tier_leads=b_tier,
            c_tier_leads=c_tier,
            rejected_candidates=rejected,
            average_score=round(float(avg), 1),
            first_trace_at=times["first_at"],
            last_trace_at=times["last_at"],
            run_duration=_duration_text(times["first_at"], times["last_at"]),
        )

    def status_lines(self) -> list[str]:
        summary = self.get_summary()
        return [
            f"Candidates discovered: {summary.total_candidates}",
            f"Researched candidates: {summary.researched_candidates}",
            f"Saved leads: {summary.saved_leads}",
            f"A-tier leads: {summary.a_tier_leads}",
            f"B-tier leads: {summary.b_tier_leads}",
            f"Rejected weak leads: {summary.rejected_candidates}",
            f"Average saved score: {summary.average_score}",
            f"Run duration: {summary.run_duration}",
        ]

    def save_budget(self, budget: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO budget_state (id, state_json)
                VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (safe_json_dumps(budget.to_dict()),),
            )

    def load_budget_state(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT state_json FROM budget_state WHERE id = 1").fetchone()
        if not row:
            return None
        return safe_json_loads(row["state_json"], {})

    def get_critic_state(self) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT last_critic_lead_count, last_critic_run_at FROM critic_state WHERE id = 1"
            ).fetchone()
        if not row:
            return {"last_critic_lead_count": 0, "last_critic_run_at": None}
        return {
            "last_critic_lead_count": int(row["last_critic_lead_count"] or 0),
            "last_critic_run_at": row["last_critic_run_at"],
        }

    def save_critic_state(self, last_critic_lead_count: int, last_critic_run_at: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO critic_state (id, last_critic_lead_count, last_critic_run_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_critic_lead_count=excluded.last_critic_lead_count,
                    last_critic_run_at=excluded.last_critic_run_at
                """,
                (last_critic_lead_count, last_critic_run_at),
            )


def _duration_text(first_at: str | None, last_at: str | None) -> str:
    if not first_at or not last_at:
        return "0m"
    from datetime import datetime

    try:
        first = datetime.fromisoformat(first_at)
        last = datetime.fromisoformat(last_at)
    except ValueError:
        return "0m"
    seconds = max(0, int((last - first).total_seconds()))
    hours, rem = divmod(seconds, 3600)
    minutes, _ = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
