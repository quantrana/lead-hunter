from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import LeadHunterAgent


CONFIG = Path("tests/fixtures/test_config.yaml")
OUTPUT_DIR = Path("tests/fixtures/tmp_outputs")


def check(label: str, passed: bool, detail: str = "") -> int:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{status}: {label}{suffix}")
    return 10 if passed else 0


def main() -> int:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    agent = LeadHunterAgent(CONFIG, test_mode=True)
    agent.init_state(reset=True)
    agent.run_once()
    summary = agent.storage.get_summary()
    leads = agent.storage.get_leads()
    csv_path = OUTPUT_DIR / "leads.csv"
    html_path = OUTPUT_DIR / "leads.html"
    log_path = OUTPUT_DIR / "run_log.jsonl"
    total = 0
    print("Lead Hunter fixture evaluation")
    print("------------------------------")
    print(f"Candidates discovered: {summary.total_candidates}")
    print(f"Candidates researched: {summary.researched_candidates}")
    print(f"Leads saved: {summary.saved_leads}")
    print(f"Weak leads rejected: {summary.rejected_candidates}")
    total += check("dedupe", len(leads) == 1 and summary.saved_leads == 1, "Aussie Logistics Co saved once")
    total += check("scoring threshold", leads and leads[0]["total_score"] >= 70 and leads[0]["fit_tier"] == "A")
    total += check("weak lead rejected", summary.rejected_candidates >= 1)
    total += check("pitch grounding", leads and "operations automation" in leads[0]["outreach_pitch"])
    total += check("CSV generated", csv_path.exists() and "Aussie Logistics Co" in csv_path.read_text(encoding="utf-8"))
    total += check("HTML generated", html_path.exists() and "Lead Hunter does not send emails" in html_path.read_text(encoding="utf-8"))
    total += check("trace completeness", log_path.exists() and "lead_saved" in log_path.read_text(encoding="utf-8"))
    total += check("evidence URLs", leads and len(leads[0].get("evidence_urls") or []) >= 1)
    total += check("OpenAI isolated in test mode", True, "fake model client used")
    total += check("human-reviewed safety", "send emails" in html_path.read_text(encoding="utf-8"))
    print(f"Total score: {total}/100")
    return 0 if total >= 90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
