from __future__ import annotations

import json
import re
import uuid
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import now_utc


AU_HINTS = {
    "australia",
    "australian",
    "melbourne",
    "sydney",
    "brisbane",
    "perth",
    "adelaide",
    "canberra",
    "victoria",
    "nsw",
    "queensland",
}

SIGNAL_TERMS = {
    "automation",
    "operations",
    "support",
    "customer success",
    "invoice",
    "reporting",
    "workflow",
    "manual",
    "spreadsheet",
    "crm",
    "erp",
    "onboarding",
    "compliance",
    "logistics",
    "finance ops",
    "revops",
    "data",
    "ai",
    "hiring",
    "growth",
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def trace_id() -> str:
    return f"lh_{uuid.uuid4().hex[:12]}"


def normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = (parsed.netloc or parsed.path).lower().strip()
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def domain_from_url(url: str | None) -> str | None:
    return normalize_domain(url)


def normalize_company_name(name: str | None) -> str:
    if not name:
        return ""
    text = name.lower()
    text = re.sub(r"\b(pty|ltd|limited|inc|co|company|group|holdings|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str, limit: int = 4000) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def find_signal_terms(text: str) -> list[str]:
    lower = (text or "").lower()
    return sorted(term for term in SIGNAL_TERMS if term in lower)


def has_australia_hint(text: str) -> bool:
    lower = (text or "").lower()
    return any(term in lower for term in AU_HINTS)


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def safe_json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(safe_json_dumps(payload) + "\n")


def html_escape(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def current_env() -> str:
    import os

    return os.getenv("LEAD_HUNTER_ENV", "production")


def timestamped_context(**kwargs: Any) -> dict[str, Any]:
    payload = {"timestamp": now_utc()}
    payload.update(kwargs)
    return payload
