from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .models import EvidenceItem
from .utils import compact_text, normalize_domain


DOMAIN_SUFFIX_PATTERNS = [
    r"\s+Pty\.?\s+Ltd\.?$",
    r"\s+Ltd\.?$",
    r"\s+Limited$",
    r"\s+Group$",
    r"\s+Australia$",
    r"\s+Aus$",
    r"\s+\(AU\)$",
]


def resolve_domain(company_name: str, budget=None) -> str | None:
    for domain in _domain_candidates(company_name):
        if _domain_resolves(domain, budget=budget):
            return domain
    return None


def fetch_company_pages(domain: str, user_agent: str, inner_delay_seconds: float = 1.5, budget=None) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    attempts = [
        ("homepage", [f"https://{domain}/"]),
        ("about", [f"https://{domain}/about", f"https://{domain}/about-us", f"https://{domain}/our-company"]),
        ("careers", [f"https://{domain}/careers", f"https://{domain}/jobs", f"https://{domain}/work-with-us"]),
    ]
    headers = {"User-Agent": user_agent}
    for signal_type, urls in attempts:
        page = None
        for url in urls:
            page = _fetch_company_page(url, signal_type, headers, budget=budget)
            if page:
                pages.append(page)
                break
        if signal_type != attempts[-1][0]:
            time.sleep(max(0.0, inner_delay_seconds))
    return pages


def discovery_evidence_page(candidate) -> dict[str, str]:
    return {
        "url": candidate.source_url,
        "title": f"{candidate.company_name} discovery signal",
        "text": compact_text(candidate.signal_text, 4000),
        "signal_type": "discovery",
        "quote_or_summary": compact_text(candidate.signal_text, 500),
        "why_it_matters": "The discovery source names the company and the public signal that made it worth researching.",
    }


def page_dict_to_evidence(page: dict[str, str]) -> EvidenceItem:
    return EvidenceItem(
        url=page["url"],
        title=page.get("title") or page.get("signal_type", "Company page").title(),
        quote_or_summary=page.get("quote_or_summary") or compact_text(page.get("text", ""), 500),
        signal_type=page.get("signal_type", "company_page"),
        why_it_matters=page.get("why_it_matters") or "This company-owned page provides primary evidence for lead qualification.",
    )


def _domain_candidates(company_name: str) -> list[str]:
    bases: list[str] = []
    current = re.sub(r"\s+", " ", company_name).strip()
    if current:
        bases.append(current)
    changed = True
    while changed and current:
        changed = False
        for pattern in DOMAIN_SUFFIX_PATTERNS:
            stripped = re.sub(pattern, "", current, flags=re.IGNORECASE).strip()
            if stripped != current:
                current = stripped
                if current and current not in bases:
                    bases.append(current)
                changed = True
                break
    domains: list[str] = []
    for base in bases:
        compact_slug, hyphen_slug = _slugs(base)
        for slug in [compact_slug, hyphen_slug]:
            if not slug:
                continue
            for suffix in [".com.au", ".com"]:
                domain = f"{slug}{suffix}"
                if domain not in domains:
                    domains.append(domain)
    return domains


def _slugs(value: str) -> tuple[str, str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", value).lower()
    words = [word for word in re.split(r"[\s-]+", cleaned) if word]
    return "".join(words), "-".join(words)


def _domain_resolves(domain: str, budget=None) -> bool:
    headers = {"User-Agent": "LeadHunter/1.0 research bot for public business information"}
    for scheme in ["https", "http"]:
        url = f"{scheme}://{domain}"
        try:
            if budget:
                budget.record_web_fetch()
            response = httpx.head(url, timeout=5, follow_redirects=False, headers=headers)
        except Exception:
            continue
        if response.status_code == 200:
            return True
        if response.status_code in {301, 302}:
            location = response.headers.get("location", "")
            if _same_root_redirect(domain, location):
                return True
    return False


def _same_root_redirect(domain: str, location: str) -> bool:
    if not location:
        return False
    target = normalize_domain(location)
    source_root = _root(domain)
    target_root = _root(target)
    return bool(target_root and source_root == target_root)


def _root(domain: str | None) -> str | None:
    normalized = normalize_domain(domain)
    if not normalized:
        return None
    parts = normalized.split(".")
    if len(parts) >= 3 and parts[-2:] == ["com", "au"]:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return normalized


def _fetch_company_page(url: str, signal_type: str, headers: dict[str, str], budget=None) -> dict[str, str] | None:
    try:
        if budget:
            budget.record_web_fetch()
        response = httpx.get(url, timeout=10, follow_redirects=True, headers=headers)
    except Exception:
        return None
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        return None
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type and content_type:
        return None
    html = response.text[:200_000]
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else signal_type.title()
    text = soup.get_text(" ", strip=True)
    text = _head_tail(compact_text(text, 8000), 4000)
    if not text:
        return None
    return {
        "url": str(response.url),
        "title": compact_text(title, 240),
        "text": text,
        "signal_type": signal_type,
        "quote_or_summary": compact_text(text, 500),
        "why_it_matters": "Company-owned page used as primary evidence.",
    }


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 20) // 2)
    return f"{text[:half].rstrip()} ... {text[-half:].lstrip()}"
