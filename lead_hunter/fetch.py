from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

from .models import FetchedPage, RunConfig
from .utils import compact_text


class Fetcher:
    def __init__(self, run_config: RunConfig, test_mode: bool = False, budget=None) -> None:
        self.run_config = run_config
        self.test_mode = test_mode
        self.budget = budget
        self._last_fetch_at = 0.0

    def fetch(self, url: str) -> FetchedPage:
        if self.test_mode:
            return self._fetch_fixture_or_file(url)
        self._respect_delay()
        if self.budget:
            self.budget.record_web_fetch()
        headers = {"User-Agent": self.run_config.user_agent}
        try:
            with httpx.Client(
                timeout=self.run_config.request_timeout_seconds,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = client.get(url)
                content_type = response.headers.get("content-type", "")
                if response.status_code >= 400:
                    return FetchedPage(
                        url=url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        error=f"HTTP {response.status_code}",
                    )
                if "text" not in content_type and "html" not in content_type and "xml" not in content_type:
                    return FetchedPage(
                        url=url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        error=f"Unsupported content type: {content_type}",
                    )
                text = response.text[:600_000]
                return self.parse_html(url, str(response.url), text, response.status_code)
        except Exception as exc:  # pragma: no cover - network dependent
            return FetchedPage(url=url, final_url=url, error=str(exc))

    def fetch_raw_text(self, url: str) -> tuple[str, str | None]:
        if self.test_mode:
            if url.startswith("fixture://"):
                path = Path(url.replace("fixture://", "", 1))
            elif url.startswith("file://"):
                path = Path(url.replace("file://", "", 1))
            else:
                return "", "Live fetch disabled in test mode"
            if not path.exists():
                return "", f"Fixture not found: {path}"
            return path.read_text(encoding="utf-8"), None
        self._respect_delay()
        if self.budget:
            self.budget.record_web_fetch()
        headers = {"User-Agent": self.run_config.user_agent}
        try:
            with httpx.Client(
                timeout=self.run_config.request_timeout_seconds,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = client.get(url)
                if response.status_code >= 400:
                    return "", f"HTTP {response.status_code}"
                return response.text[:600_000], None
        except Exception as exc:  # pragma: no cover - network dependent
            return "", str(exc)

    def parse_rss(self, content: str, source_url: str) -> list[dict]:
        feed = feedparser.parse(content)
        entries: list[dict] = []
        for entry in feed.entries[:50]:
            entries.append(
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", source_url),
                    "summary": compact_text(entry.get("summary", "") or entry.get("description", ""), 1200),
                    "published": entry.get("published", ""),
                }
            )
        return entries

    def parse_html(self, url: str, final_url: str, html: str, status_code: int | None = None) -> FetchedPage:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = compact_text(soup.title.get_text(" ", strip=True) if soup.title else "", 240)
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "")
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            absolute = urljoin(final_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in {"http", "https", "file"}:
                links.append(absolute)
        visible_text = soup.get_text(" ", strip=True)
        visible_text = re.sub(r"\s+", " ", visible_text)
        return FetchedPage(
            url=url,
            final_url=final_url,
            title=title,
            text=compact_text(visible_text, 12000),
            html=html[:1_000_000],
            links=links[:100],
            status_code=status_code,
        )

    def _fetch_fixture_or_file(self, url: str) -> FetchedPage:
        if url.startswith("fixture://"):
            path = Path(url.replace("fixture://", "", 1))
        elif url.startswith("file://"):
            path = Path(url.replace("file://", "", 1))
        else:
            return FetchedPage(url=url, final_url=url, error="Live fetch disabled in test mode")
        if not path.exists():
            return FetchedPage(url=url, final_url=url, error=f"Fixture not found: {path}")
        content = path.read_text(encoding="utf-8")
        return self.parse_html(url, url, content, 200)

    def _respect_delay(self) -> None:
        delay = max(0.0, float(self.run_config.crawl_delay_seconds))
        elapsed = time.time() - self._last_fetch_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_fetch_at = time.time()
