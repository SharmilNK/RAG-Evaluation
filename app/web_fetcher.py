from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from readability import Document
from lxml import html

from app.debug_log import add_debug


class FetchError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _debug_enabled() -> bool:
    return os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}


def _extract_text(raw_html: str) -> str:
    doc = Document(raw_html)
    summary_html = doc.summary(html_partial=True)
    title = doc.short_title()

    try:
        parsed = html.fromstring(summary_html)
        text = parsed.text_content()
    except (ValueError, TypeError):
        text = ""

    return title.strip(), " ".join(text.split())


def _fetch_with_firecrawl(url: str, api_key: str) -> Optional[Dict[str, str]]:
    max_retries = int(os.getenv("VITELIS_FIRECRAWL_MAX_RETRIES", "2"))
    base_backoff = float(os.getenv("VITELIS_FIRECRAWL_BACKOFF", "2.0"))
    for attempt in range(max_retries + 1):
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": url, "formats": ["markdown", "html"]},
            timeout=30,
        )
        if response.status_code == 429 and attempt < max_retries:
            retry_after = response.json().get("error", "")
            wait_time = base_backoff * (2**attempt)
            if _debug_enabled():
                add_debug(f"[fetch] firecrawl 429; retrying in {wait_time:.1f}s: {url}")
            time.sleep(wait_time)
            continue
        if response.status_code >= 400:
            raise FetchError(f"Firecrawl error {response.status_code}: {response.text}")

        data = response.json().get("data", {})
        text = data.get("markdown") or data.get("text") or ""
        title = data.get("title") or ""
        if not text:
            return None
        return {
            "title": title.strip(),
            "text": " ".join(text.split()),
        }
    return None


def _fetch_with_readability(url: str) -> Optional[Dict[str, str]]:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; VitelisBot/0.2)"},
        timeout=20,
    )
    if response.status_code >= 400:
        raise FetchError(f"HTTP {response.status_code} for {url}")
    title, text = _extract_text(response.text)
    if not text:
        return None
    return {"title": title, "text": text}


def fetch_page(url: str) -> Optional[Dict[str, str]]:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if api_key:
        try:
            result = _fetch_with_firecrawl(url, api_key)
            if _debug_enabled() and result is not None:
                add_debug(f"[fetch] firecrawl ok: {url}")
            return result
        except (requests.RequestException, FetchError) as exc:
            if _debug_enabled():
                add_debug(f"[fetch] firecrawl failed, fallback: {url} ({exc})")

    try:
        result = _fetch_with_readability(url)
        if _debug_enabled() and result is not None:
            add_debug(f"[fetch] readability ok: {url}")
        return result
    except (requests.RequestException, FetchError):
        if _debug_enabled():
            add_debug(f"[fetch] readability failed: {url}")
        return None


def build_source_record(url: str, title: str, text: str, tier: int) -> Dict[str, str]:
    parsed = urlparse(url)
    domain = parsed.netloc
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return {
        "source_id": f"{domain}-{digest}",
        "url": url,
        "title": title,
        "text": text,
        "domain": domain,
        "retrieved_at": _now_iso(),
        "tier": tier,
    }
