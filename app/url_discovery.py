from __future__ import annotations

import os
from typing import Iterable, List
from urllib.parse import urlparse, urlunparse

import requests


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="")
    normalized = urlunparse(cleaned)
    return normalized.rstrip("/")


def _base_urls(domain: str) -> List[str]:
    base = f"https://{domain}".rstrip("/")
    paths = [
        "",
        "/about",
        "/company",
        "/product",
        "/products",
        "/solutions",
        "/pricing",
        "/careers",
        "/jobs",
        "/press",
        "/news",
        "/blog",
        "/investors",
        "/investor-relations",
        "/annual-report",
        "/sustainability",
        "/responsible-ai",
        "/ai",
        "/security",
        "/privacy",
    ]
    return [f"{base}{path}" for path in paths]


def _search_queries(company_name: str, domain: str) -> List[str]:
    return [
        f"{company_name} AI strategy site:{domain}",
        f"{company_name} responsible AI site:{domain}",
        f"{company_name} machine learning jobs site:{domain}",
    ]


def _search_with_serpapi(queries: Iterable[str], api_key: str) -> List[str]:
    urls: List[str] = []
    for query in queries:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "engine": "google", "api_key": api_key},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        for result in data.get("organic_results", []):
            link = result.get("link")
            if link:
                urls.append(link)
    return urls


def discover_urls(company_name: str, domain: str, max_urls: int = 30) -> List[str]:
    urls = []
    urls.extend(_base_urls(domain))

    api_key = os.getenv("SERPAPI_API_KEY")
    if api_key:
        try:
            urls.extend(_search_with_serpapi(_search_queries(company_name, domain), api_key))
        except requests.RequestException:
            pass

    normalized: List[str] = []
    seen = set()
    for url in urls:
        cleaned = _normalize_url(url)
        if not cleaned:
            continue
        parsed = urlparse(cleaned)
        if parsed.netloc and not parsed.netloc.endswith(domain):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
        if len(normalized) >= max_urls:
            break

    return normalized
