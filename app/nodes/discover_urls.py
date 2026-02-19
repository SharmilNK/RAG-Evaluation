from __future__ import annotations

from typing import Dict, List

from app.url_discovery import discover_urls


def discover_urls_node(state: Dict) -> Dict:
    company_name = state["company_name"]
    company_domain = state["company_domain"]

    urls = discover_urls(company_name=company_name, domain=company_domain, max_urls=30)

    return {
        "target_urls": urls,
        "url_count": len(urls),
    }
