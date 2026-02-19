from __future__ import annotations

import os
from typing import Dict, List

from app.models import SourceDoc
from app.observability import get_tracer


def _stub_sources(company_id: str) -> List[Dict[str, str]]:
    return [
        {
            "url": f"https://{company_id}.com/",
            "text": "We build analytics software for B2B teams. Strong product quality and roadmap focus.",
            "page_type": "home",
        },
        {
            "url": f"https://{company_id}.com/product",
            "text": "Our product improves revenue pipeline visibility with automation and reporting.",
            "page_type": "product",
        },
        {
            "url": f"https://{company_id}.com/pricing",
            "text": "Pricing tiers reflect scalable growth and retention. Costs are optimized for efficiency.",
            "page_type": "pricing",
        },
        {
            "url": f"https://{company_id}.com/about",
            "text": "About us: trusted brand with strong customer support and credible leadership.",
            "page_type": "about",
        },
        {
            "url": f"https://{company_id}.com/careers",
            "text": "We are hiring across sales and engineering to scale growth and improve processes.",
            "page_type": "careers",
        },
    ]


def _assign_tier(text: str) -> int:
    keywords_tier1 = ["product", "pricing", "revenue", "pipeline"]
    if any(keyword in text.lower() for keyword in keywords_tier1):
        return 1
    if len(text) > 80:
        return 2
    return 3


def sources_node(state: Dict) -> Dict:
    tracer = get_tracer()
    with tracer.span("sources"):
        company_id = state["company_id"]
        raw_sources = _stub_sources(company_id)

        seen = set()
        sources: List[SourceDoc] = []
        for idx, raw in enumerate(raw_sources):
            url = raw["url"].rstrip("/")
            if url in seen:
                continue
            seen.add(url)

            text = raw["text"].strip()
            if company_id.lower() not in text.lower() and "product" not in text.lower():
                continue

            tier = _assign_tier(text)
            source = SourceDoc(
                source_id=f"source_{idx}",
                url=url,
                text=text,
                tier=tier,
                page_type=raw["page_type"],
            )
            sources.append(source)

        return {
            "sources": [source.model_dump() for source in sources],
            "collection_id": f"collection_{state['run_id']}",
        }
