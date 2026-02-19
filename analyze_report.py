"""
Enhanced Report Analyzer - Visualize source quality, confidence metrics, and improvement opportunities.

Usage:
    python analyze_report.py app/output/report_73cd4a6f.yaml
    python analyze_report.py app/output/report_73cd4a6f.yaml --kpi strat_ai_vision
"""

import os
import sys
import yaml
from collections import defaultdict, Counter
from typing import Dict, List, Any
from urllib.parse import urlparse


def load_report(report_path: str) -> Dict:
    """Load YAML report file."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def analyze_source_distribution(report: Dict) -> Dict[str, Any]:
    """Analyze where sources are coming from across the entire report."""

    # Collect all sources from citations
    all_sources = []
    all_domains = []
    all_tiers = []

    for kpi in report.get('kpi_results', []):
        citations = kpi.get('citations', [])
        details = kpi.get('details', {})
        tier_dist = details.get('tier_distribution', {})

        # Track tier distribution
        for tier_level in ['tier1', 'tier2', 'tier3']:
            count = tier_dist.get(tier_level, 0)
            all_tiers.extend([tier_level] * count)

        # Track sources and domains
        for citation in citations:
            url = citation.get('url', '')
            source_id = citation.get('source_id', '')
            all_sources.append(source_id)
            if url:
                domain = urlparse(url).netloc
                all_domains.append(domain)

    # Count occurrences
    source_counts = Counter(all_sources)
    domain_counts = Counter(all_domains)
    tier_counts = Counter(all_tiers)

    return {
        'total_citations': len(all_sources),
        'unique_sources': len(source_counts),
        'unique_domains': len(domain_counts),
        'most_cited_sources': source_counts.most_common(10),
        'most_cited_domains': domain_counts.most_common(10),
        'tier_distribution': dict(tier_counts),
        'tier_percentages': {
            tier: (count / len(all_tiers) * 100) if all_tiers else 0
            for tier, count in tier_counts.items()
        }
    }


def analyze_confidence_factors(kpi_result: Dict) -> Dict[str, Any]:
    """Break down confidence score calculation for a KPI."""

    details = kpi_result.get('details', {})
    confidence = kpi_result.get('confidence', 0.0)
    citations = kpi_result.get('citations', [])

    tier_dist = details.get('tier_distribution', {})
    corroboration = details.get('corroboration_score', 0.0)
    unique_sources = details.get('unique_sources', 0)
    llm_used = details.get('llm_used', False)

    # Estimate base confidence (work backwards from final confidence)
    # This is approximate since we don't know the exact base
    estimated_base = 0.5 if llm_used else 0.4

    # Calculate tier quality boost
    avg_tier = tier_dist.get('avg', 3.0)
    tier_boost_max = 0.15
    tier_boost = max(0.0, tier_boost_max - (avg_tier - 1.0) * (tier_boost_max / 2.0))

    # Calculate corroboration boost
    corroboration_boost_max = 0.15
    corroboration_boost = corroboration * corroboration_boost_max

    # Diversity boost
    diversity_boost = 0.05 if unique_sources >= 3 else 0.0

    # Penalties
    citation_penalty = -0.3 if llm_used and not citations else 0.0
    evidence_count = tier_dist.get('tier1', 0) + tier_dist.get('tier2', 0) + tier_dist.get('tier3', 0)
    low_evidence_penalty = -0.2 if evidence_count < 3 else 0.0

    # Reconstruct confidence (may not match exactly due to estimation)
    reconstructed = estimated_base + tier_boost + corroboration_boost + diversity_boost + citation_penalty + low_evidence_penalty
    reconstructed = max(0.0, min(1.0, reconstructed))

    return {
        'final_confidence': confidence,
        'estimated_base': estimated_base,
        'components': {
            'tier_quality_boost': round(tier_boost, 3),
            'corroboration_boost': round(corroboration_boost, 3),
            'diversity_boost': round(diversity_boost, 3),
            'citation_penalty': round(citation_penalty, 3),
            'low_evidence_penalty': round(low_evidence_penalty, 3),
        },
        'reconstructed_confidence': round(reconstructed, 2),
        'explanation': []
    }


def format_confidence_breakdown(factors: Dict) -> str:
    """Format confidence breakdown as readable text."""
    lines = []
    lines.append("\n  Confidence Breakdown:")
    lines.append(f"    Base Confidence:       {factors['estimated_base']:.2f}")

    components = factors['components']
    for name, value in components.items():
        if value != 0:
            symbol = "+" if value > 0 else ""
            lines.append(f"    {name.replace('_', ' ').title():<23} {symbol}{value:.3f}")

    lines.append(f"    {'='*23} {'='*6}")
    lines.append(f"    Final Confidence:      {factors['final_confidence']:.2f}")
    lines.append(f"    (Reconstructed:        {factors['reconstructed_confidence']:.2f})")

    return "\n".join(lines)


def analyze_kpi_evidence(kpi_result: Dict) -> Dict[str, Any]:
    """Analyze evidence sources for a specific KPI."""

    citations = kpi_result.get('citations', [])
    details = kpi_result.get('details', {})
    tier_dist = details.get('tier_distribution', {})

    evidence_sources = []
    for citation in citations:
        source_id = citation.get('source_id', '')
        url = citation.get('url', '')
        quote = citation.get('quote', '')

        # Try to infer tier from source_id pattern
        # In reality, we'd need to track this from the original indexing
        inferred_tier = 2  # Default
        if 'investor' in url or 'annual-report' in url or '.pdf' in url:
            inferred_tier = 1
        elif 'news' in url or 'blog' in url:
            inferred_tier = 2

        evidence_sources.append({
            'source_id': source_id,
            'url': url,
            'quote': quote[:150] + '...' if len(quote) > 150 else quote,
            'inferred_tier': inferred_tier,
        })

    return {
        'num_citations': len(citations),
        'unique_sources': details.get('unique_sources', 0),
        'tier_distribution': tier_dist,
        'corroboration_score': details.get('corroboration_score', 0.0),
        'k_used': details.get('k_used', 0),
        'evidence_sources': evidence_sources,
    }


def identify_improvements(report: Dict) -> List[Dict[str, Any]]:
    """Identify areas for improvement based on report analysis."""

    improvements = []

    for kpi in report.get('kpi_results', []):
        kpi_id = kpi.get('kpi_id', '')
        confidence = kpi.get('confidence', 0.0)
        details = kpi.get('details', {})
        tier_dist = details.get('tier_distribution', {})
        corroboration = details.get('corroboration_score', 0.0)
        unique_sources = details.get('unique_sources', 0)
        citations = kpi.get('citations', [])

        # Low confidence
        if confidence < 0.5:
            improvements.append({
                'kpi_id': kpi_id,
                'issue': 'Low Confidence',
                'value': confidence,
                'recommendation': f"Confidence is {confidence:.2f}. Consider fetching more sources or improving query.",
            })

        # Poor tier quality
        avg_tier = tier_dist.get('avg', 3.0)
        if avg_tier > 2.5:
            improvements.append({
                'kpi_id': kpi_id,
                'issue': 'Poor Source Quality',
                'value': avg_tier,
                'recommendation': f"Average tier is {avg_tier:.2f}. Need more tier 1 sources (investor docs, press releases).",
            })

        # Low corroboration
        if corroboration < 0.3 and unique_sources > 1:
            improvements.append({
                'kpi_id': kpi_id,
                'issue': 'Low Corroboration',
                'value': corroboration,
                'recommendation': f"Corroboration score is {corroboration:.2f}. Sources don't agree strongly.",
            })

        # Single source
        if unique_sources <= 1:
            improvements.append({
                'kpi_id': kpi_id,
                'issue': 'Single Source',
                'value': unique_sources,
                'recommendation': "Only 1 unique source. Need diverse evidence from multiple sources.",
            })

        # Missing citations
        if not citations and details.get('llm_used'):
            improvements.append({
                'kpi_id': kpi_id,
                'issue': 'Missing Citations',
                'value': 0,
                'recommendation': "LLM used but no citations provided. Major confidence penalty applied.",
            })

    return improvements


def print_report_summary(report: Dict):
    """Print high-level report summary."""
    print("=" * 80)
    print("REPORT SUMMARY")
    print("=" * 80)
    print(f"Company:       {report.get('company_name', 'N/A')}")
    print(f"Domain:        {report.get('company_domain', 'N/A')}")
    print(f"Run ID:        {report.get('run_id', 'N/A')}")
    print(f"Timestamp:     {report.get('timestamp', 'N/A')}")
    print(f"URLs Fetched:  {report.get('url_count', 0)}")
    print(f"Overall Score: {report.get('overall_score', 0.0):.2f}")
    print()

    # Pillar scores
    print("Pillar Scores:")
    for pillar in report.get('pillar_scores', []):
        print(f"  {pillar['pillar']:<30} Score: {pillar['score']:.2f}  Confidence: {pillar['confidence']:.2f}")
    print()


def print_source_analysis(source_dist: Dict):
    """Print source distribution analysis."""
    print("=" * 80)
    print("SOURCE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Total Citations:   {source_dist['total_citations']}")
    print(f"Unique Sources:    {source_dist['unique_sources']}")
    print(f"Unique Domains:    {source_dist['unique_domains']}")
    print()

    print("Tier Distribution:")
    tier_dist = source_dist['tier_distribution']
    tier_pct = source_dist['tier_percentages']
    for tier in ['tier1', 'tier2', 'tier3']:
        count = tier_dist.get(tier, 0)
        pct = tier_pct.get(tier, 0)
        print(f"  {tier.upper():<10} {count:>3} citations ({pct:>5.1f}%)")
    print()

    print("Most Cited Domains:")
    for domain, count in source_dist['most_cited_domains'][:5]:
        print(f"  {domain:<40} {count:>3} citations")
    print()

    print("Most Cited Sources:")
    for source_id, count in source_dist['most_cited_sources'][:5]:
        print(f"  {source_id:<50} {count:>3} citations")
    print()


def print_kpi_detail(kpi_result: Dict):
    """Print detailed analysis for a specific KPI."""
    kpi_id = kpi_result.get('kpi_id', '')
    kpi_name = kpi_result.get('kpi_id', '').replace('_', ' ').title()
    score = kpi_result.get('score', 0)
    confidence = kpi_result.get('confidence', 0.0)

    print("=" * 80)
    print(f"KPI DETAIL: {kpi_id}")
    print("=" * 80)
    print(f"Name:       {kpi_name}")
    print(f"Pillar:     {kpi_result.get('pillar', 'N/A')}")
    print(f"Type:       {kpi_result.get('type', 'N/A')}")
    print(f"Score:      {score}")
    print(f"Confidence: {confidence:.2f}")
    print()

    # Confidence breakdown
    factors = analyze_confidence_factors(kpi_result)
    print(format_confidence_breakdown(factors))
    print()

    # Evidence analysis
    evidence = analyze_kpi_evidence(kpi_result)
    print(f"\n  Evidence Summary:")
    print(f"    Citations:         {evidence['num_citations']}")
    print(f"    Unique Sources:    {evidence['unique_sources']}")
    print(f"    Corroboration:     {evidence['corroboration_score']:.2f}")
    print(f"    K Used:            {evidence['k_used']}")

    tier_dist = evidence['tier_distribution']
    print(f"    Tier Distribution: T1={tier_dist.get('tier1', 0)}, T2={tier_dist.get('tier2', 0)}, T3={tier_dist.get('tier3', 0)} (avg={tier_dist.get('avg', 0):.2f})")
    print()

    # Evidence sources
    print("\n  Evidence Sources:")
    for i, source in enumerate(evidence['evidence_sources'], 1):
        print(f"\n    [{i}] Source: {source['source_id']}")
        print(f"        URL:   {source['url']}")
        print(f"        Tier:  {source['inferred_tier']}")
        print(f"        Quote: {source['quote']}")
    print()


def print_improvements(improvements: List[Dict]):
    """Print improvement recommendations."""
    print("=" * 80)
    print("IMPROVEMENT OPPORTUNITIES")
    print("=" * 80)

    if not improvements:
        print("No major issues found! Report quality is good.")
        return

    # Group by issue type
    by_issue = defaultdict(list)
    for imp in improvements:
        by_issue[imp['issue']].append(imp)

    for issue_type, items in by_issue.items():
        print(f"\n{issue_type} ({len(items)} KPIs affected):")
        for item in items[:5]:  # Show first 5
            print(f"  - {item['kpi_id']:<30} {item['recommendation']}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_report.py <report_path> [--kpi KPI_ID]")
        print("Example: python analyze_report.py app/output/report_73cd4a6f.yaml")
        print("Example: python analyze_report.py app/output/report_73cd4a6f.yaml --kpi strat_ai_vision")
        sys.exit(1)

    report_path = sys.argv[1]

    # Check for specific KPI flag
    specific_kpi = None
    if '--kpi' in sys.argv:
        kpi_index = sys.argv.index('--kpi')
        if kpi_index + 1 < len(sys.argv):
            specific_kpi = sys.argv[kpi_index + 1]

    # Load report
    try:
        report = load_report(report_path)
    except Exception as e:
        print(f"Error loading report: {e}")
        sys.exit(1)

    # If specific KPI requested, show only that
    if specific_kpi:
        kpi_result = None
        for kpi in report.get('kpi_results', []):
            if kpi.get('kpi_id') == specific_kpi:
                kpi_result = kpi
                break

        if not kpi_result:
            print(f"KPI '{specific_kpi}' not found in report.")
            print(f"Available KPIs: {[k['kpi_id'] for k in report.get('kpi_results', [])]}")
            sys.exit(1)

        print_kpi_detail(kpi_result)
        return

    # Full report analysis
    print_report_summary(report)

    source_dist = analyze_source_distribution(report)
    print_source_analysis(source_dist)

    improvements = identify_improvements(report)
    print_improvements(improvements)

    # Show available KPIs for detailed analysis
    print("=" * 80)
    print("AVAILABLE KPIs FOR DETAILED ANALYSIS")
    print("=" * 80)
    print("\nTo analyze a specific KPI, run:")
    print(f"  python analyze_report.py {report_path} --kpi <KPI_ID>\n")
    print("Available KPIs:")
    for kpi in report.get('kpi_results', []):
        kpi_id = kpi['kpi_id']
        score = kpi['score']
        confidence = kpi['confidence']
        citations = len(kpi.get('citations', []))
        print(f"  {kpi_id:<35} Score: {score}  Confidence: {confidence:.2f}  Citations: {citations}")
    print()


if __name__ == "__main__":
    main()
