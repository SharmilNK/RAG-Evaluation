"""
Compare two reports to see the impact of enhanced features.

Usage:
    python compare_reports.py app/output/report_0fe61b4d.yaml app/output/report_73cd4a6f.yaml
"""

import sys
import yaml
from typing import Dict, List, Tuple


def load_report(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compare_kpi(kpi1: Dict, kpi2: Dict) -> Dict:
    """Compare a single KPI between two reports."""
    return {
        'kpi_id': kpi1.get('kpi_id'),
        'score_before': kpi1.get('score'),
        'score_after': kpi2.get('score'),
        'score_change': kpi2.get('score') - kpi1.get('score'),
        'confidence_before': kpi1.get('confidence'),
        'confidence_after': kpi2.get('confidence'),
        'confidence_change': kpi2.get('confidence') - kpi1.get('confidence'),
        'citations_before': len(kpi1.get('citations', [])),
        'citations_after': len(kpi2.get('citations', [])),
        'has_details_before': 'tier_distribution' in kpi1.get('details', {}),
        'has_details_after': 'tier_distribution' in kpi2.get('details', {}),
        'tier_dist_after': kpi2.get('details', {}).get('tier_distribution', {}),
        'corroboration_after': kpi2.get('details', {}).get('corroboration_score', 0.0),
    }


def print_comparison(report1: Dict, report2: Dict):
    """Print detailed comparison between two reports."""

    print("=" * 100)
    print("REPORT COMPARISON")
    print("=" * 100)
    print(f"{'Metric':<30} {'BEFORE':<20} {'AFTER':<20} {'CHANGE':<20}")
    print("-" * 100)
    print(f"{'Overall Score':<30} {report1.get('overall_score', 0):<20.2f} {report2.get('overall_score', 0):<20.2f} {report2.get('overall_score', 0) - report1.get('overall_score', 0):<20.2f}")
    print()

    # Compare pillar scores
    print("Pillar Scores:")
    pillars1 = {p['pillar']: p for p in report1.get('pillar_scores', [])}
    pillars2 = {p['pillar']: p for p in report2.get('pillar_scores', [])}

    for pillar_name in pillars1.keys():
        p1 = pillars1.get(pillar_name, {})
        p2 = pillars2.get(pillar_name, {})

        score1 = p1.get('score', 0)
        score2 = p2.get('score', 0)
        conf1 = p1.get('confidence', 0)
        conf2 = p2.get('confidence', 0)

        print(f"  {pillar_name:<28}")
        print(f"    {'Score':<26} {score1:<20.2f} {score2:<20.2f} {score2-score1:<+20.2f}")
        print(f"    {'Confidence':<26} {conf1:<20.2f} {conf2:<20.2f} {conf2-conf1:<+20.2f}")
    print()

    # KPI-level comparison
    kpis1 = {k['kpi_id']: k for k in report1.get('kpi_results', [])}
    kpis2 = {k['kpi_id']: k for k in report2.get('kpi_results', [])}

    comparisons = []
    for kpi_id in kpis1.keys():
        if kpi_id in kpis2:
            comparisons.append(compare_kpi(kpis1[kpi_id], kpis2[kpi_id]))

    # Show biggest improvements
    print("=" * 100)
    print("BIGGEST CONFIDENCE IMPROVEMENTS")
    print("=" * 100)
    by_confidence = sorted(comparisons, key=lambda x: x['confidence_change'], reverse=True)
    for comp in by_confidence[:10]:
        if comp['confidence_change'] > 0:
            print(f"{comp['kpi_id']:<35} {comp['confidence_before']:.2f} -> {comp['confidence_after']:.2f} ({comp['confidence_change']:+.2f})")
            if comp['has_details_after']:
                tier_dist = comp['tier_dist_after']
                corr = comp['corroboration_after']
                print(f"  {'':35} Tier: T1={tier_dist.get('tier1',0)}, T2={tier_dist.get('tier2',0)}, T3={tier_dist.get('tier3',0)} (avg={tier_dist.get('avg',0):.2f})  Corr: {corr:.2f}")
    print()

    # Show biggest score changes
    print("=" * 100)
    print("BIGGEST SCORE CHANGES")
    print("=" * 100)
    by_score = sorted(comparisons, key=lambda x: abs(x['score_change']), reverse=True)
    for comp in by_score[:10]:
        if comp['score_change'] != 0:
            print(f"{comp['kpi_id']:<35} {comp['score_before']:.1f} -> {comp['score_after']:.1f} ({comp['score_change']:+.1f})")
    print()

    # Show enhanced features adoption
    print("=" * 100)
    print("ENHANCED FEATURES ADOPTION")
    print("=" * 100)
    enhanced_count = sum(1 for c in comparisons if c['has_details_after'])
    print(f"KPIs with enhanced details: {enhanced_count}/{len(comparisons)}")
    print()

    if enhanced_count > 0:
        # Average tier distribution
        avg_tier1 = sum(c['tier_dist_after'].get('tier1', 0) for c in comparisons if c['has_details_after']) / enhanced_count
        avg_tier2 = sum(c['tier_dist_after'].get('tier2', 0) for c in comparisons if c['has_details_after']) / enhanced_count
        avg_tier3 = sum(c['tier_dist_after'].get('tier3', 0) for c in comparisons if c['has_details_after']) / enhanced_count
        avg_tier_avg = sum(c['tier_dist_after'].get('avg', 3.0) for c in comparisons if c['has_details_after']) / enhanced_count

        print("Average Tier Distribution per KPI:")
        print(f"  Tier 1: {avg_tier1:.2f} sources")
        print(f"  Tier 2: {avg_tier2:.2f} sources")
        print(f"  Tier 3: {avg_tier3:.2f} sources")
        print(f"  Avg Tier: {avg_tier_avg:.2f}")
        print()

        # Average corroboration
        avg_corr = sum(c['corroboration_after'] for c in comparisons if c['has_details_after']) / enhanced_count
        print(f"Average Corroboration Score: {avg_corr:.2f}")
        print()

    # Confidence distribution
    print("=" * 100)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 100)
    print(f"{'Range':<20} {'BEFORE':<15} {'AFTER':<15}")
    print("-" * 50)

    ranges = [
        ('0.0 - 0.3', 0.0, 0.3),
        ('0.3 - 0.5', 0.3, 0.5),
        ('0.5 - 0.7', 0.5, 0.7),
        ('0.7 - 0.9', 0.7, 0.9),
        ('0.9 - 1.0', 0.9, 1.0),
    ]

    for label, low, high in ranges:
        count_before = sum(1 for c in comparisons if low <= c['confidence_before'] < high or (high == 1.0 and c['confidence_before'] == 1.0))
        count_after = sum(1 for c in comparisons if low <= c['confidence_after'] < high or (high == 1.0 and c['confidence_after'] == 1.0))
        print(f"{label:<20} {count_before:<15} {count_after:<15}")
    print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_reports.py <report1_path> <report2_path>")
        print("Example: python compare_reports.py app/output/report_0fe61b4d.yaml app/output/report_73cd4a6f.yaml")
        sys.exit(1)

    report1_path = sys.argv[1]
    report2_path = sys.argv[2]

    try:
        report1 = load_report(report1_path)
        report2 = load_report(report2_path)
    except Exception as e:
        print(f"Error loading reports: {e}")
        sys.exit(1)

    print_comparison(report1, report2)


if __name__ == "__main__":
    main()
