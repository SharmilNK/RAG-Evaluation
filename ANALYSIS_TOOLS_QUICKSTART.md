# Analysis Tools - Quick Start Guide

## Available Tools

### 1. `analyze_report.py` - Report Inspector
**Purpose:** Understand source quality, confidence factors, and identify improvements

**Basic Usage:**
```bash
# Full report analysis
python analyze_report.py app/output/report_73cd4a6f.yaml

# Specific KPI deep-dive
python analyze_report.py app/output/report_73cd4a6f.yaml --kpi strat_ai_vision
```

**Output:**
- Report summary (overall score, pillars)
- Source distribution (tier breakdown, domains, citations)
- Improvement opportunities (grouped by issue type)
- Available KPIs for detailed analysis

**When to Use:**
- ✅ After every new report generation
- ✅ When investigating low confidence scores
- ✅ To understand where sources are from
- ✅ To validate enhanced features are working

---

### 2. `compare_reports.py` - Before/After Comparison
**Purpose:** See the impact of changes (configuration, features, keywords)

**Usage:**
```bash
python compare_reports.py app/output/report_OLD.yaml app/output/report_NEW.yaml
```

**Output:**
- Overall score changes
- Pillar-level improvements
- Biggest confidence improvements (with tier/corroboration details)
- Biggest score changes
- Enhanced features adoption
- Confidence distribution shift

**When to Use:**
- ✅ After enabling/disabling features
- ✅ After changing boost factors
- ✅ After updating keywords
- ✅ To validate improvements
- ✅ Weekly progress tracking

---

### 3. `test_enhanced_features.py` - Feature Validation
**Purpose:** Verify enhanced features are working correctly

**Usage:**
```bash
python test_enhanced_features.py
```

**Output:**
- Tier weighting tests
- Corroboration detection tests
- Dynamic k tuning tests
- Pass/fail results

**When to Use:**
- ✅ After code changes
- ✅ After configuration updates
- ✅ To debug issues
- ✅ Before deployments

---

## Common Workflows

### Workflow 1: Investigate Low Confidence KPI

**Problem:** KPI has confidence <0.5

**Steps:**
```bash
# 1. Identify the KPI
python analyze_report.py app/output/report_XXX.yaml | grep "Confidence: 0.[0-4]"

# 2. Deep-dive the KPI
python analyze_report.py app/output/report_XXX.yaml --kpi <KPI_ID>

# 3. Check the confidence breakdown
#    Look for:
#    - Low Evidence Penalty (-0.2) → Need more sources
#    - Citation Penalty (-0.3) → Missing LLM citations
#    - Low tier quality → Need tier 1 sources
#    - Low corroboration → Sources don't agree

# 4. Check evidence sources
#    - Are there any citations?
#    - What tiers are the sources?
#    - Do the quotes look relevant?

# 5. Fix the root cause
#    - Add keywords (if no evidence)
#    - Fetch more URLs (if low evidence)
#    - Improve query (if poor relevance)
```

**Example Output:**
```
Confidence Breakdown:
  Base Confidence:        0.40
  Low Evidence Penalty   -0.200  ← PROBLEM: <3 chunks
  ======================== ======
  Final Confidence:       0.20

Evidence Summary:
  Citations:         0           ← NO EVIDENCE FOUND
  Unique Sources:    0
```

**Action:** Add more keywords or fetch more sources

---

### Workflow 2: Validate Configuration Change

**Problem:** Changed boost factors, want to see impact

**Steps:**
```bash
# 1. Run baseline report (before change)
python run_analysis.py vodafone.com

# 2. Change configuration in .env
#    Example: VITELIS_TIER1_BOOST=2.0 (was 1.5)

# 3. Run new report (after change)
python run_analysis.py vodafone.com

# 4. Compare reports
python compare_reports.py app/output/report_OLD.yaml app/output/report_NEW.yaml

# 5. Look for:
#    - Confidence improvements (should increase)
#    - KPIs with good tier 1 presence (should improve most)
#    - Overall score stability (shouldn't change drastically)
```

**Example Output:**
```
BIGGEST CONFIDENCE IMPROVEMENTS
prod_ai_pricing_signal   0.70 → 0.85 (+0.15)
  Tier: T1=3, T2=3 (avg=1.50)  ← Good tier mix benefited from boost
```

---

### Workflow 3: Understand Source Distribution

**Problem:** Want to know where evidence is coming from

**Steps:**
```bash
# 1. Run full analysis
python analyze_report.py app/output/report_XXX.yaml

# 2. Check SOURCE DISTRIBUTION ANALYSIS section
#    Look for:
#    - Tier percentages (target: >20% tier 1)
#    - Unique domains (more is better)
#    - Most cited domains (is it diverse?)
#    - Most cited sources (any over-reliance?)
```

**Example Output:**
```
Tier Distribution:
  TIER1       15 citations ( 22.7%)  ← GOOD: >20%
  TIER2       51 citations ( 77.3%)
  TIER3        0 citations (  0.0%)  ← GREAT: No low-quality

Most Cited Domains:
  www.vodafone.com        20 citations  ← Primary domain
  vodafone.com             8 citations
  vb2b.vodafone.com        2 citations  ← Good diversity

Most Cited Sources:
  AI Framework PDF         5 citations  ← Heavily cited (good!)
  CTO AI thoughts          4 citations
```

**Interpretation:**
- ✅ Good tier 1 presence (22.7%)
- ✅ No tier 3 sources
- ✅ Diverse domains
- ⚠️ One source cited 5 times (acceptable, but monitor)

---

### Workflow 4: Weekly Progress Tracking

**Goal:** Track improvements over time

**Steps:**
```bash
# 1. Keep a baseline report
cp app/output/report_CURRENT.yaml baselines/report_week1.yaml

# 2. After making improvements, run new report
python run_analysis.py vodafone.com

# 3. Compare to baseline
python compare_reports.py baselines/report_week1.yaml app/output/report_NEW.yaml

# 4. Track metrics:
#    - Overall score change
#    - Pillar confidence improvements
#    - % of KPIs with confidence >0.9
#    - Average tier quality
#    - Corroboration scores

# 5. Update baseline for next week
cp app/output/report_NEW.yaml baselines/report_week2.yaml
```

**Metrics to Track:**
| Metric | Week 1 | Week 2 | Week 3 | Target |
|--------|--------|--------|--------|--------|
| Avg Confidence | 0.75 | 0.83 | 0.87 | >0.85 |
| % KPIs >0.9 conf | 45% | 55% | 61% | >60% |
| Avg Tier | 1.85 | 1.77 | 1.68 | <1.75 |
| Avg Corroboration | 0.85 | 0.95 | 0.98 | >0.80 |

---

## Key Insights to Look For

### 🟢 Green Flags (Good)
- ✅ Confidence: 0.9 - 1.0
- ✅ Tier avg: <1.8
- ✅ Corroboration: >0.8
- ✅ Unique sources: ≥4
- ✅ Citations: ≥2
- ✅ Tier 1 presence: ≥20%

### 🟡 Yellow Flags (Monitor)
- ⚠️ Confidence: 0.5 - 0.7
- ⚠️ Tier avg: 1.8 - 2.2
- ⚠️ Corroboration: 0.4 - 0.7
- ⚠️ Unique sources: 2-3
- ⚠️ Citations: 1
- ⚠️ Tier 1 presence: 10-20%

### 🔴 Red Flags (Fix)
- ❌ Confidence: <0.5
- ❌ Tier avg: >2.2
- ❌ Corroboration: <0.4
- ❌ Unique sources: 0-1
- ❌ Citations: 0
- ❌ Tier 1 presence: <10%

---

## Interpreting Confidence Breakdowns

### Pattern 1: Perfect Confidence (1.0)
```
Base:                0.50
Tier Boost:         +0.15
Corroboration:      +0.15
Diversity:          +0.05
==========================
Final:               0.85 → 1.0
```
**Interpretation:** Excellent evidence quality, multiple sources agree, high tier quality

---

### Pattern 2: Good Confidence (0.7-0.9)
```
Base:                0.50
Tier Boost:         +0.10
Corroboration:      +0.12
Diversity:          +0.05
==========================
Final:               0.77
```
**Interpretation:** Good evidence, could improve tier quality slightly

---

### Pattern 3: Medium Confidence (0.5-0.7)
```
Base:                0.50
Tier Boost:         +0.05
Corroboration:      +0.08
Diversity:          +0.00
Low Evidence:       -0.20
==========================
Final:               0.43
```
**Interpretation:** Found evidence but not enough (<3 chunks), only 2 sources

---

### Pattern 4: Low Confidence (<0.5)
```
Base:                0.40
Tier Boost:         +0.00
Corroboration:      +0.00
Low Evidence:       -0.20
==========================
Final:               0.20
```
**Interpretation:** No or minimal evidence found, need better keywords or more sources

---

## Quick Reference Card

| Task | Command | Key Output |
|------|---------|------------|
| Full report analysis | `python analyze_report.py <report>` | Source distribution, improvements |
| KPI deep-dive | `python analyze_report.py <report> --kpi <ID>` | Confidence breakdown, evidence |
| Before/after comparison | `python compare_reports.py <old> <new>` | Score changes, confidence shifts |
| Feature validation | `python test_enhanced_features.py` | Pass/fail tests |

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Confidence | >0.9 | 0.5-0.7 | <0.5 |
| Avg Tier | <1.8 | 1.8-2.2 | >2.2 |
| Corroboration | >0.8 | 0.4-0.7 | <0.4 |
| Unique Sources | ≥4 | 2-3 | 0-1 |

| Issue | Root Cause | Solution |
|-------|------------|----------|
| Low confidence | No evidence | Add keywords, fetch more URLs |
| Poor tier quality | Only news sources | Prioritize investor docs, PDFs |
| Low corroboration | Sources disagree | Review evidence relevance |
| Single source | Limited URLs | Expand URL discovery |

---

## Tips & Best Practices

1. **Always run analysis after generating a report** - Don't wait!

2. **Focus on patterns, not individual KPIs** - If multiple KPIs have the same issue, fix the root cause

3. **Track metrics over time** - Use compare tool weekly

4. **Validate improvements** - Always compare before/after when changing config

5. **Deep-dive low confidence KPIs** - These need immediate attention

6. **Monitor tier distribution** - Target >20% tier 1 sources

7. **Check for single-source KPIs** - These are less trustworthy

8. **Use corroboration as a quality signal** - <0.5 means sources don't agree

9. **Test after code changes** - Run test_enhanced_features.py

10. **Document insights** - Keep notes on what works and what doesn't

---

## Troubleshooting

### Problem: "No such file"
**Solution:** Check the report path, use absolute path if needed

### Problem: Test failures
**Solution:** Check .env configuration, ensure features are enabled

### Problem: Confidence always 0.3-0.6
**Solution:** Enable enhanced confidence, check VITELIS_ENHANCED_CONFIDENCE=true

### Problem: No tier distribution in reports
**Solution:** Re-run analysis, ensure tier_weighting.py is being used

### Problem: Compare shows "0 KPIs"
**Solution:** Ensure both reports are for the same company/KPI set

---

## Next Steps

1. Run `python analyze_report.py app/output/report_73cd4a6f.yaml`
2. Identify the 2 low-confidence KPIs
3. Deep-dive with `--kpi` flag
4. Fix root causes (keywords, sources)
5. Compare before/after
6. Iterate!

For detailed insights and recommendations, see [ANALYSIS_INSIGHTS.md](ANALYSIS_INSIGHTS.md)
