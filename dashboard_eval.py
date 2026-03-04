"""
dashboard_eval.py — Ground-Truth Evaluation Dashboard
Run with: streamlit run dashboard_eval.py

Loads all app/output/eval_*.json files and displays three tabs:
  1. KPI Comparison  — pipeline score vs analyst answer, side by side
  2. Source Analysis — which URLs were cited most (pipeline vs analyst)
  3. Unmatched       — KPIs or data points with no counterpart
"""
from __future__ import annotations

import json
import os
from collections import Counter
from glob import glob

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────── #
# Page config
# ─────────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Vitelis — Ground Truth Comparison",
    page_icon="🔍",
    layout="wide",
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "app", "output")


# ─────────────────────────────────────────────────────────────────────────── #
# Data loading
# ─────────────────────────────────────────────────────────────────────────── #
@st.cache_data(ttl=60)
def load_all_eval_reports() -> dict[str, dict]:
    """Load all eval_*.json files, keyed by company name."""
    pattern = os.path.join(EVAL_DIR, "eval_*.json")
    reports: dict[str, dict] = {}
    for path in sorted(glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            company = data.get("company_name", os.path.basename(path))
            # If multiple runs for same company, keep the latest (last alphabetically by run_id)
            if company not in reports or data.get("run_id", "") > reports[company].get("run_id", ""):
                reports[company] = data
        except Exception:
            pass
    return reports


# ─────────────────────────────────────────────────────────────────────────── #
# Helper: build comparison dataframe
# ─────────────────────────────────────────────────────────────────────────── #
def build_comparison_df(report: dict) -> pd.DataFrame:
    rows = []
    for c in report.get("comparisons", []):
        rows.append({
            "KPI ID": c.get("kpi_id", ""),
            "KPI Name (pipeline)": c.get("kpi_name", ""),
            "Pipeline Score": c.get("pipeline_score", 0),
            "GT Data Point": c.get("ground_truth_name", ""),
            "GT Answer": c.get("ground_truth_answer", ""),
            "GT Explanation": c.get("ground_truth_explanation", ""),
            "Match Confidence": c.get("match_confidence", 0),
            "Pipeline Sources": c.get("pipeline_sources", []),
            "GT Sources": c.get("ground_truth_sources", []),
            "Pipeline Rationale": c.get("pipeline_rationale", ""),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────── #
# Main app
# ─────────────────────────────────────────────────────────────────────────── #
st.title("🔍 Ground Truth Evaluation Dashboard")
st.caption("Compares the pipeline's KPI scores against analyst research findings.")

all_reports = load_all_eval_reports()

if not all_reports:
    st.warning(
        "No evaluation reports found in `app/output/eval_*.json`.\n\n"
        "Run `python run_eval.py --company-folder \"Orange S.A\"` first."
    )
    st.stop()

# ── Company selector ─────────────────────────────────────────────────────── #
companies = sorted(all_reports.keys())
selected_company = st.selectbox("Select company", companies)
report = all_reports[selected_company]

col1, col2, col3, col4 = st.columns(4)
comparisons = report.get("comparisons", [])
col1.metric("Run ID", report.get("run_id", "—"))
col2.metric("Matched KPIs", len(comparisons))
col3.metric("Unmatched KPIs", len(report.get("unmatched_kpis", [])))
col4.metric("Unmatched GT points", len(report.get("unmatched_data_points", [])))

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────── #
tab1, tab2, tab3 = st.tabs(["📊 KPI Comparison", "🔗 Source Analysis", "⚠️ Unmatched"])

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — KPI Comparison
# ════════════════════════════════════════════════════════════════════════════ #
with tab1:
    st.subheader("Pipeline Score vs Analyst Ground Truth")

    df = build_comparison_df(report)
    if df.empty:
        st.info("No matched comparisons to display.")
    else:
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        min_confidence = filter_col1.slider(
            "Minimum match confidence", 0.0, 1.0, 0.35, step=0.05
        )
        score_filter = filter_col2.multiselect(
            "Filter by pipeline score",
            options=[1, 2, 3, 4, 5],
            default=[],
        )

        filtered = df[df["Match Confidence"] >= min_confidence]
        if score_filter:
            filtered = filtered[filtered["Pipeline Score"].isin(score_filter)]

        st.caption(f"Showing {len(filtered)} of {len(df)} matched KPI pairs")

        # Display table — core columns
        display_cols = [
            "KPI Name (pipeline)", "Pipeline Score", "GT Data Point",
            "GT Answer", "Match Confidence",
        ]
        st.dataframe(
            filtered[display_cols].style.background_gradient(
                subset=["Pipeline Score"], cmap="RdYlGn", vmin=1, vmax=5
            ).background_gradient(
                subset=["Match Confidence"], cmap="Blues", vmin=0, vmax=1
            ),
            use_container_width=True,
            height=400,
        )

        # Detail expander — click to see full rationale & sources
        st.subheader("KPI Detail")
        kpi_names = filtered["KPI Name (pipeline)"].tolist()
        if kpi_names:
            selected_kpi = st.selectbox("Select a KPI for full detail", kpi_names)
            row = filtered[filtered["KPI Name (pipeline)"] == selected_kpi].iloc[0]

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Pipeline output**")
                st.metric("Score", row["Pipeline Score"])
                st.markdown(f"*Rationale:* {row['Pipeline Rationale']}")
                if row["Pipeline Sources"]:
                    with st.expander("Pipeline sources cited"):
                        for url in row["Pipeline Sources"]:
                            st.markdown(f"- [{url}]({url})")

            with detail_col2:
                st.markdown("**Analyst ground truth**")
                st.markdown(f"**Data point:** {row['GT Data Point']}")
                st.markdown(f"**Answer:** `{row['GT Answer']}`")
                st.markdown(f"*Explanation:* {row['GT Explanation']}")
                if row["GT Sources"]:
                    with st.expander(f"Analyst sources ({len(row['GT Sources'])})"):
                        for url in row["GT Sources"][:20]:
                            st.markdown(f"- [{url}]({url})")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Source Analysis
# ════════════════════════════════════════════════════════════════════════════ #
with tab2:
    st.subheader("Source Attribution")

    if not comparisons:
        st.info("No comparisons available.")
    else:
        # Pipeline sources
        all_pipeline_urls: list[str] = []
        for c in comparisons:
            all_pipeline_urls.extend(c.get("pipeline_sources", []))

        # Analyst sources
        all_gt_urls: list[str] = []
        for c in comparisons:
            all_gt_urls.extend(c.get("ground_truth_sources", []))

        src_col1, src_col2 = st.columns(2)

        with src_col1:
            st.markdown("**Top Pipeline Sources**  *(URLs cited most across KPIs)*")
            pipeline_counts = Counter(all_pipeline_urls).most_common(20)
            if pipeline_counts:
                pipe_df = pd.DataFrame(pipeline_counts, columns=["URL", "Citations"])
                st.dataframe(pipe_df, use_container_width=True, height=350)
            else:
                st.info("No pipeline source citations found.")

        with src_col2:
            st.markdown("**Top Analyst Sources**  *(URLs used most across data points)*")
            gt_counts = Counter(all_gt_urls).most_common(20)
            if gt_counts:
                gt_df = pd.DataFrame(gt_counts, columns=["URL", "Citations"])
                st.dataframe(gt_df, use_container_width=True, height=350)
            else:
                st.info("No analyst source URLs found.")

        st.divider()
        st.subheader("Per-KPI Source Overlap")
        st.caption("How much do the pipeline's cited URLs overlap with the analyst's URLs for each KPI?")

        overlap_rows = []
        for c in comparisons:
            p_set = set(c.get("pipeline_sources", []))
            g_set = set(c.get("ground_truth_sources", []))
            if p_set or g_set:
                overlap = len(p_set & g_set)
                union = len(p_set | g_set)
                jaccard = round(overlap / union, 3) if union > 0 else 0.0
                overlap_rows.append({
                    "KPI": c.get("kpi_name", c.get("kpi_id", "")),
                    "Pipeline URLs": len(p_set),
                    "Analyst URLs": len(g_set),
                    "Shared URLs": overlap,
                    "Jaccard similarity": jaccard,
                })

        if overlap_rows:
            overlap_df = pd.DataFrame(overlap_rows).sort_values("Jaccard similarity", ascending=False)
            st.dataframe(
                overlap_df.style.background_gradient(
                    subset=["Jaccard similarity"], cmap="Greens", vmin=0, vmax=1
                ),
                use_container_width=True,
                height=350,
            )

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Unmatched
# ════════════════════════════════════════════════════════════════════════════ #
with tab3:
    st.subheader("Unmatched Items")

    unmatched_kpis = report.get("unmatched_kpis", [])
    unmatched_gt = report.get("unmatched_data_points", [])

    um_col1, um_col2 = st.columns(2)

    with um_col1:
        st.markdown(f"**KPIs with no ground-truth match** ({len(unmatched_kpis)})")
        st.caption(
            "These KPIs were scored by the pipeline but could not be matched "
            "to any analyst data point by name."
        )
        if unmatched_kpis:
            st.dataframe(pd.DataFrame({"KPI Name": unmatched_kpis}), use_container_width=True)
        else:
            st.success("All pipeline KPIs were matched.")

    with um_col2:
        st.markdown(f"**Analyst data points with no KPI match** ({len(unmatched_gt)})")
        st.caption(
            "These analyst findings exist in the ground truth but no pipeline KPI "
            "was matched to them — they may represent gaps in the KPI set."
        )
        if unmatched_gt:
            st.dataframe(
                pd.DataFrame({"Ground Truth Data Point": unmatched_gt}),
                use_container_width=True,
            )
        else:
            st.success("All analyst data points were matched.")
