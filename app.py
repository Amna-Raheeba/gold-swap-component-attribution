"""Streamlit dashboard for Gold-Swap Component Attribution.

Three tabs: Attribution (per-component gains and ceiling shares),
Fix Order (greedy vs optimal, plus the budget comparison), and
Evaluation (headline metrics across all systems).
"""

import streamlit as st

from gold_swap.attribution import attribute
from gold_swap.evaluate import evaluate
from gold_swap.fixorder import budget_fix, compare_fix_orders
from gold_swap.masking import find_masked
from gold_swap.systems import SYSTEMS, make_cases

st.set_page_config(page_title="Gold-Swap Component Attribution", layout="wide")
st.title("Gold-Swap Component Attribution")
st.write(
    "Replace each component with its gold (ideal) version, hold the "
    "rest fixed, and measure what it could win. The biggest gainer "
    "is the weakest link."
)

system_key = st.sidebar.selectbox("System", sorted(SYSTEMS))
n_cases = st.sidebar.slider("Cases", 20, 500, 200, step=20)
seed = st.sidebar.number_input("Seed", value=7, step=1)
target = st.sidebar.slider("Target score", 0.80, 0.99, 0.95, step=0.01)
budget = st.sidebar.slider("Fix budget", 1.0, 11.0, 6.0, step=1.0)

system = SYSTEMS[system_key]
cases = make_cases(system, n=n_cases, seed=int(seed))
report = attribute(system, cases)

tab_attr, tab_fix, tab_eval = st.tabs(["Attribution", "Fix Order", "Evaluation"])

with tab_attr:
    st.subheader("Attribution: %s" % system.name)
    st.write("Baseline mean score %.4f over %d cases; ceiling %.4f."
             % (report.baseline, report.n_cases, report.ceiling))
    st.table({
        "Component": [a.name for a in report.attributions],
        "Gold-swap gain": ["%.4f" % a.gain for a in report.attributions],
        "Ceiling share": ["%.3f" % a.ceiling_share for a in report.attributions],
        "Bottleneck rank": [a.bottleneck_rank for a in report.attributions],
    })
    st.write("Bottleneck: **%s**. Interaction residual: %.4f "
             "(positive = superadditive gains)."
             % (report.bottleneck(), report.interaction_residual))
    st.subheader("Pairwise interactions")
    st.table({
        "Pair": sorted(report.pairwise),
        "Interaction": ["%+.4f" % report.pairwise[p] for p in sorted(report.pairwise)],
    })

with tab_fix:
    st.subheader("Fix order (target %.2f)" % target)
    fix = compare_fix_orders(system, cases, target=target)
    st.write("Greedy order: **%s**." % ", ".join(fix.greedy_order))
    st.write("Scores after each fix: %s."
             % ", ".join("%.4f" % s for s in fix.greedy_scores))
    st.write("Optimal minimal fix set: **%s** (%d components). Greedy is "
             "optimal here: %s."
             % (", ".join(fix.optimal_set), fix.optimal_size,
                fix.greedy_is_optimal))
    st.subheader("Budget fix (budget %.1f)" % budget)
    bfix = budget_fix(system, cases, budget=budget)
    st.table({
        "Strategy": ["Optimal (exhaustive)", "Greedy by gain",
                     "Greedy by gain per cost"],
        "Subset": [", ".join(bfix.optimal_subset),
                   ", ".join(bfix.greedy_gain_subset),
                   ", ".join(bfix.greedy_ratio_subset)],
        "Score": ["%.4f" % bfix.optimal_score,
                  "%.4f" % bfix.greedy_gain_score,
                  "%.4f" % bfix.greedy_ratio_score],
        "Cost": ["%.1f" % bfix.optimal_cost,
                 "%.1f" % bfix.greedy_gain_cost,
                 "%.1f" % bfix.greedy_ratio_cost],
    })
    masked = find_masked(system, cases)
    st.subheader("Masked components")
    if masked:
        for m in masked:
            st.write("- **%s** masked by **%s** (ratio %.3f)."
                     % (m.masked, m.masker, m.suppression_ratio))
    else:
        st.write("None found in this system.")

with tab_eval:
    st.subheader("Evaluation headline")
    results = evaluate()
    head = results["headline"]
    st.table({
        "Metric": [
            "Max gain-recovery error",
            "Max ceiling-accounting error",
            "Greedy optimal rate (minimal fix set)",
            "Budget: greedy by gain optimal (50 random)",
            "Budget: greedy by gain/cost optimal (50 random)",
            "Budget mean gaps (gain / ratio)",
            "Masking precision / recall",
            "Masking false positives (healthy systems)",
        ],
        "Value": [
            "%.6f" % head["max_gain_error"],
            "%.6f" % head["max_accounting_error"],
            "%.3f" % head["greedy_optimal_rate"],
            "%.2f" % head["random_budget_greedy_gain_optimal_rate"],
            "%.2f" % head["random_budget_greedy_ratio_optimal_rate"],
            "%.4f / %.4f" % (head["random_budget_mean_gain_gap"],
                             head["random_budget_mean_ratio_gap"]),
            "%.2f / %.2f" % (head["masking_precision"], head["masking_recall"]),
            "%d" % head["masking_control_false_positives"],
        ],
    })
    st.write("Bottlenecks: %s." % head["bottlenecks"])
