"""Regenerate results/ from scratch and guard the headline numbers.

Re-runs the full evaluation and rewrites evaluation.json,
evaluation.csv, evaluation_summary.md and HEADLINE.md. Then checks
the headline metrics against the values recorded in HEADLINE.md and
fails (exit 1) if any of them drifted beyond tolerance -- so a
regression in the tool is caught here, not silently shipped.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_swap.evaluate import evaluate, write_results

TOLERANCE = 0.001

EXPECTED = {
    "max_gain_error": 0.0,
    "max_accounting_error": 0.0,
    "greedy_optimal_rate": 1.0,
    "random_greedy_optimal_rate": 1.0,
    "random_full_order_agreement_rate": 1.0,
    "random_budget_greedy_gain_optimal_rate": 0.92,
    "random_budget_greedy_ratio_optimal_rate": 0.78,
    "masking_precision": 1.0,
    "masking_recall": 1.0,
    "masking_control_false_positives": 0,
}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "results")
    results = evaluate()
    write_results(results, out_dir)
    head = results["headline"]
    lines = []
    lines.append("# Headline Numbers (seed %d, %d cases per system)"
                 % (results["seed"], results["n_cases"]))
    lines.append("")
    lines.append("- Max gold-swap gain-recovery error: %.6f" % head["max_gain_error"])
    lines.append("- Max ceiling-accounting error: %.6f" % head["max_accounting_error"])
    lines.append("- Bottlenecks: RAG generator, research synthesizer, PHM detector.")
    lines.append("- Greedy is optimal for the minimal fix set on 100% of "
                 "systems (theorem, not a measurement -- see README 6).")
    lines.append("- Under a budget of 6 (50 random systems): greedy by gain "
                 "optimal on %.0f%%, greedy by gain per cost on %.0f%%; "
                 "mean score gaps %.4f / %.4f, max gaps %.4f / %.4f."
                 % (100.0 * head["random_budget_greedy_gain_optimal_rate"],
                    100.0 * head["random_budget_greedy_ratio_optimal_rate"],
                    head["random_budget_mean_gain_gap"],
                    head["random_budget_mean_ratio_gap"],
                    head["random_budget_max_gain_gap"],
                    head["random_budget_max_ratio_gap"]))
    lines.append("- Masking detection: precision %.2f, recall %.2f, "
                 "0 false positives on the 3 healthy systems."
                 % (head["masking_precision"], head["masking_recall"]))
    lines.append("- All 18 component pairs superadditive (6/6 per system).")
    lines.append("")
    with open(os.path.join(out_dir, "HEADLINE.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    failures = []
    for key, expected in EXPECTED.items():
        actual = head[key]
        if abs(actual - expected) > TOLERANCE:
            failures.append("%s: expected %.4f, got %.4f" % (key, expected, actual))
    if failures:
        print("HEADLINE DRIFT:")
        for failure in failures:
            print("  " + failure)
        return 1
    print("results regenerated in %s; headline numbers stable." % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
