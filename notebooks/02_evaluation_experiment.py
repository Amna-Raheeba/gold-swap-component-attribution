"""Notebook 02: evaluation experiment -- reproduce the headline numbers.

Run with: python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_swap.evaluate import evaluate


def main():
    results = evaluate()
    head = results["headline"]
    print("max gain-recovery error: %.6f" % head["max_gain_error"])
    print("max ceiling-accounting error: %.6f" % head["max_accounting_error"])
    print("bottlenecks: %s" % head["bottlenecks"])
    print("greedy optimal rate (minimal fix set): %.3f" % head["greedy_optimal_rate"])
    print("random full-order agreement rate: %.3f"
          % head["random_full_order_agreement_rate"])
    print("budget (n=50 random systems): greedy by gain optimal %.2f, "
          "greedy by gain per cost optimal %.2f"
          % (head["random_budget_greedy_gain_optimal_rate"],
             head["random_budget_greedy_ratio_optimal_rate"]))
    print("budget mean gaps: gain %.4f, ratio %.4f"
          % (head["random_budget_mean_gain_gap"],
             head["random_budget_mean_ratio_gap"]))
    print("masking precision=%.2f recall=%.2f, control false positives=%d"
          % (head["masking_precision"], head["masking_recall"],
             head["masking_control_false_positives"]))


if __name__ == "__main__":
    main()
