"""Evaluation: run the tool on all three systems and measure it.

Every number the evaluation reports is a measurement of the tool
against ground truth the tool never sees:

* **Gain recovery** -- experimental gold-swap gains vs analytic gains
  computed from the known model. The tool only runs black-box
  experiments; the analytic values are an independent check. Any
  disagreement is a bug in the runner.
* **Ceiling accounting** -- the ceiling shares plus the interaction
  residual must sum to exactly 1 (an accounting identity).
* **Greedy vs optimal** -- how often the greedy fix order reaches a
  0.95 target with the minimal number of fixes, and the cumulative
  score gap when it does not.
* **Masking detection** -- precision/recall on the deliberately
  broken phm_masked system, where the near-dead detector masks the
  diagnoser and reporter by construction.
* **Interaction profile** -- how many component pairs are
  superadditive vs subadditive (the model predicts superadditivity
  everywhere: fixing two components together wins more than the sum
  of fixing each alone).
"""

import csv
import json
import os

from .attribution import attribute
from .fixorder import budget_fix, compare_fix_orders
from .masking import find_masked
from .swap import analytic_gains, gold_swap_gains
from .systems import SYSTEMS, make_cases

TARGET = 0.95
N_CASES = 200
SEED = 7


def evaluate_system(system, cases, target=TARGET, budget=6.0):
    """Full evaluation of one system; returns a metrics dict."""
    report = attribute(system, cases)
    fix = compare_fix_orders(system, cases, target=target)
    budget_result = budget_fix(system, cases, budget=budget)
    experimental = gold_swap_gains(system, cases)
    analytic = analytic_gains(system, cases)
    max_gain_error = max(
        abs(experimental["gains"][name] - analytic[name])
        for name in system.names()
    )
    shares_sum = sum(a.ceiling_share for a in report.attributions)
    accounting_error = abs(shares_sum + report.interaction_residual - 1.0)
    superadditive = sum(1 for v in report.pairwise.values() if v > 1e-12)
    subadditive = sum(1 for v in report.pairwise.values() if v < -1e-12)
    return {
        "system": system.name,
        "n_cases": len(cases),
        "baseline": report.baseline,
        "ceiling": report.ceiling,
        "bottleneck": report.bottleneck(),
        "gains": {a.name: a.gain for a in report.attributions},
        "ceiling_shares": {a.name: a.ceiling_share for a in report.attributions},
        "interaction_residual": report.interaction_residual,
        "max_gain_error": max_gain_error,
        "accounting_error": accounting_error,
        "greedy_order": fix.greedy_order,
        "greedy_steps_to_target": fix.greedy_steps_to_target,
        "optimal_set": fix.optimal_set,
        "optimal_size": fix.optimal_size,
        "greedy_is_optimal": fix.greedy_is_optimal,
        "greedy_full_order": fix.greedy_full_order,
        "full_order_agreement": fix.full_order_agreement,
        "cumulative_order_gap": fix.cumulative_order_gap,
        "budget": budget,
        "budget_optimal_subset": budget_result.optimal_subset,
        "budget_optimal_score": budget_result.optimal_score,
        "budget_optimal_cost": budget_result.optimal_cost,
        "budget_greedy_gain_subset": budget_result.greedy_gain_subset,
        "budget_greedy_gain_score": budget_result.greedy_gain_score,
        "budget_greedy_gain_gap": (
            budget_result.optimal_score - budget_result.greedy_gain_score
        ),
        "budget_greedy_gain_is_optimal": budget_result.greedy_gain_is_optimal,
        "budget_greedy_ratio_subset": budget_result.greedy_ratio_subset,
        "budget_greedy_ratio_score": budget_result.greedy_ratio_score,
        "budget_greedy_ratio_gap": (
            budget_result.optimal_score - budget_result.greedy_ratio_score
        ),
        "budget_greedy_ratio_is_optimal": budget_result.greedy_ratio_is_optimal,
        "score_gap_after_two_fixes": fix.score_gap_after_two_fixes,
        "pairwise_superadditive": superadditive,
        "pairwise_subadditive": subadditive,
        "pairwise_total": len(report.pairwise),
    }


def evaluate_masking():
    """Masking detection on the deliberately broken system."""
    system = SYSTEMS["phm_masked"]
    cases = make_cases(system, n=N_CASES, seed=SEED)
    found = find_masked(system, cases)
    # Ground truth by construction: the near-dead detector suppresses
    # every other component's gain to ~38% of its unmasked value, so
    # all three siblings are masked by it, and the detector itself
    # (whose gain alone is large) is masked by nothing.
    expected = {
        ("sensor", "detector"),
        ("diagnoser", "detector"),
        ("reporter", "detector"),
    }
    got = {(m.masked, m.masker) for m in found}
    true_positives = len(got & expected)
    precision = true_positives / len(got) if got else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    # Control: the three healthy systems must report no masking.
    control_found = 0
    for key in ("rag", "research", "phm"):
        control_found += len(find_masked(SYSTEMS[key], make_cases(SYSTEMS[key], n=N_CASES, seed=SEED)))
    return {
        "system": "phm_masked",
        "found": sorted("%s masked by %s" % (m.masked, m.masker) for m in found),
        "expected": sorted("%s masked by %s" % pair for pair in expected),
        "precision": precision,
        "recall": recall,
        "n_found": len(found),
        "control_false_positives": control_found,
    }


def evaluate_random_systems(n_systems=50, n_cases=100, seed=1234, target=TARGET, budget=6.0):
    """Greedy vs optimal across random systems.

    Two comparisons, with opposite characters:

    * Without a budget, greedy is PROVABLY optimal for the minimal
      fix set and (empirically, on all 53 systems) for the full
      order too -- in the series model the score after fixing a set
      depends only on the product of the unfixed reliabilities, so
      the greedy order is the optimal order. The comparison is kept
      as a check that the implementation honours the theorem.
    * Under a budget the problem becomes knapsack-like, and both
      greedy strategies (by gain, by gain per cost) can genuinely
      lose to the exhaustive optimum. That is the real fix-order
      measurement: agreement rates and mean score gaps.
    """
    import random as _random

    from .fixorder import budget_fix
    from .models import Component, System

    rng = _random.Random(seed)
    agreements = 0
    full_agreements = 0
    gaps = []
    two_fix_gaps = []
    budget_gain_agree = 0
    budget_ratio_agree = 0
    budget_gain_gaps = []
    budget_ratio_gaps = []
    for i in range(n_systems):
        specs = [
            ("c%d" % j, rng.uniform(0.5, 0.95), "", float(rng.randint(1, 5)))
            for j in range(4)
        ]
        system = System(
            name="rand-%02d" % i,
            description="random system",
            components=tuple(Component(*s) for s in specs),
        )
        cases = make_cases(system, n=n_cases, seed=seed + i)
        fix = compare_fix_orders(system, cases, target=target)
        agreements += 1 if fix.greedy_is_optimal else 0
        full_agreements += 1 if fix.full_order_agreement else 0
        gaps.append(fix.cumulative_order_gap)
        two_fix_gaps.append(fix.score_gap_after_two_fixes)
        bfix = budget_fix(system, cases, budget=budget)
        budget_gain_agree += 1 if bfix.greedy_gain_is_optimal else 0
        budget_ratio_agree += 1 if bfix.greedy_ratio_is_optimal else 0
        budget_gain_gaps.append(bfix.optimal_score - bfix.greedy_gain_score)
        budget_ratio_gaps.append(bfix.optimal_score - bfix.greedy_ratio_score)
    return {
        "n_systems": n_systems,
        "budget": budget,
        "greedy_optimal_rate": agreements / n_systems,
        "full_order_agreement_rate": full_agreements / n_systems,
        "mean_cumulative_order_gap": sum(gaps) / len(gaps),
        "max_cumulative_order_gap": max(gaps),
        "mean_score_gap_after_two_fixes": sum(two_fix_gaps) / len(two_fix_gaps),
        "budget_greedy_gain_optimal_rate": budget_gain_agree / n_systems,
        "budget_greedy_ratio_optimal_rate": budget_ratio_agree / n_systems,
        "budget_mean_gain_gap": sum(budget_gain_gaps) / len(budget_gain_gaps),
        "budget_max_gain_gap": max(budget_gain_gaps),
        "budget_mean_ratio_gap": sum(budget_ratio_gaps) / len(budget_ratio_gaps),
        "budget_max_ratio_gap": max(budget_ratio_gaps),
    }


def evaluate(seed=SEED, n_cases=N_CASES, target=TARGET):
    """Evaluate all three systems plus the masking demo."""
    systems = {}
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        cases = make_cases(system, n=n_cases, seed=seed)
        systems[key] = evaluate_system(system, cases, target=target)
    masking = evaluate_masking()
    random_systems = evaluate_random_systems()
    greedy_optimal_rate = sum(
        1 for m in systems.values() if m["greedy_is_optimal"]
    ) / len(systems)
    return {
        "seed": seed,
        "n_cases": n_cases,
        "target": target,
        "systems": systems,
        "masking": masking,
        "random_systems": random_systems,
        "headline": {
            "max_gain_error": max(
                m["max_gain_error"] for m in systems.values()
            ),
            "max_accounting_error": max(
                m["accounting_error"] for m in systems.values()
            ),
            "greedy_optimal_rate": greedy_optimal_rate,
            "random_greedy_optimal_rate": random_systems["greedy_optimal_rate"],
            "random_full_order_agreement_rate": random_systems[
                "full_order_agreement_rate"
            ],
            "random_budget_greedy_gain_optimal_rate": random_systems[
                "budget_greedy_gain_optimal_rate"
            ],
            "random_budget_greedy_ratio_optimal_rate": random_systems[
                "budget_greedy_ratio_optimal_rate"
            ],
            "random_budget_mean_gain_gap": random_systems[
                "budget_mean_gain_gap"
            ],
            "random_budget_max_gain_gap": random_systems[
                "budget_max_gain_gap"
            ],
            "random_budget_mean_ratio_gap": random_systems[
                "budget_mean_ratio_gap"
            ],
            "random_budget_max_ratio_gap": random_systems[
                "budget_max_ratio_gap"
            ],
            "random_mean_cumulative_order_gap": random_systems[
                "mean_cumulative_order_gap"
            ],
            "random_max_cumulative_order_gap": random_systems[
                "max_cumulative_order_gap"
            ],
            "random_mean_score_gap_after_two_fixes": random_systems[
                "mean_score_gap_after_two_fixes"
            ],
            "mean_cumulative_order_gap": sum(
                m["cumulative_order_gap"] for m in systems.values()
            )
            / len(systems),
            "mean_score_gap_after_two_fixes": sum(
                m["score_gap_after_two_fixes"] for m in systems.values()
            )
            / len(systems),
            "masking_precision": masking["precision"],
            "masking_recall": masking["recall"],
            "masking_control_false_positives": masking["control_false_positives"],
            "bottlenecks": {
                key: m["bottleneck"] for key, m in systems.items()
            },
        },
    }


def write_results(results, out_dir):
    """Write evaluation.json, evaluation.csv and evaluation_summary.md."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "evaluation.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "system", "baseline", "ceiling", "bottleneck",
            "greedy_steps_to_target", "optimal_size", "greedy_is_optimal",
            "cumulative_order_gap", "max_gain_error",
        ])
        for key in ("rag", "research", "phm"):
            m = results["systems"][key]
            writer.writerow([
                key, "%.4f" % m["baseline"], "%.4f" % m["ceiling"],
                m["bottleneck"], m["greedy_steps_to_target"],
                m["optimal_size"], m["greedy_is_optimal"],
                "%.4f" % m["cumulative_order_gap"],
                "%.6f" % m["max_gain_error"],
            ])
    lines = []
    lines.append("# Gold-Swap Evaluation Summary")
    lines.append("")
    lines.append("Seed %d, %d cases per system, target score %.2f."
                 % (results["seed"], results["n_cases"], results["target"]))
    lines.append("")
    head = results["headline"]
    lines.append("- Max gain-recovery error: %.6f" % head["max_gain_error"])
    lines.append("- Max ceiling-accounting error: %.6f" % head["max_accounting_error"])
    lines.append("- Greedy reaches the target with the minimal fix set on %.0f%% of the "
                 "3 hand-built systems and %.0f%% of 50 random systems -- "
                 "expected: greedy is provably optimal for the minimal "
                 "fix set in the series model (see README section 6), "
                 "and its full order matched the optimal full order on "
                 "%.0f%% of random systems. The real fix-order "
                 "measurement is under a budget of 6: greedy by gain "
                 "matches the exhaustive optimum on %.0f%% of random "
                 "systems (mean score gap %.4f, max %.4f), greedy by "
                 "gain per cost on %.0f%% (mean gap %.4f, max %.4f)."
                 % (100.0 * head["greedy_optimal_rate"],
                    100.0 * head["random_greedy_optimal_rate"],
                    100.0 * head["random_full_order_agreement_rate"],
                    100.0 * head["random_budget_greedy_gain_optimal_rate"],
                    head["random_budget_mean_gain_gap"],
                    head["random_budget_max_gain_gap"],
                    100.0 * head["random_budget_greedy_ratio_optimal_rate"],
                    head["random_budget_mean_ratio_gap"],
                    head["random_budget_max_ratio_gap"]))
    lines.append("- Masking detection: precision %.2f, recall %.2f; "
                 "false positives on the 3 healthy systems: %d."
                 % (head["masking_precision"], head["masking_recall"],
                    head["masking_control_false_positives"]))
    lines.append("")
    for key in ("rag", "research", "phm"):
        m = results["systems"][key]
        lines.append("## %s" % key)
        lines.append("")
        lines.append("- Baseline %.4f, ceiling %.4f, bottleneck %s."
                     % (m["baseline"], m["ceiling"], m["bottleneck"]))
        lines.append("- Gains: %s."
                     % ", ".join("%s %.4f" % (n, g) for n, g in m["gains"].items()))
        lines.append("- Ceiling shares: %s; interaction residual %.4f."
                     % (", ".join("%s %.3f" % (n, s) for n, s in m["ceiling_shares"].items()),
                        m["interaction_residual"]))
        lines.append("- Greedy fix order: %s (%d steps to %.2f); optimal set: %s."
                     % (", ".join(m["greedy_order"]), m["greedy_steps_to_target"],
                        results["target"], ", ".join(m["optimal_set"])))
        lines.append("- Budget %.1f: optimal %s (score %.4f, cost %.1f); "
                     "greedy by gain %s (score %.4f, gap %.4f); greedy by "
                     "gain per cost %s (score %.4f, gap %.4f)."
                     % (m["budget"], ", ".join(m["budget_optimal_subset"]),
                        m["budget_optimal_score"], m["budget_optimal_cost"],
                        ", ".join(m["budget_greedy_gain_subset"]),
                        m["budget_greedy_gain_score"],
                        m["budget_greedy_gain_gap"],
                        ", ".join(m["budget_greedy_ratio_subset"]),
                        m["budget_greedy_ratio_score"],
                        m["budget_greedy_ratio_gap"]))
        lines.append("")
    with open(os.path.join(out_dir, "evaluation_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return out_dir
