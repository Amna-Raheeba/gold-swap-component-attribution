"""Markdown rendering of attribution and fix-order reports."""


def render_attribution(report):
    lines = []
    lines.append("# Gold-Swap Attribution: %s" % report.system_name)
    lines.append("")
    lines.append("Baseline mean score %.4f over %d cases; all-gold %.4f; "
                 "ceiling %.4f."
                 % (report.baseline, report.n_cases, report.all_gold,
                    report.ceiling))
    lines.append("")
    lines.append("| Rank | Component | Gold-swap gain | Ceiling share |")
    lines.append("| ---: | --------- | -------------: | ------------: |")
    for a in report.attributions:
        lines.append("| %d | %s | %.4f | %.3f |"
                     % (a.bottleneck_rank, a.name, a.gain, a.ceiling_share))
    lines.append("")
    lines.append("Interaction residual (1 - sum of shares): %.4f. "
                 "A positive residual means the gains are superadditive: "
                 "fixing components together wins more than the sum of "
                 "fixing each alone, so the individual gains leave "
                 "headroom unaccounted for." % report.interaction_residual)
    lines.append("")
    if report.pairwise:
        lines.append("## Pairwise interactions")
        lines.append("")
        for pair, value in sorted(report.pairwise.items()):
            kind = "superadditive" if value > 1e-12 else (
                "subadditive" if value < -1e-12 else "independent")
            lines.append("- %s: %+.4f (%s)" % (pair, value, kind))
        lines.append("")
    return "\n".join(lines)


def render_fix_order(result):
    lines = []
    lines.append("# Fix Order: %s (target %.2f)" % (result.system_name, result.target))
    lines.append("")
    lines.append("Greedy order: %s." % ", ".join(result.greedy_order))
    lines.append("")
    lines.append("Scores after each fix: %s."
                 % ", ".join("%.4f" % s for s in result.greedy_scores))
    lines.append("")
    lines.append("Greedy reaches the target in %d steps; the optimal "
                 "minimal fix set is %s (%d components)."
                 % (result.greedy_steps_to_target,
                    ", ".join(result.optimal_set) or "none",
                    result.optimal_size))
    lines.append("")
    gap = result.cumulative_order_gap
    if abs(gap) < 1e-9:
        gap = 0.0
    lines.append("Greedy is optimal here: %s. Cumulative order gap "
                 "(optimal minus greedy total score): %.4f."
                 % (result.greedy_is_optimal, gap))
    lines.append("")
    return "\n".join(lines)


def render_budget(result):
    lines = []
    lines.append("# Budget Fix: %s (budget %.1f)" % (result.system_name, result.budget))
    lines.append("")
    lines.append("Baseline %.4f." % result.baseline)
    lines.append("")
    lines.append("- Optimal: %s (score %.4f, cost %.1f)."
                 % (", ".join(result.optimal_subset) or "nothing",
                    result.optimal_score, result.optimal_cost))
    lines.append("- Greedy by gain: %s (score %.4f, cost %.1f; gap %.4f)."
                 % (", ".join(result.greedy_gain_subset) or "nothing",
                    result.greedy_gain_score, result.greedy_gain_cost,
                    result.optimal_score - result.greedy_gain_score))
    lines.append("- Greedy by gain per cost: %s (score %.4f, cost %.1f; gap %.4f)."
                 % (", ".join(result.greedy_ratio_subset) or "nothing",
                    result.greedy_ratio_score, result.greedy_ratio_cost,
                    result.optimal_score - result.greedy_ratio_score))
    lines.append("")
    return "\n".join(lines)


def render_masking(results):
    lines = []
    lines.append("# Masked Components")
    lines.append("")
    if not results:
        lines.append("No masked components found.")
        lines.append("")
        return "\n".join(lines)
    for r in results:
        lines.append("- **%s** is masked by **%s**: gain alone %.4f, "
                     "gain with the masker fixed %.4f (suppression "
                     "ratio %.3f -- the masker lets through only %.0f%% "
                     "of the win)."
                     % (r.masked, r.masker, r.gain_alone,
                        r.gain_with_masker_gold, r.suppression_ratio,
                        100.0 * r.suppression_ratio))
    lines.append("")
    lines.append("Fix the masker first: until it is fixed, the masked "
                 "component looks healthy even though it is weak.")
    lines.append("")
    return "\n".join(lines)
