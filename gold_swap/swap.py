"""The gold-swap experiment runner (black-box).

The runner is the only part of the tool that touches the system: it
runs a system on a set of cases with a given set of components
replaced by their gold versions, and records the mean score. The
attribution, fix-order and masking modules only ever see these
experimental results -- they never see component qualities or case
sensitivities. That separation is what makes the evaluation honest:
the tool's gains are measured by experiment, and the analytic ground
truth (computed from the known model) is an independent check.
"""

from .models import score


def run_system(system, cases, gold=frozenset()):
    """Mean end-to-end score over cases with ``gold`` components ideal.

    Returns a dict with the mean score, the per-case scores, and the
    number of cases, so callers can audit the mean.
    """
    gold = frozenset(gold)
    per_case = [score(system, case, gold=gold) for case in cases]
    mean = sum(per_case) / len(per_case) if per_case else 0.0
    return {"mean": mean, "per_case": per_case, "n": len(per_case)}


def gold_swap_gains(system, cases):
    """One gold-swap experiment per component, plus the baseline.

    Returns a dict with the baseline mean score, the all-gold mean
    score (always 1.0 by construction), and the per-component gain
    (gold-swap mean minus baseline mean). This is N+1 runs for an
    N-component system -- the cheapest complete attribution.
    """
    baseline = run_system(system, cases)
    gains = {}
    for name in system.names():
        swapped = run_system(system, cases, gold=frozenset([name]))
        gains[name] = swapped["mean"] - baseline["mean"]
    all_gold = run_system(system, cases, gold=frozenset(system.names()))
    return {
        "baseline": baseline["mean"],
        "all_gold": all_gold["mean"],
        "gains": gains,
        "n": len(cases),
    }


def analytic_gains(system, cases):
    """Ground-truth gains computed directly from the known model.

    Used only by the evaluation as an independent check on the
    experimental gains. In a real deployment there is no analytic
    ground truth -- the experiment IS the measurement. Here the model
    is fully known, so the two must agree exactly; any disagreement
    is a bug in the runner, not in the model.
    """
    from .models import reliability

    gains = {}
    for name in system.names():
        total_gain = 0.0
        for case in cases:
            base = score(system, case)
            # Gold-swapping a component sets its reliability to 1, so
            # the swapped score is the product of the OTHER components'
            # reliabilities -- computed directly, never by division
            # (a zero reliability would divide by zero).
            swapped = 1.0
            for other in system.components:
                if other.name != name:
                    swapped *= reliability(other, case)
            total_gain += swapped - base
        gains[name] = total_gain / len(cases) if cases else 0.0
    return gains
