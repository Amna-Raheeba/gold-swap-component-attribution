"""Fix order: which component to improve first.

Two orders are computed:

* **Greedy** -- repeatedly gold-swap the component with the largest
  *marginal* gain given the components already fixed, until the target
  score is reached or everything is gold. This is the order a team
  would actually follow: always fix the currently-biggest win.
* **Optimal** -- exhaustive search over subsets for the *minimal fix
  set* reaching the target, and over permutations for the best full
  ordering. Feasible because systems here have at most 5 components
  (2^5 subsets, 5! orders). The comparison between greedy and optimal
  is the honest measurement in this module: greedy is not always
  optimal, and the tool reports exactly how often it is and what it
  costs when it is not.
"""

import itertools
from dataclasses import dataclass

from .swap import run_system


@dataclass
class FixOrderResult:
    system_name: str
    target: float
    greedy_order: list
    greedy_scores: list
    greedy_steps_to_target: int
    optimal_set: list
    optimal_size: int
    greedy_is_optimal: bool
    greedy_full_order: list
    full_order_agreement: bool
    cumulative_order_gap: float
    score_gap_after_two_fixes: float


def greedy_fix_order(system, cases, target=0.95, stop_at_target=True):
    """Greedy fix order: always fix the currently-biggest marginal gain.

    Returns (order, cumulative_scores): the order is the list of
    component names in the order they were fixed; the scores are the
    mean system score after each fix, starting with the baseline.
    With ``stop_at_target`` the order stops once the target is
    reached; otherwise it fixes every component (the full order used
    for the cumulative-gap comparison).
    """
    fixed = set()
    order = []
    scores = [run_system(system, cases)["mean"]]
    remaining = set(system.names())
    while remaining and (not stop_at_target or scores[-1] < target):
        best_name = None
        best_score = -1.0
        for name in sorted(remaining):
            candidate = fixed | {name}
            mean = run_system(system, cases, gold=frozenset(candidate))["mean"]
            if mean > best_score:
                best_score = mean
                best_name = name
        fixed.add(best_name)
        remaining.remove(best_name)
        order.append(best_name)
        scores.append(best_score)
    return order, scores


def optimal_fix_set(system, cases, target=0.95):
    """Smallest subset of components whose gold-swap reaches the target.

    Exhaustive over subsets, smallest first; ties broken by the
    highest score, then by name order so the result is deterministic.
    Returns (subset, score). An empty list means the baseline already
    reaches the target; None means even all-gold cannot reach it
    (impossible here, since all-gold scores 1.0).
    """
    names = system.names()
    baseline = run_system(system, cases)["mean"]
    if baseline >= target:
        return [], baseline
    for size in range(1, len(names) + 1):
        best = None
        best_score = -1.0
        for subset in itertools.combinations(sorted(names), size):
            mean = run_system(system, cases, gold=frozenset(subset))["mean"]
            if mean > best_score:
                best_score = mean
                best = list(subset)
        if best_score >= target:
            return best, best_score
    return None, run_system(system, cases, gold=frozenset(names))["mean"]


def optimal_full_order(system, cases):
    """Best full ordering by cumulative score after each fix.

    Exhaustive over permutations: the order maximising the sum of the
    mean scores after each prefix fix. For a 4-component system this
    is 24 permutations -- cheap, and the exact answer greedy is
    measured against.
    """
    names = system.names()
    best_order = None
    best_total = -1.0
    for perm in itertools.permutations(sorted(names)):
        fixed = set()
        total = 0.0
        for name in perm:
            fixed.add(name)
            total += run_system(system, cases, gold=frozenset(fixed))["mean"]
        if total > best_total:
            best_total = total
            best_order = list(perm)
    return best_order


def compare_fix_orders(system, cases, target=0.95):
    """Greedy vs optimal, in one result object.

    Note on the minimal-fix-set comparison: in the series reliability
    model greedy is PROVABLY optimal for it, not merely measured so.
    The score after fixing a set S depends only on the product of the
    unfixed reliabilities, so the smallest set reaching the target is
    the set of components with the smallest reliabilities -- exactly
    what greedy picks, in order. The comparison is kept as a check
    that the implementation honours the theorem (it must always be
    True); the real measurement is the full-order comparison, where
    per-case sensitivities vary and greedy can genuinely lose.
    """
    order, scores = greedy_fix_order(system, cases, target=target)
    steps = len(order)
    opt_set, _ = optimal_fix_set(system, cases, target=target)
    greedy_set = order[: len(opt_set)] if opt_set is not None else order
    is_optimal = (
        opt_set is not None
        and set(greedy_set) == set(opt_set)
        and len(greedy_set) == len(opt_set)
    )
    # Cumulative-order gap: the best full ordering maximises the sum
    # of the scores after each prefix fix; greedy maximises each step
    # myopically, which is not the same thing. Both orders are FULL
    # orders (every component fixed) so the totals are comparable --
    # comparing a target-stopped greedy order against a full optimal
    # order would be a length artifact, not a measurement.
    opt_order = optimal_full_order(system, cases)
    _full_order, full_scores = greedy_fix_order(
        system, cases, target=target, stop_at_target=False
    )
    greedy_total = sum(full_scores[1:])
    fixed = set()
    opt_total = 0.0
    for name in opt_order:
        fixed.add(name)
        opt_total += run_system(system, cases, gold=frozenset(fixed))["mean"]
    gap_cumulative = opt_total - greedy_total
    full_order_agreement = _full_order == opt_order
    gap_two = _prefix_gap(system, cases, order, 2)
    return FixOrderResult(
        system_name=system.name,
        target=target,
        greedy_order=order,
        greedy_scores=scores,
        greedy_steps_to_target=steps,
        optimal_set=opt_set if opt_set is not None else [],
        optimal_size=len(opt_set) if opt_set is not None else -1,
        greedy_is_optimal=is_optimal,
        greedy_full_order=_full_order,
        full_order_agreement=full_order_agreement,
        cumulative_order_gap=gap_cumulative,
        score_gap_after_two_fixes=gap_two,
    )


def _prefix_gap(system, cases, greedy_order, k):
    """Best k-fix prefix score minus greedy's k-fix prefix score."""
    names = system.names()
    if k > len(names):
        return 0.0
    best = -1.0
    for subset in itertools.combinations(sorted(names), k):
        mean = run_system(system, cases, gold=frozenset(subset))["mean"]
        if mean > best:
            best = mean
    greedy_prefix = run_system(
        system, cases, gold=frozenset(greedy_order[:k])
    )["mean"]
    return best - greedy_prefix


@dataclass
class BudgetFixResult:
    system_name: str
    budget: float
    baseline: float
    optimal_subset: list
    optimal_score: float
    optimal_cost: float
    greedy_gain_subset: list
    greedy_gain_score: float
    greedy_gain_cost: float
    greedy_ratio_subset: list
    greedy_ratio_score: float
    greedy_ratio_cost: float
    greedy_gain_is_optimal: bool
    greedy_ratio_is_optimal: bool


def _subset_cost(system, subset):
    return sum(system.component(name).cost for name in subset)


def _greedy_fill(system, cases, budget, key):
    """Greedily take components in ``key`` order while the budget allows.

    ``key`` maps a component name to its priority (higher first).
    Returns (subset, score, cost).
    """
    ordered = sorted(system.names(), key=lambda n: (-key(n), n))
    subset = []
    spent = 0.0
    for name in ordered:
        cost = system.component(name).cost
        if spent + cost <= budget:
            subset.append(name)
            spent += cost
    mean = run_system(system, cases, gold=frozenset(subset))["mean"]
    return subset, mean, spent


def budget_fix(system, cases, budget=6.0):
    """Fix order under a budget: which fixes buy the most score.

    Three strategies, all measured by black-box experiment:

    * **optimal** -- exhaustive search over subsets with total cost
      at most the budget, maximising the mean score (ties: lower
      cost, then name order, so the result is deterministic).
    * **greedy by gain** -- take components in gold-swap-gain order
      while the budget allows. Ignores cost entirely.
    * **greedy by gain per cost** -- take components in
      gain-per-cost order while the budget allows. The textbook
      knapsack heuristic -- which is famously NOT always optimal.

    The comparison is the honest fix-order measurement in this tool:
    without a budget greedy is provably optimal (see
    ``compare_fix_orders``), but under a budget both greedy
    strategies can lose to the exhaustive optimum, and the tool
    reports exactly how often and by how much.
    """
    from .swap import gold_swap_gains

    baseline = run_system(system, cases)["mean"]
    gains = gold_swap_gains(system, cases)["gains"]
    names = sorted(system.names())
    best_subset = []
    best_score = baseline
    best_cost = 0.0
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            cost = _subset_cost(system, subset)
            if cost > budget:
                continue
            mean = run_system(system, cases, gold=frozenset(subset))["mean"]
            if mean > best_score + 1e-12 or (
                abs(mean - best_score) <= 1e-12 and cost < best_cost
            ):
                best_subset = list(subset)
                best_score = mean
                best_cost = cost
    gain_subset, gain_score, gain_cost = _greedy_fill(
        system, cases, budget, key=lambda n: gains[n]
    )
    ratio_subset, ratio_score, ratio_cost = _greedy_fill(
        system, cases, budget,
        key=lambda n: gains[n] / system.component(n).cost,
    )
    return BudgetFixResult(
        system_name=system.name,
        budget=budget,
        baseline=baseline,
        optimal_subset=best_subset,
        optimal_score=best_score,
        optimal_cost=best_cost,
        greedy_gain_subset=gain_subset,
        greedy_gain_score=gain_score,
        greedy_gain_cost=gain_cost,
        greedy_ratio_subset=ratio_subset,
        greedy_ratio_score=ratio_score,
        greedy_ratio_cost=ratio_cost,
        greedy_gain_is_optimal=set(gain_subset) == set(best_subset),
        greedy_ratio_is_optimal=set(ratio_subset) == set(best_subset),
    )
