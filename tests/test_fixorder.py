"""Tests for fix order: greedy vs optimal, and the budget comparison."""

import pytest

from gold_swap.fixorder import (
    budget_fix,
    compare_fix_orders,
    greedy_fix_order,
    optimal_fix_set,
    optimal_full_order,
)
from gold_swap.systems import SYSTEMS, make_cases


def test_greedy_is_optimal_for_minimal_fix_set():
    # Theorem, not a measurement: in the series model the score after
    # fixing a set depends only on the product of the unfixed
    # reliabilities, so the minimal set is the smallest reliabilities
    # -- exactly what greedy picks. The implementation must honour it.
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        fix = compare_fix_orders(system, make_cases(system, n=100, seed=7))
        assert fix.greedy_is_optimal


def test_greedy_full_order_matches_optimal_on_hand_built():
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        cases = make_cases(system, n=100, seed=7)
        fix = compare_fix_orders(system, cases)
        assert fix.full_order_agreement
        assert abs(fix.cumulative_order_gap) < 1e-9


def test_optimal_fix_set_is_minimal():
    system = SYSTEMS["rag"]
    cases = make_cases(system, n=100, seed=7)
    subset, score = optimal_fix_set(system, cases, target=0.80)
    assert score >= 0.80
    # Removing any one component from the set must drop below target.
    from gold_swap.swap import run_system

    for name in subset:
        rest = [n for n in subset if n != name]
        assert run_system(system, cases, gold=frozenset(rest))["mean"] < 0.80


def test_greedy_order_scores_monotone():
    system = SYSTEMS["research"]
    cases = make_cases(system, n=50, seed=7)
    _order, scores = greedy_fix_order(system, cases, stop_at_target=False)
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)


def test_optimal_full_order_maximises_prefix_sum():
    system = SYSTEMS["phm"]
    cases = make_cases(system, n=50, seed=7)
    order = optimal_full_order(system, cases)
    assert sorted(order) == sorted(system.names())
    from gold_swap.swap import run_system

    fixed = set()
    total = 0.0
    for name in order:
        fixed.add(name)
        total += run_system(system, cases, gold=frozenset(fixed))["mean"]
    # Any other order must score no better.
    import itertools

    for perm in itertools.permutations(sorted(system.names())):
        fixed = set()
        other = 0.0
        for name in perm:
            fixed.add(name)
            other += run_system(system, cases, gold=frozenset(fixed))["mean"]
        assert total >= other - 1e-12


def test_budget_greedy_by_gain_loses_on_rag():
    # The showcase: the generator has the biggest gain but costs 5 of
    # a budget of 6, so greedy-by-gain wastes the budget on it and
    # scores below the exhaustive optimum.
    system = SYSTEMS["rag"]
    cases = make_cases(system, n=200, seed=7)
    result = budget_fix(system, cases, budget=6.0)
    assert result.optimal_score > result.greedy_gain_score + 1e-9
    assert not result.greedy_gain_is_optimal
    assert result.greedy_ratio_is_optimal
    assert result.optimal_cost <= 6.0


def test_budget_optimal_never_exceeds_budget():
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        cases = make_cases(system, n=50, seed=7)
        result = budget_fix(system, cases, budget=6.0)
        assert result.optimal_cost <= 6.0
        assert result.greedy_gain_cost <= 6.0
        assert result.greedy_ratio_cost <= 6.0
        assert result.optimal_score >= result.greedy_gain_score - 1e-12
        assert result.optimal_score >= result.greedy_ratio_score - 1e-12


def test_budget_zero_buys_nothing():
    system = SYSTEMS["rag"]
    cases = make_cases(system, n=50, seed=7)
    result = budget_fix(system, cases, budget=0.5)
    assert result.optimal_subset == []
    assert result.optimal_score == pytest.approx(result.baseline)
