"""Tests for the gold-swap experiment runner."""

import pytest

from gold_swap.models import score
from gold_swap.swap import analytic_gains, gold_swap_gains, run_system
from gold_swap.systems import SYSTEMS, make_cases


def test_run_system_mean_matches_manual():
    system = SYSTEMS["rag"]
    cases = make_cases(system, n=10, seed=7)
    result = run_system(system, cases)
    manual = sum(score(system, c) for c in cases) / len(cases)
    assert result["mean"] == pytest.approx(manual)
    assert result["n"] == 10
    assert len(result["per_case"]) == 10


def test_run_system_empty_cases():
    system = SYSTEMS["rag"]
    assert run_system(system, [])["mean"] == 0.0


def test_all_gold_scores_one():
    system = SYSTEMS["phm"]
    cases = make_cases(system, n=20, seed=7)
    result = run_system(system, cases, gold=frozenset(system.names()))
    assert result["mean"] == pytest.approx(1.0)


def test_gold_swap_gains_match_analytic():
    # The experimental gains (black-box runs) must agree exactly with
    # the analytic ground truth computed from the known model.
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        cases = make_cases(system, n=50, seed=7)
        experimental = gold_swap_gains(system, cases)
        analytic = analytic_gains(system, cases)
        for name in system.names():
            assert experimental["gains"][name] == pytest.approx(
                analytic[name], abs=1e-12
            )
        assert experimental["all_gold"] == pytest.approx(1.0)


def test_gains_are_nonnegative_and_bounded():
    system = SYSTEMS["research"]
    cases = make_cases(system, n=50, seed=7)
    result = gold_swap_gains(system, cases)
    for gain in result["gains"].values():
        assert 0.0 <= gain <= 1.0 - result["baseline"] + 1e-12


def test_cases_are_deterministic():
    system = SYSTEMS["rag"]
    first = make_cases(system, n=30, seed=7)
    second = make_cases(system, n=30, seed=7)
    assert first == second
    third = make_cases(system, n=30, seed=8)
    assert first != third


def test_analytic_gains_handle_zero_reliability():
    # A component with quality 0 and sensitivity 1 has reliability 0;
    # the analytic gain must still compute (no division by zero).
    from gold_swap.models import Component, System
    from gold_swap.systems import make_cases as mk

    system = System(
        name="dead",
        description="",
        components=(
            Component(name="a", quality=0.0),
            Component(name="b", quality=0.9),
        ),
    )
    cases = mk(system, n=10, seed=7)
    gains = analytic_gains(system, cases)
    assert gains["a"] > 0.0
    experimental = gold_swap_gains(system, cases)
    assert experimental["gains"]["a"] == pytest.approx(gains["a"], abs=1e-12)


def test_systems_json_round_trip():
    # data/systems.json must load back to the same definitions.
    from gold_swap.systems import SYSTEMS, load_systems

    loaded = load_systems()
    assert set(loaded) == set(SYSTEMS)
    for key, system in SYSTEMS.items():
        assert loaded[key].names() == system.names()
        for comp in system.components:
            assert loaded[key].component(comp.name).quality == comp.quality
            assert loaded[key].component(comp.name).cost == comp.cost
