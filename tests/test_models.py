"""Tests for the core model: components, systems, scoring."""

import pytest

from gold_swap.models import Case, Component, System, reliability, score


def _system():
    return System(
        name="tiny",
        description="two components",
        components=(
            Component(name="a", quality=0.8, description="first"),
            Component(name="b", quality=0.6, description="second"),
        ),
    )


def test_component_quality_bounds():
    with pytest.raises(ValueError):
        Component(name="x", quality=1.5)
    with pytest.raises(ValueError):
        Component(name="x", quality=-0.1)
    with pytest.raises(ValueError):
        Component(name="x", quality=0.5, cost=0.0)


def test_system_requires_components():
    with pytest.raises(ValueError):
        System(name="empty", description="", components=())


def test_system_rejects_duplicate_names():
    with pytest.raises(ValueError):
        System(
            name="dup",
            description="",
            components=(
                Component(name="a", quality=0.5),
                Component(name="a", quality=0.7),
            ),
        )


def test_system_component_lookup():
    system = _system()
    assert system.component("a").quality == 0.8
    with pytest.raises(KeyError):
        system.component("nope")


def test_score_is_product_of_reliabilities():
    system = _system()
    case = Case(case_id="c1", sensitivities={"a": 1.0, "b": 1.0})
    # r_a = 1 - 0.2 = 0.8; r_b = 1 - 0.4 = 0.6; score = 0.48
    assert score(system, case) == pytest.approx(0.48)


def test_sensitivity_zero_means_component_costs_nothing():
    system = _system()
    case = Case(case_id="c1", sensitivities={"a": 0.0, "b": 1.0})
    assert score(system, case) == pytest.approx(0.6)


def test_gold_component_scores_one():
    system = _system()
    case = Case(case_id="c1", sensitivities={"a": 1.0, "b": 1.0})
    assert score(system, case, gold=frozenset(["a"])) == pytest.approx(0.6)
    assert score(system, case, gold=frozenset(["a", "b"])) == pytest.approx(1.0)


def test_reliability_at_gold_quality():
    comp = Component(name="a", quality=1.0)
    case = Case(case_id="c1", sensitivities={"a": 1.0})
    assert reliability(comp, case) == pytest.approx(1.0)


def test_case_default_sensitivity_is_zero():
    case = Case(case_id="c1")
    assert case.sensitivity("anything") == 0.0
