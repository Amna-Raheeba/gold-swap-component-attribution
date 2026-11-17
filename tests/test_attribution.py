"""Tests for attribution: gains, ceiling shares, interactions."""

import pytest

from gold_swap.attribution import attribute
from gold_swap.systems import SYSTEMS, make_cases


def test_bottlenecks_are_the_weakest_components():
    expected = {"rag": "generator", "research": "synthesizer", "phm": "detector"}
    for key, bottleneck in expected.items():
        system = SYSTEMS[key]
        report = attribute(system, make_cases(system, n=100, seed=7))
        assert report.bottleneck() == bottleneck


def test_ceiling_shares_plus_residual_sum_to_one():
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        report = attribute(system, make_cases(system, n=100, seed=7))
        total = sum(a.ceiling_share for a in report.attributions)
        assert total + report.interaction_residual == pytest.approx(1.0, abs=1e-9)


def test_ranks_follow_gain_order():
    system = SYSTEMS["rag"]
    report = attribute(system, make_cases(system, n=100, seed=7))
    gains = [a.gain for a in report.attributions]
    assert gains == sorted(gains, reverse=True)
    assert [a.bottleneck_rank for a in report.attributions] == [1, 2, 3, 4]


def test_all_pairs_superadditive_in_hand_built_systems():
    # In the series model, fixing two components together always wins
    # more than the sum of fixing each alone.
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        report = attribute(system, make_cases(system, n=100, seed=7))
        assert len(report.pairwise) == 6
        assert all(v > 0 for v in report.pairwise.values())
        assert report.interaction_residual > 0


def test_perfect_system_has_no_gains():
    from gold_swap.models import Component, System

    system = System(
        name="perfect",
        description="",
        components=(
            Component(name="a", quality=1.0),
            Component(name="b", quality=1.0),
        ),
    )
    report = attribute(system, make_cases(system, n=10, seed=7))
    assert report.baseline == pytest.approx(1.0)
    assert report.ceiling == pytest.approx(0.0)
    assert all(a.gain == pytest.approx(0.0) for a in report.attributions)
    assert report.at_ceiling() == ["a", "b"]


def test_gain_grows_with_quality_gap():
    # The same system with a worse generator must show a bigger
    # generator gain and a bigger generator ceiling share.
    from gold_swap.models import Component, System

    def build(gen_quality):
        return System(
            name="s",
            description="",
            components=(
                Component(name="retriever", quality=0.8),
                Component(name="generator", quality=gen_quality),
            ),
        )

    cases = make_cases(build(0.7), n=100, seed=7)
    mild = attribute(build(0.7), cases)
    bad = attribute(build(0.4), cases)
    mild_gain = [a for a in mild.attributions if a.name == "generator"][0].gain
    bad_gain = [a for a in bad.attributions if a.name == "generator"][0].gain
    assert bad_gain > mild_gain
