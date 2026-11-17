"""Tests for masked-component detection."""

from gold_swap.masking import find_masked
from gold_swap.systems import SYSTEMS, make_cases


def test_masked_system_finds_all_three_pairs():
    system = SYSTEMS["phm_masked"]
    cases = make_cases(system, n=200, seed=7)
    found = {(m.masked, m.masker) for m in find_masked(system, cases)}
    assert found == {
        ("sensor", "detector"),
        ("diagnoser", "detector"),
        ("reporter", "detector"),
    }


def test_masking_suppression_ratio_matches_detector_reliability():
    # In the series model the ratio is (approximately) the masker's
    # mean reliability: the detector at quality 0.05 lets through
    # only ~38% of any other component's win.
    system = SYSTEMS["phm_masked"]
    cases = make_cases(system, n=200, seed=7)
    for result in find_masked(system, cases):
        assert 0.30 < result.suppression_ratio < 0.45
        assert result.gain_alone < result.gain_with_masker_gold


def test_healthy_systems_have_no_masking():
    for key in ("rag", "research", "phm"):
        system = SYSTEMS[key]
        cases = make_cases(system, n=200, seed=7)
        assert find_masked(system, cases) == []


def test_masker_itself_is_never_masked():
    system = SYSTEMS["phm_masked"]
    cases = make_cases(system, n=200, seed=7)
    found = find_masked(system, cases)
    assert all(m.masked != "detector" for m in found)


def test_stricter_threshold_finds_nothing():
    # A ratio threshold of 0.1 demands near-total suppression; the
    # detector only suppresses to ~38%, so nothing qualifies.
    system = SYSTEMS["phm_masked"]
    cases = make_cases(system, n=200, seed=7)
    assert find_masked(system, cases, ratio_threshold=0.1) == []
