"""Tests for the evaluation and its headline numbers."""

from gold_swap.evaluate import evaluate, evaluate_masking, evaluate_random_systems


def test_evaluate_headline_values():
    results = evaluate()
    head = results["headline"]
    assert head["max_gain_error"] < 1e-9
    assert head["max_accounting_error"] < 1e-9
    assert head["bottlenecks"] == {
        "rag": "generator",
        "research": "synthesizer",
        "phm": "detector",
    }
    assert head["masking_precision"] == 1.0
    assert head["masking_recall"] == 1.0
    assert head["masking_control_false_positives"] == 0
    # Greedy is provably optimal without a budget (theorem).
    assert head["greedy_optimal_rate"] == 1.0
    assert head["random_greedy_optimal_rate"] == 1.0
    assert head["random_full_order_agreement_rate"] == 1.0


def test_budget_measurement_is_non_vacuous():
    # The budget comparison must actually separate the strategies:
    # greedy by gain loses on some random systems, and greedy by
    # gain per cost loses on more -- otherwise it would be a theorem
    # in disguise, not a measurement.
    random_systems = evaluate_random_systems()
    assert random_systems["budget_greedy_gain_optimal_rate"] < 1.0
    assert random_systems["budget_greedy_ratio_optimal_rate"] < 1.0
    assert (
        random_systems["budget_greedy_ratio_optimal_rate"]
        < random_systems["budget_greedy_gain_optimal_rate"]
    )
    assert random_systems["budget_max_gain_gap"] > 0.0
    assert random_systems["budget_max_ratio_gap"] > 0.0


def test_evaluate_masking_ground_truth():
    masking = evaluate_masking()
    assert masking["precision"] == 1.0
    assert masking["recall"] == 1.0
    assert masking["n_found"] == 3


def test_evaluate_is_deterministic():
    first = evaluate()
    second = evaluate()
    assert first == second
