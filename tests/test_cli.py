"""Tests for the CLI."""

import os
import tempfile

import pytest

from gold_swap.cli import main


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Gold-Swap Attribution: rag" in out
    assert "Fix Order" in out


def test_attribute_runs(capsys):
    assert main(["attribute", "--system", "phm", "--cases", "20"]) == 0
    out = capsys.readouterr().out
    assert "detector" in out


def test_fixorder_runs(capsys):
    assert main(["fixorder", "--system", "rag", "--cases", "20"]) == 0
    out = capsys.readouterr().out
    assert "Budget Fix" in out
    assert "Masked Components" in out


def test_evaluate_writes_results():
    with tempfile.TemporaryDirectory() as tmp:
        assert main(["evaluate", "--cases", "20", "--out", tmp]) == 0
        assert os.path.exists(os.path.join(tmp, "evaluation.json"))
        assert os.path.exists(os.path.join(tmp, "evaluation.csv"))
        assert os.path.exists(os.path.join(tmp, "evaluation_summary.md"))


def test_unknown_system_exits():
    with pytest.raises(SystemExit):
        main(["attribute", "--system", "nope"])
