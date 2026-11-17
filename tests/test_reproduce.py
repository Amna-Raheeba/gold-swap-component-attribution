"""Tests for the reproduce-results script and its drift guard."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reproduce_results_exits_zero():
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "reproduce_results.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headline numbers stable" in result.stdout


def test_results_files_exist():
    for name in ("evaluation.json", "evaluation.csv", "evaluation_summary.md", "HEADLINE.md"):
        assert os.path.exists(os.path.join(ROOT, "results", name))
