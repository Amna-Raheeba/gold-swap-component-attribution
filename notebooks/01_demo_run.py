"""Notebook 01: demo run -- attribute the RAG pipeline.

Run with: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_swap.attribution import attribute
from gold_swap.fixorder import budget_fix, compare_fix_orders
from gold_swap.masking import find_masked
from gold_swap.report import (
    render_attribution,
    render_budget,
    render_fix_order,
    render_masking,
)
from gold_swap.systems import SYSTEMS, make_cases


def main():
    system = SYSTEMS["rag"]
    cases = make_cases(system, n=200, seed=7)
    print(render_attribution(attribute(system, cases)))
    print(render_fix_order(compare_fix_orders(system, cases)))
    print(render_budget(budget_fix(system, cases, budget=6.0)))
    print("Masking on the healthy RAG system:")
    print(render_masking(find_masked(system, cases)))
    masked_system = SYSTEMS["phm_masked"]
    masked_cases = make_cases(masked_system, n=200, seed=7)
    print("Masking on the deliberately broken PHM system:")
    print(render_masking(find_masked(masked_system, masked_cases)))


if __name__ == "__main__":
    main()
