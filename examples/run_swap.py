"""Example: run a gold-swap attribution on the demo RAG system.

Run with: python examples/run_swap.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_swap.attribution import attribute
from gold_swap.models import Component, System
from gold_swap.report import render_attribution
from gold_swap.systems import make_cases


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "demo_system.json"), "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    system = System(
        name=spec["name"],
        description=spec["description"],
        components=tuple(
            Component(
                name=c["name"],
                quality=c["quality"],
                description=c["description"],
                cost=c["cost"],
            )
            for c in spec["components"]
        ),
    )
    cases = make_cases(system, n=50, seed=7)
    print(render_attribution(attribute(system, cases)))


if __name__ == "__main__":
    main()
