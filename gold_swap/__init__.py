"""Gold-swap component attribution.

Attribute a multi-component system's end-to-end performance to its
individual components by gold-swapping each one: replace component X
with its gold (ideal) version while holding everything else fixed and
measure the performance gain. The component whose gold-swap gains the
most is the weakest link; the component whose gold-swap gains nothing
is already at its ceiling.

This is ablation's forward twin: instead of removing a component to
see what breaks, you perfect it to see what it could win.
"""

from .models import Case, Component, System
from .swap import gold_swap_gains, run_system
from .attribution import AttributionReport, attribute
from .fixorder import FixOrderResult, greedy_fix_order, optimal_fix_set
from .masking import MaskingResult, find_masked

__all__ = [
    "Case",
    "Component",
    "System",
    "gold_swap_gains",
    "run_system",
    "AttributionReport",
    "attribute",
    "FixOrderResult",
    "greedy_fix_order",
    "optimal_fix_set",
    "MaskingResult",
    "find_masked",
]

__version__ = "0.1.0"
