"""Component attribution: gains, ceiling shares, interactions.

The attribution report answers three questions:

1. **How much does each component cost?** The gold-swap gain:
   ``score(system with X gold) - score(baseline)``.
2. **What share of the remaining headroom is it?** The ceiling share:
   ``gain_X / (1 - baseline)``. The ceiling is the total headroom
   left (all-gold scores 1.0). Shares do NOT sum to 1 in general:
   gold-swaps interact, so the report also carries the interaction
   residual ``1 - sum(shares)``. A **positive** residual means the
   gains are superadditive -- fixing components together wins more
   than the sum of fixing each alone, so the individual gains leave
   headroom unaccounted for. A **negative** residual means the gains
   are subadditive -- they overlap, and fixing two components
   double-counts part of the win.
3. **Which components interact?** The pairwise interaction
   ``gain_XY - gain_X - gain_Y``: positive is superadditive, negative
   is subadditive, zero is independent.
"""

from dataclasses import dataclass, field

from .swap import gold_swap_gains, run_system


@dataclass
class ComponentAttribution:
    name: str
    gain: float
    ceiling_share: float
    bottleneck_rank: int


@dataclass
class AttributionReport:
    system_name: str
    baseline: float
    all_gold: float
    ceiling: float
    attributions: list
    interaction_residual: float
    pairwise: dict = field(default_factory=dict)
    n_cases: int = 0

    def bottleneck(self):
        """The component with the largest gold-swap gain."""
        if not self.attributions:
            return None
        return self.attributions[0].name

    def at_ceiling(self):
        """Components whose gold-swap gains (almost) nothing."""
        return [a.name for a in self.attributions if a.gain < 1e-9]


def attribute(system, cases):
    """Build the full attribution report from gold-swap experiments."""
    result = gold_swap_gains(system, cases)
    baseline = result["baseline"]
    ceiling = result["all_gold"] - baseline
    attributions = []
    ranked = sorted(result["gains"].items(), key=lambda kv: -kv[1])
    for rank, (name, gain) in enumerate(ranked, start=1):
        share = gain / ceiling if ceiling > 0 else 0.0
        attributions.append(
            ComponentAttribution(
                name=name, gain=gain, ceiling_share=share, bottleneck_rank=rank
            )
        )
    residual = 1.0 - sum(a.ceiling_share for a in attributions)
    pairwise = _pairwise_interactions(system, cases, result["gains"], baseline)
    return AttributionReport(
        system_name=system.name,
        baseline=baseline,
        all_gold=result["all_gold"],
        ceiling=ceiling,
        attributions=attributions,
        interaction_residual=residual,
        pairwise=pairwise,
        n_cases=len(cases),
    )


def _pairwise_interactions(system, cases, gains, baseline):
    """Joint-gain minus sum-of-single-gains for every component pair."""
    pairwise = {}
    names = system.names()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = frozenset([names[i], names[j]])
            joint = run_system(system, cases, gold=pair)["mean"] - baseline
            value = joint - gains[names[i]] - gains[names[j]]
            pairwise["%s+%s" % (names[i], names[j])] = value
    return pairwise
