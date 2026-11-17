"""Core data model: components, systems, cases.

The scoring model is the series-system reliability model from
reliability engineering. A system is a serial pipeline of components.
For one case, component ``i`` with quality ``q_i`` and case
sensitivity ``s_i`` (both in [0, 1]) has reliability::

    r_i = 1 - (1 - q_i) * s_i

and the end-to-end score is the product of the component
reliabilities::

    score = prod_i r_i

A gold component is ideal: quality 1.0, so ``r_i = 1`` and it costs
the system nothing. The model is deterministic and fully specified by
the system definition plus the per-case sensitivities, which makes
every gold-swap gain exactly reproducible -- and exactly checkable
against an analytic ground truth in the evaluation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Component:
    """One component of a multi-component system.

    ``cost`` is the engineering effort to bring the component to
    gold, in arbitrary budget units (default 1.0). It only matters
    for the budget-constrained fix order: without a budget, greedy
    is provably optimal (see fixorder.py), so the budget is what
    makes the fix-order comparison a measurement instead of a
    theorem.
    """

    name: str
    quality: float
    description: str = ""
    cost: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(
                "quality must be in [0, 1], got %.3f" % self.quality
            )
        if self.cost <= 0:
            raise ValueError("cost must be positive, got %.3f" % self.cost)


@dataclass(frozen=True)
class System:
    """A serial multi-component system."""

    name: str
    description: str
    components: tuple

    def __post_init__(self):
        if not self.components:
            raise ValueError("a system needs at least one component")
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError("duplicate component name in system %r" % self.name)

    def component(self, name):
        """Return the component with this name, or raise KeyError."""
        for comp in self.components:
            if comp.name == name:
                return comp
        raise KeyError("no component named %r in system %r" % (name, self.name))

    def names(self):
        return [c.name for c in self.components]


@dataclass(frozen=True)
class Case:
    """One evaluation case: per-component sensitivities in [0, 1].

    A sensitivity of 0 means the case does not exercise the component
    at all (its loss is zero whatever its quality); a sensitivity of 1
    means the component's full quality gap is exposed.
    """

    case_id: str
    sensitivities: dict = field(default_factory=dict)

    def sensitivity(self, name):
        return self.sensitivities.get(name, 0.0)


def reliability(component, case):
    """Per-case reliability of one component."""
    loss = (1.0 - component.quality) * case.sensitivity(component.name)
    return 1.0 - loss


def score(system, case, gold=frozenset()):
    """End-to-end score of a system on one case.

    ``gold`` is the set of component names replaced by their ideal
    versions (reliability 1.0). With no gold components this is the
    baseline score; with all components gold the score is 1.0.
    """
    total = 1.0
    for comp in system.components:
        if comp.name in gold:
            continue
        total *= reliability(comp, case)
    return total
