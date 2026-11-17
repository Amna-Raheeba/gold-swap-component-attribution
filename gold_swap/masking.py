"""Masked-component detection.

A component is **masked** when gold-swapping it alone gains little
relative to what it gains once another component is fixed. The masker
is usually a near-dead component: while it fails, nothing else can
matter, so every other component's gold-swap gain is suppressed --
even though those components are themselves weak. Fix the masker
first and the masked components' real weaknesses appear.

Detection is a second-order experiment: for each ordered pair (X, Y),
compare X's gain alone against X's gain with Y already gold. X is
masked by Y when the ratio ``gain_X_alone / gain_X_given_Y_gold``
is below one half (Y suppresses more than half of X's win) and the
unmasked gain is at least the threshold (so the weakness is real,
not noise).

In the series reliability model the ratio has an exact meaning: it
is (approximately) the masker's mean reliability. A masker at
reliability 0.38 suppresses every other component's gain to 38% of
its unmasked value -- which is why the detector in the broken PHM
variant masks all three of its siblings.
"""

from dataclasses import dataclass

from .swap import run_system

RATIO_THRESHOLD = 0.5


@dataclass
class MaskingResult:
    masked: str
    masker: str
    gain_alone: float
    gain_with_masker_gold: float
    suppression_ratio: float


def find_masked(system, cases, gain_threshold=0.02, ratio_threshold=RATIO_THRESHOLD):
    """Find (masked, masker) pairs.

    ``gain_threshold`` is the unmasked gain that counts as "a real
    weakness"; ``ratio_threshold`` is the suppression ratio below
    which X counts as masked by Y.
    """
    baseline = run_system(system, cases)["mean"]
    names = system.names()
    results = []
    for masked in names:
        alone = (
            run_system(system, cases, gold=frozenset([masked]))["mean"] - baseline
        )
        for masker in names:
            if masker == masked:
                continue
            with_masker = (
                run_system(system, cases, gold=frozenset([masker]))["mean"]
            )
            gain = (
                run_system(
                    system, cases, gold=frozenset([masked, masker])
                )["mean"]
                - with_masker
            )
            if gain < gain_threshold:
                continue
            ratio = alone / gain if gain > 0 else 1.0
            if ratio < ratio_threshold:
                results.append(
                    MaskingResult(
                        masked=masked,
                        masker=masker,
                        gain_alone=alone,
                        gain_with_masker_gold=gain,
                        suppression_ratio=ratio,
                    )
                )
    return results
