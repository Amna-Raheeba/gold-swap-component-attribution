"""The three hand-built systems and the seeded case generator.

Systems are deliberately varied so the attribution has something real
to find: one RAG pipeline whose generator is the bottleneck, one
agentic research pipeline whose synthesiser is the bottleneck, and one
PHM monitoring pipeline whose detector is the bottleneck -- plus a
fourth, deliberately broken variant used only by the masking tests, in
which a near-dead detector masks everything downstream of it.
"""

import json
import os
import random

from .models import Case, Component, System

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _system(name, description, specs):
    comps = tuple(
        Component(name=n, quality=q, description=d, cost=c)
        for n, q, d, c in specs
    )
    return System(name=name, description=description, components=comps)


RAG = _system(
    "rag",
    "RAG question-answering pipeline: retrieve, rerank, generate, verify.",
    [
        ("retriever", 0.72, "BM25 + vector hybrid retrieval over the corpus.", 3.0),
        ("reranker", 0.85, "Cross-encoder reranking of the top candidates.", 1.0),
        ("generator", 0.68, "Answer generation from the retrieved passages.", 5.0),
        ("verifier", 0.80, "Citation and consistency check on the answer.", 2.0),
    ],
)

RESEARCH = _system(
    "research",
    "Agentic research pipeline: plan, call tools, synthesise, criticise.",
    [
        ("planner", 0.75, "Decomposes the question into sub-tasks.", 2.0),
        ("tool_caller", 0.82, "Executes searches and reads sources.", 3.0),
        ("synthesizer", 0.70, "Drafts the report from tool outputs.", 4.0),
        ("critic", 0.88, "Checks the draft for unsupported claims.", 1.0),
    ],
)

PHM = _system(
    "phm",
    "Predictive-maintenance monitoring pipeline: sense, detect, diagnose, report.",
    [
        ("sensor", 0.90, "Telemetry ingestion and cleaning.", 2.0),
        ("detector", 0.65, "Anomaly detection on the sensor stream.", 4.0),
        ("diagnoser", 0.78, "Root-cause diagnosis of detected anomalies.", 3.0),
        ("reporter", 0.84, "Maintenance report and alert ranking.", 1.0),
    ],
)

# Deliberately broken variant: the detector is nearly dead, so it
# masks the diagnoser and reporter -- gold-swapping either of them
# alone gains almost nothing until the detector is fixed.
PHM_MASKED = _system(
    "phm_masked",
    "PHM pipeline with a near-dead detector (masking demo only).",
    [
        ("sensor", 0.90, "Telemetry ingestion and cleaning.", 2.0),
        ("detector", 0.05, "Anomaly detection, nearly dead.", 4.0),
        ("diagnoser", 0.78, "Root-cause diagnosis of detected anomalies.", 3.0),
        ("reporter", 0.84, "Maintenance report and alert ranking.", 1.0),
    ],
)

SYSTEMS = {
    "rag": RAG,
    "research": RESEARCH,
    "phm": PHM,
    "phm_masked": PHM_MASKED,
}


def make_cases(system, n=200, seed=7, sensitivity_low=0.3, sensitivity_high=1.0):
    """Generate ``n`` deterministic cases for a system.

    Sensitivities are drawn seeded-uniform in
    [sensitivity_low, sensitivity_high]: every case exercises every
    component at least somewhat, but how much varies, which is what
    makes per-case gains differ and the mean gain a real measurement
    rather than a single arithmetic fact.
    """
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        sens = {
            name: rng.uniform(sensitivity_low, sensitivity_high)
            for name in system.names()
        }
        cases.append(Case(case_id="%s-%03d" % (system.name, i), sensitivities=sens))
    return cases


def systems_to_json():
    """Serialise the system definitions (for data/systems.json)."""
    out = {}
    for key, system in SYSTEMS.items():
        out[key] = {
            "name": system.name,
            "description": system.description,
            "components": [
                {
                    "name": c.name,
                    "quality": c.quality,
                    "description": c.description,
                    "cost": c.cost,
                }
                for c in system.components
            ],
        }
    return out


def load_systems(path=None):
    """Load system definitions from data/systems.json."""
    path = path or os.path.join(_DATA_DIR, "systems.json")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    loaded = {}
    for key, spec in raw.items():
        loaded[key] = _system(spec["name"], spec["description"],
                              [(c["name"], c["quality"], c["description"],
                                c.get("cost", 1.0))
                               for c in spec["components"]])
    return loaded
