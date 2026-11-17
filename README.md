# Gold-Swap Component Attribution

Attribute a multi-component system's end-to-end performance to its
individual components by **gold-swapping** each one: replace component
X with its gold (ideal) version while holding everything else fixed,
and measure the performance gain. The component whose gold-swap gains
the most is the weakest link; the component whose gold-swap gains
nothing is already at its ceiling.

This is ablation's forward twin: instead of removing a component to
see what breaks, you perfect it to see what it could win.

## 1. What it does

Given a serial multi-component system (a RAG pipeline, an agentic
research pipeline, a PHM monitoring pipeline) and a set of cases, the
tool runs N+1 black-box experiments -- the baseline plus one
gold-swap per component -- and reports:

- **Gold-swap gain** per component: `score(X gold) - score(baseline)`.
- **Ceiling share**: the gain as a share of the remaining headroom
  (`1 - baseline`). Shares do not sum to 1: a positive interaction
  residual means the gains are superadditive.
- **Pairwise interactions**: joint gain minus the sum of single
  gains, for every component pair.
- **Bottleneck rank**: components ordered by gold-swap gain.
- **Fix order**: greedy (always fix the currently-biggest win) vs
  exhaustive optimal, plus a **budget-constrained** comparison
  (see section 6).
- **Masked components**: components whose weakness is hidden behind
  a near-dead component (see section 7).

## 2. Quick start

```bash
pip install -r requirements.txt
python -m gold_swap demo
python -m gold_swap attribute --system rag
python -m gold_swap fixorder --system phm
python -m gold_swap evaluate
streamlit run app.py
```

## 3. The model

Series-system reliability, from reliability engineering. For one case,
component `i` with quality `q_i` and case sensitivity `s_i` has
reliability `r_i = 1 - (1 - q_i) * s_i`, and the end-to-end score is
the product of the component reliabilities. A gold component is
ideal (`q = 1`, so `r = 1`). The model is deterministic and fully
specified by the system definition plus the per-case sensitivities,
which makes every gain exactly reproducible -- and exactly checkable
against an analytic ground truth.

## 4. Headline numbers (seed 7, 200 cases per system)

- Bottlenecks: RAG **generator**, research **synthesizer**, PHM **detector**.
- Max gold-swap gain-recovery error (experiment vs analytic): **0.000000**.
- Max ceiling-accounting error (shares + residual = 1): **0.000000**.
- All 18 component pairs superadditive (6/6 per system).
- Under a budget of 6 (50 random systems): greedy by gain optimal on
  **92%**, greedy by gain per cost on **78%**; mean score gaps 0.0050 /
  0.0120, max gaps 0.090 / 0.149. The textbook heuristic loses to plain
  greedy-by-gain here.
- Masking detection: precision **1.00**, recall **1.00**, 0 false
  positives on the 3 healthy systems.
- 50 tests, CI green on Python 3.10 / 3.11 / 3.12.

## 5. Prior art

- **"Rethinking LLM-based RAG from a decoupled perspective"** (ICLR
  2026, withdrawn): oracle retrieval + oracle utilisation analysis
  for RAG -- the closest concept, but a paper analysing two stages,
  not a general runnable tool, and it was withdrawn.
- **CanTest** (ACL 2026): counterfactual step-replacement with
  ground-truth oracles for vision-language navigation agents --
  failure localisation, not end-to-end performance attribution.
- **REFLECT** (ICML 2026 workshop) and **CHIEF**: intervention-based
  error attribution at step/agent level, not component-level
  gold-swap gain with a fix order.
- **Oracle Gap Ablation Harness** (blog post): oracle-retrieval upper
  bounds for agent harnesses -- a write-up, not a tool.

No offline, deterministic, general N-component gold-swap attribution
tool with ceiling-share accounting, a fix-order optimiser and masking
detection exists. This one differs from the author's own repos too:
plan-change-attribution finds what *caused* a plan change,
CascadeScope is forward blast radius, citation-ablation is
citation-level -- this is component-level, end-to-end performance
attribution via ideal replacement.

## 6. Honest limits

- **Greedy is provably optimal without a budget.** In the series
  model the score after fixing a set depends only on the product of
  the unfixed reliabilities, so the minimal fix set is the set of
  smallest reliabilities -- exactly what greedy picks, in order.
  The minimal-fix-set comparison is therefore a theorem check, not a
  measurement; the real fix-order measurement is the budget
  comparison, where greedy genuinely loses.
- **The systems are hand-built and synthetic.** Sensitivities are
  seeded-uniform, qualities are declared, costs are declared. The
  evaluation measures the tool against its own model, not against
  production systems.
- **Gold is defined as quality 1.0.** In production, "gold" would be
  a human reference output or a stronger model, and the gain would
  be an estimate, not an exact quantity.
- **Series model only.** The product form assumes a serial pipeline
  with no redundancy: if two components do the same job, the model
  overstates the loss of each. Parallel/redundant architectures need
  a different scoring function.
- **Masking is defined by a ratio.** A component counts as masked
  when its gold-swap gain is suppressed to under half of its
  unmasked gain. The 0.5 ratio and 0.02 gain floor are declared
  policy, not measured.

## 7. Masking, in one paragraph

A component is masked when gold-swapping it alone gains little
relative to what it gains once another component is fixed. In the
deliberately broken PHM variant the detector (quality 0.05) lets
through only ~38% of any other component's win, so the sensor,
diagnoser and reporter all look healthy until the detector is fixed.
Fix the masker first.

## 8. Layout

```
gold_swap/        core package (models, systems, swap, attribution,
                  fixorder, masking, evaluate, report, cli)
app.py            Streamlit dashboard (Attribution / Fix Order / Evaluation)
data/             systems.json (system definitions)
examples/         demo system + runnable example + sample report
notebooks/        01 demo run, 02 evaluation experiment
scripts/          build_cases.py, reproduce_results.py (drift guard)
results/          evaluation.json/.csv, evaluation_summary.md, HEADLINE.md
tests/            68 tests
```
