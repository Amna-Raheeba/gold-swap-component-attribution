# Changelog

## 0.1.0 — 2026-11-17

- Initial release.
- Gold-swap attribution: per-component gains, ceiling shares,
  interaction residual, pairwise interactions, bottleneck ranking.
- Fix order: greedy vs exhaustive optimal (minimal fix set, full
  order), plus budget-constrained comparison (greedy by gain,
  greedy by gain per cost, exhaustive optimal).
- Masked-component detection via second-order (ratio) experiments.
- Three hand-built systems (RAG, agentic research, PHM) plus a
  deliberately broken PHM variant for masking.
- Evaluation: gain recovery vs analytic ground truth, ceiling
  accounting identity, greedy-vs-optimal, 50 random systems,
  masking precision/recall with a healthy-system control.
- Streamlit dashboard (3 tabs), demo + evaluation notebooks,
  reproduce-results script with a headline-drift guard.
- 50 tests, CI on Python 3.10 / 3.11 / 3.12.
