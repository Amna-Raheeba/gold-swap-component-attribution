# Gold-Swap Attribution: rag

Baseline mean score 0.5224 over 20 cases; all-gold 1.0000; ceiling 0.4776.

| Rank | Component | Gold-swap gain | Ceiling share |
| ---: | --------- | -------------: | ------------: |
| 1 | generator | 0.1298 | 0.272 |
| 2 | retriever | 0.1103 | 0.231 |
| 3 | verifier | 0.0747 | 0.156 |
| 4 | reranker | 0.0555 | 0.116 |

Interaction residual (1 - sum of shares): 0.2248. A positive residual means the gains are superadditive: fixing components together wins more than the sum of fixing each alone, so the individual gains leave headroom unaccounted for.

## Pairwise interactions

- generator+verifier: +0.0187 (superadditive)
- reranker+generator: +0.0136 (superadditive)
- reranker+verifier: +0.0080 (superadditive)
- retriever+generator: +0.0288 (superadditive)
- retriever+reranker: +0.0115 (superadditive)
- retriever+verifier: +0.0156 (superadditive)

# Fix Order: rag (target 0.95)

Greedy order: generator, retriever, verifier, reranker.

Scores after each fix: 0.5224, 0.6522, 0.7913, 0.9044, 1.0000.

Greedy reaches the target in 4 steps; the optimal minimal fix set is generator, reranker, retriever, verifier (4 components).

Greedy is optimal here: True. Cumulative order gap (optimal minus greedy total score): 0.0000.

