# Gold-Swap Evaluation Summary

Seed 7, 200 cases per system, target score 0.95.

- Max gain-recovery error: 0.000000
- Max ceiling-accounting error: 0.000000
- Greedy reaches the target with the minimal fix set on 100% of the 3 hand-built systems and 100% of 50 random systems -- expected: greedy is provably optimal for the minimal fix set in the series model (see README section 6), and its full order matched the optimal full order on 100% of random systems. The real fix-order measurement is under a budget of 6: greedy by gain matches the exhaustive optimum on 92% of random systems (mean score gap 0.0050, max 0.0901), greedy by gain per cost on 78% (mean gap 0.0120, max 0.1495).
- Masking detection: precision 1.00, recall 1.00; false positives on the 3 healthy systems: 0.

## rag

- Baseline 0.5159, ceiling 0.4841, bottleneck generator.
- Gains: generator 0.1389, retriever 0.1046, verifier 0.0744, reranker 0.0561.
- Ceiling shares: generator 0.287, retriever 0.216, verifier 0.154, reranker 0.116; interaction residual 0.2274.
- Greedy fix order: generator, retriever, verifier, reranker (4 steps to 0.95); optimal set: generator, reranker, retriever, verifier.
- Budget 6.0: optimal reranker, retriever, verifier (score 0.7873, cost 6.0); greedy by gain generator, reranker (score 0.7263, gap 0.0610); greedy by gain per cost reranker, verifier, retriever (score 0.7873, gap 0.0000).

## research

- Baseline 0.5545, ceiling 0.4455, bottleneck synthesizer.
- Gains: synthesizer 0.1377, planner 0.0982, tool_caller 0.0741, critic 0.0453.
- Ceiling shares: synthesizer 0.309, planner 0.220, tool_caller 0.166, critic 0.102; interaction residual 0.2025.
- Greedy fix order: synthesizer, planner, tool_caller, critic (4 steps to 0.95); optimal set: critic, planner, synthesizer, tool_caller.
- Budget 6.0: optimal planner, synthesizer (score 0.8149, cost 6.0); greedy by gain synthesizer, planner (score 0.8149, gap 0.0000); greedy by gain per cost planner, critic, tool_caller (score 0.8006, gap 0.0143).

## phm

- Baseline 0.5557, ceiling 0.4443, bottleneck detector.
- Gains: detector 0.1658, diagnoser 0.0947, reporter 0.0623, sensor 0.0355.
- Ceiling shares: detector 0.373, diagnoser 0.213, reporter 0.140, sensor 0.080; interaction residual 0.1935.
- Greedy fix order: detector, diagnoser, reporter, sensor (4 steps to 0.95); optimal set: detector, diagnoser, reporter, sensor.
- Budget 6.0: optimal detector, reporter (score 0.8023, cost 5.0); greedy by gain detector, reporter (score 0.8023, gap 0.0000); greedy by gain per cost reporter, detector (score 0.8023, gap 0.0000).
