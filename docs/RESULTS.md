# Results

Benchmark tables and chart references. Numbers are recorded only after a real measurement; unmeasured values stay as `TODO(human)` placeholders.

## Phase 1 — foundations

- Tests: 62 passing (`./mvnw verify`), JaCoCo report generated for all modules.
- Exact index correctness: brute-force results compared with ground truth computed by the same exact index yield `recall@10 = 1.0` (asserted in `RecallTest`, synthetic vectors).
- Real SIFT1M 10k subset check is deferred to Phase 6 per the synthetic-first decision in `DECISIONS.md`.

## Phase 2 baseline

(pending)

## Optimization waterfall

(pending)

## Baseline comparison

(pending)
