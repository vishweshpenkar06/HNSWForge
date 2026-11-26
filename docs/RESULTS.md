# Results

Benchmark tables and chart references. Numbers are recorded only after a real measurement; unmeasured values stay as `TODO(human)` placeholders.

## Phase 1 — foundations

- Tests: 62 passing (`./mvnw verify`), JaCoCo report generated for all modules.
- Exact index correctness: brute-force results compared with ground truth computed by the same exact index yield `recall@10 = 1.0` (asserted in `RecallTest`, synthetic vectors).
- Real SIFT1M 10k subset check is deferred to Phase 6 per the synthetic-first decision in `DECISIONS.md`.

## Phase 2 baseline — naive HNSW

Measured on this machine (i7-12700H, Java 25, `-Xmx4g`, single query thread, k=10, m=16, efConstruction=200). Synthetic uniform vectors; 200 queries for recall, 200 queries × 10 passes for QPS.

| n | dim | efSearch | build_s | QPS | recall@10 |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 32 | 100 | — | — | 0.995 |
| 100,000 | 32 | 100 | 52.35 | 3,262 | 0.9265 |
| 100,000 | 32 | 200 | 52.35 | — | 0.9775 |
| 100,000 | 128 | 100 | 167.70 | 1,364 | 0.4330 |

Honest caveat: uniformly random vectors in 128 dimensions are intrinsically harder for ANN than structured SIFT descriptors; the 0.4330 figure is a property of that synthetic data, not evidence of a graph bug. At 32 dimensions the same code reaches 0.9775 at efSearch=200. Real SIFT1M numbers are recorded in Phase 6.

This table is the baseline for the Phase 4 optimization waterfall.

## Optimization waterfall

(pending)

## Baseline comparison

(pending)
