# Resume Bullets

Measured numbers only. Every figure below comes from `docs/RESULTS.md`; a number that is not measured stays a `TODO(human)` placeholder.

## Variant A — Research / Systems

- Implemented HNSW approximate nearest-neighbor search from scratch in Java 25 with zero runtime dependencies, validated by 270 tests and 90.3% core line coverage.
- Achieved recall@10 of 0.9995 to 1.0000 on 10,000-vector synthetic benchmarks at 3,026 to 4,760 QPS, verified against exact brute-force ground truth.
- Engineered SIMD distance kernels delivering a 1.56x QPS gain (714 to 1,112 QPS) and a 1.59x build-time gain (48.10 s to 30.18 s), reducing final build time from 52.35 s to 29.08 s.
- Benchmarked honestly against hnswlib, Lucene, and pgvector, documenting a measured 3.0x to 4.8x QPS gap and 3.1x build-time gap to hnswlib at equal recall with four evidence-based hypotheses.

## Variant B — Applied / ML Infra

- Built a dependency-free vector search index supporting exact and HNSW approximate k-NN with tunable M, efConstruction, and efSearch, covered by 270 tests at 90.3% core line coverage.
- Reached recall@10 of 0.9995 to 1.0000 at 3,026 to 4,760 QPS on 10,000 synthetic vectors, with a RepoPrism adapter hitting 100% hit@6 on a synthetic fixture against exact search.
- Reduced build time from 52.35 s to 29.08 s using SIMD kernels (1.59x), flat memory layouts, and prefetch-friendly traversal, while holding 320.867 B/op allocation on the search path.
- Added mmap persistence loading a 264 MB, 1,000,000-vector index file in 0.0776 s, and measured 2.66x parallel-build speedup at 8 threads.

## Variant C — Startup / General

- Built a vector search engine from the ground up in Java with no external dependencies, shipping 270 tests and 90.3% core line coverage.
- Delivered 0.9995 to 1.0000 recall@10 at 3,026 to 4,760 QPS on a 10,000-vector benchmark, measured against exact ground truth rather than estimated.
- Cut build time from 52.35 s to 29.08 s and sped up parallel builds 2.66x with 8 threads; memory-mapped a 264 MB, 1M-vector index in 0.0776 s.
- Compared directly with hnswlib, Lucene, and pgvector, reporting a candid 3.0x to 4.8x gap to hnswlib at equal recall and the four reasons behind it.

TODO(human): if any additional metric is required by a specific job posting (for example real SIFT1M/GloVe QPS, production RepoPrism hit@6, or index size in MB per million vectors), replace this placeholder with a measured value only after the real-data run is complete.
