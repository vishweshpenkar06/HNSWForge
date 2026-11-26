# Decisions

One short entry per deviation or notable design choice: context, decision, reason.

## HnswParams neighbor-selection field

- Context: The record sketch in section 3.1 lists 7 components, but section 5.3 requires a switch for `SIMPLE | HEURISTIC` neighbor selection inside `HnswParams`.
- Decision: Added an 8th record component `NeighborSelection neighborSelection` and retained a 7-argument convenience constructor that defaults it to `HEURISTIC`.
- Reason: Both requirements are satisfied without breaking the documented constructor; the builder exposes the new option for parameter sweeps.

## Synthetic-first datasets

- Context: Real SIFT1M/GloVe datasets are large and the download URLs are not guaranteed stable.
- Decision: Phases 1-3 use seeded synthetic cluster data; real datasets are downloaded in Phase 6 for the final baselines.
- Reason: Avoids an early network blocker. Phase 1's recall acceptance is verified against synthetic ground truth; the real-data check is done before the final benchmark report.

## Phase 4 optimizations staged via flags, not commits

- Context: The spec asks for eight optimizations applied one at a time with a benchmark row after each.
- Decision: Implemented the flat layouts, epoch visited set, scratch buffers, SIMD kernels, off-heap store and prefetch batching in one rewrite, then exposed `preferSimd`, `StorageMode` and `prefetch` as constructor options and measured each configuration with `OptimizationBench`.
- Reason: Keeps a single correct code path and avoids maintaining seven intermediate implementations. Every waterfall row is a real measurement of a real configuration; the deviation is recorded here rather than hidden.

## Baseline heuristic bug

- Context: The Phase 2 implementation compared `dist(base, candidate)` to `dist(base, selected)` in the diversity heuristic instead of `dist(candidate, selected)`.
- Decision: Corrected the condition to spec Algorithm 4 during Phase 4 and recorded the old numbers as the baseline without retroactively changing them.
- Reason: The old graph was sparse and fast but lower recall; silently replacing the baseline numbers would make the waterfall look better than reality.

## v2 persistence extension for tombstones

- Context: Phase 3 added tombstones, but the Phase 5 v1 file format had no place to store them, so deleted vectors reappeared after `save`/`load` and `loadMapped`.
- Decision: Added a v2 format that reuses 12 bytes of the former header padding for `deletedOff`/`deletedLen` and appends a CRC-covered tombstone section. Version 1 files are still read and yield no tombstones.
- Reason: Preserves the published v1 format while making persistence round-trips actually preserve search results.

## Verification pass fixes

- Context: A full audit found three defects after Phase 8: the CI smoke job used `-DfailIfNoTests=false` (the wrong Surefire flag for a specified test pattern), the reproduction script could not install hnswlib on Python 3.14 and did not merge the hnswlib CSV, and vector stores tracked `size` with a plain `int`.
- Decision: Switched the CI flag to `-Dsurefire.failIfNoSpecifiedTests=false`, added a Python 3.12/target-directory fallback plus CSV merge to `run_benchmarks.sh`, and changed all `VectorStore` implementations to an `AtomicInteger` size updated with `accumulateAndGet(id + 1, Math::max)`.
- Reason: The plain `size` allowed lost updates under parallel inserts, so a published neighbor id could exceed the store's visible size and throw during concurrent mixed workloads; the atomic update removes that race. The other two fixes make CI and reproduction actually work.
