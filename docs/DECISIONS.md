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
