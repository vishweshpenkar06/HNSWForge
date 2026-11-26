# HNSWForge

Dependency-free HNSW approximate nearest-neighbor search for Java 25, with SIMD kernels, off-heap storage, concurrent inserts and mmap persistence.

> This README is expanded in Phase 8. Build and benchmark instructions are in `docs/RESULTS.md`.

## Modules

- `hnswforge-core` — the library (zero runtime dependencies)
- `hnswforge-eval` — dataset readers, ground truth, recall, CLI
- `hnswforge-bench` — JMH microbenchmarks and comparison harness
- `hnswforge-repoprism` — adapter for the RepoPrism retrieval backend

## Build

```bash
./mvnw verify
```

Java 25 is required. The Vector API incubator module is added automatically by the build.

## License

Apache-2.0. See `LICENSE`.
