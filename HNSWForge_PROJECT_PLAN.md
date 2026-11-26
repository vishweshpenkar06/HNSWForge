# HNSWForge: A Vector Search Engine Built From Scratch in Java

> Working name. A dependency-free Java library implementing HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor search, with SIMD distance kernels, off-heap storage, concurrent inserts, and disk persistence. Benchmarked honestly against hnswlib, Lucene, and pgvector, and integrated into RepoPrism.

---

## 1. Why this project

| Goal | How this project serves it |
|---|---|
| Stand out for 2028 hiring | Few undergrads implement a core data-structure from a paper, then optimize and benchmark it |
| Show systems depth | SIMD, off-heap memory, concurrency, mmap persistence, profiling |
| Show rigor | Every claim backed by recall, QPS, memory, and build-time numbers |
| Connect to existing work | Swaps into RepoPrism as a retrieval backend, with before/after eval numbers |
| Be defensible in interviews | Every number is reproducible; every design choice has a reason |

**Target resume bullet (fill in real numbers):**
> Implemented HNSW in Java with SIMD distance kernels, off-heap storage, and mmap persistence; achieved X% recall@10 at Y QPS on SIFT1M (within Z× of hnswlib); integrated into RepoPrism, improving/maintaining hit@6 at N%.

---

## 2. Scope

### In scope (MVP)
- HNSW index: insert, k-NN search, tunable `M`, `efConstruction`, `efSearch`
- Distance metrics: L2, cosine, inner product
- Exact (brute-force) search for ground truth
- Recall@k evaluation harness
- SIMD distance computation (Java Vector API)
- Flat/off-heap storage layout
- Concurrent inserts and lock-free reads
- Save/load index to disk (mmap)
- Benchmark suite vs hnswlib, Lucene HNSW, pgvector
- RepoPrism integration as a pluggable backend

### Stretch goals
- Deletes (tombstones, then optional graph repair)
- int8 scalar quantization (4x memory reduction) and product quantization
- Filtered search (metadata predicates during traversal)
- Tiny REST/gRPC wrapper
- Incremental/streaming index updates with background compaction

### Out of scope
- Distributed sharding
- GPU acceleration
- Full database features (transactions, SQL)

---

## 3. Architecture

```
hnswforge/
├── hnswforge-core/           # the library (no external deps)
│   ├── distance/             # L2, cosine, IP; scalar + SIMD implementations
│   ├── storage/              # vector store (heap array -> off-heap segment)
│   ├── graph/                # layered adjacency structure
│   ├── index/                # HnswIndex: insert, search, params
│   ├── persist/              # serialization, mmap loader
│   └── exact/                # BruteForceIndex (ground truth)
├── hnswforge-bench/          # JMH microbenchmarks + end-to-end harness
├── hnswforge-eval/           # recall@k, dataset loaders (SIFT1M, GloVe)
├── hnswforge-repoprism/      # adapter for RepoPrism integration
├── hnswforge-server/         # (stretch) REST wrapper
└── docs/                     # architecture, benchmark results, blog post
```

### Core API sketch

```java
public interface VectorIndex extends AutoCloseable {
    int add(float[] vector);                       // returns internal id
    List<SearchResult> search(float[] query, int k);
    List<SearchResult> search(float[] query, int k, int efSearch);
    int size();
    void save(Path path) throws IOException;
    static VectorIndex load(Path path) throws IOException { ... }
}

public record SearchResult(int id, float distance) {}

public record HnswParams(
    int dimension,
    Metric metric,        // L2, COSINE, INNER_PRODUCT
    int m,                // max neighbors per node (layers > 0), default 16
    int efConstruction,   // candidate list size during build, default 200
    int efSearch          // candidate list size during search, default 64
) {}
```

### Key design decisions to be ready to explain
1. **Why layers?** Upper layers are sparse "express lanes" giving log-like entry into the graph; layer 0 holds all nodes.
2. **Why a random level?** Level is drawn from an exponential distribution (`floor(-ln(U) * mL)`, `mL = 1/ln(M)`), which keeps layer sizes geometrically decreasing.
3. **Why the neighbor-selection heuristic?** It prefers diverse neighbors (not just the closest), which preserves connectivity across clusters and improves recall.
4. **Why flat arrays instead of objects?** Cache locality, no per-object header overhead, and less GC pressure.
5. **Why off-heap?** Large indexes avoid GC pauses and can be memory-mapped from disk.

---

## 4. Algorithm reference (know this cold)

### Search (per layer)
1. Start from the entry point at the top layer.
2. Greedy descend: at each upper layer, move to the neighbor closest to the query until no improvement.
3. At layer 0, run best-first search with a candidate min-heap and a result max-heap bounded by `ef`.
4. Return the top `k` of the result set.

### Insert
1. Draw a random level `L` for the new node.
2. Descend from the entry point to layer `L+1` greedily.
3. For each layer from `min(L, topLayer)` down to 0: search with `efConstruction`, select up to `M` neighbors with the heuristic, connect bidirectionally, and prune neighbors that exceed their max degree.
4. If `L` exceeds the current top layer, update the entry point.

### Parameters and tradeoffs

| Param | Increase it to... | Cost |
|---|---|---|
| `M` | improve recall, especially in high dimensions | more memory, slower build |
| `efConstruction` | build a higher-quality graph | slower build |
| `efSearch` | improve recall at query time | lower QPS |

---

## 5. Datasets

| Dataset | Vectors | Dim | Metric | Use |
|---|---|---|---|---|
| SIFT1M | 1,000,000 | 128 | L2 | Main benchmark |
| GloVe-100 | ~1.18M | 100 | Cosine/angular | Second benchmark |
| RepoPrism chunks | your corpus | 384 (MiniLM) | Cosine | Real-world integration test |
| Small synthetic | 10k-100k | any | any | Fast unit/dev tests |

Start every experiment on 10k-100k vectors; scale to 1M only once correct.

---

## 6. Week-by-week plan (about 8 weeks, part-time)

### Week 1: Understand and build the baseline
- [ ] Read the HNSW paper (Malkov & Yashunin, 2016/2018). Write a 1-page summary in your own words.
- [ ] Explain layers, `M`, `efConstruction`, `efSearch`, and the heuristic aloud without notes.
- [ ] Implement `BruteForceIndex` (exact k-NN).
- [ ] Implement dataset loaders (fvecs/ivecs for SIFT; text for GloVe).
- [ ] Implement `recall@k` evaluator comparing approximate results to ground truth.
- **Exit criteria:** brute force runs on SIFT subset; recall evaluator verified (brute force vs itself = 1.0).

### Week 2: Naive HNSW that works
- [ ] Single-threaded HNSW with simple Java objects and scalar distance.
- [ ] Insert + search following the paper (simple neighbor selection first).
- [ ] Unit tests: tiny hand-checked graphs, determinism with a fixed seed.
- **Exit criteria:** recall@10 >= 95% on 100k vectors with sensible params.

### Week 3: Correctness and tuning
- [ ] Add the heuristic neighbor selection; compare against simple selection.
- [ ] Parameter sweeps over `M`, `efConstruction`, `efSearch`.
- [ ] Produce first recall vs QPS curve (plot with Python/matplotlib or any tool).
- [ ] Optional: tombstone deletes.
- **Exit criteria:** documented recall/QPS curve; heuristic shows measurable benefit.

### Week 4: Performance engineering
- [ ] Replace object graph with flat primitive arrays (`int[]` adjacency, contiguous vectors).
- [ ] SIMD distance kernels with the Java Vector API (`jdk.incubator.vector`).
- [ ] Off-heap vector storage via the Foreign Function & Memory API.
- [ ] JMH microbenchmarks for distance kernels and end-to-end search.
- [ ] Profile with async-profiler; record each optimization with before/after numbers.
- **Exit criteria:** at least 3-5x speedup over Week 3 baseline, with a log of what contributed what.

### Week 5: Concurrency and persistence
- [ ] Concurrent insert: per-node (or striped) locks; lock-free reads where safe.
- [ ] Stress tests: concurrent insert + search, verify no corruption and recall holds.
- [ ] `save()` / `load()` to a binary format with a versioned header.
- [ ] Memory-mapped loading to avoid full deserialization.
- **Exit criteria:** multi-thread build speedup measured; reload gives identical search results.

### Week 6: Honest benchmarking
- [ ] Fix hardware, JVM flags, thread count, and dataset for all runs; record them in the README.
- [ ] Baselines: hnswlib (Python harness or JNI), Lucene HNSW, pgvector HNSW.
- [ ] Metrics: recall@10 vs QPS curve, build time, index size, peak memory, p50/p95/p99 latency.
- [ ] Report wins *and* losses, with hypotheses for the gaps.
- **Exit criteria:** reproducible benchmark script and a results table + charts.

### Week 7: RepoPrism integration
- [ ] Define a `RetrievalBackend` interface in RepoPrism if not present.
- [ ] Add HNSWForge adapter; keep pgvector as an alternative.
- [ ] Rerun the 72-case eval: hit@6, latency, throughput, memory.
- **Exit criteria:** before/after table for RepoPrism.

### Week 8: Package and publish
- [ ] README: problem, quick start, architecture diagram, benchmark charts, limitations.
- [ ] Publish to JitPack or Maven Central.
- [ ] CI via GitHub Actions: build, tests, a small benchmark smoke test.
- [ ] Blog post: what you built, what surprised you, what failed.
- [ ] Update resume with real measured numbers.
- **Exit criteria:** a stranger can clone, run one command, and reproduce a benchmark.

---

## 7. Testing strategy

| Layer | Tests |
|---|---|
| Distance | SIMD vs scalar equality within epsilon, random vectors, odd dimensions |
| Graph invariants | Degree bounds respected, bidirectional links, entry point valid |
| Search | Recall vs brute force across params; k > size; empty index; duplicate vectors |
| Concurrency | Parallel inserts with random seeds; searches during inserts; no exceptions or lost nodes |
| Persistence | Round-trip identical results; corrupted/truncated file handled; version mismatch |
| Property-based | Random data, random params, recall never below a floor |

Goal: high test count and clear invariants, like your other projects.

---

## 8. Benchmark methodology (do this right)

1. Same machine, same data, same `k`, same number of threads for every system.
2. Warm up the JVM before measuring; use JMH for microbenchmarks.
3. Report **recall and QPS together** (a recall vs QPS curve), never QPS alone.
4. Report build time, index size, and peak memory.
5. Run multiple times; show variance or median.
6. Publish the exact commands and configs.

**Chart set for the README:**
- Recall@10 vs QPS (per system, per dataset)
- Build time vs dataset size
- Memory per million vectors
- Latency percentiles at fixed recall
- Optimization waterfall (each Week 4 change's contribution)

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Recall is low due to subtle bugs | Brute-force baseline from Day 1; test on small data first |
| Vector API is still incubating | Keep a scalar fallback; document required JVM flags |
| Concurrency bugs corrupt the graph | Start with a global lock, then refine; stress test heavily |
| Benchmarks look unfair | Publish setup; accept and explain losses |
| AI-generated core you can't explain | Write or deeply review insert/search/selection yourself |
| Scope creep | Freeze MVP at Week 6; stretch goals only if ahead |

---

## 10. Interview preparation checklist

Be able to answer, without notes:
- [ ] Why does HNSW use multiple layers? What is the expected layer distribution?
- [ ] What do `M`, `efConstruction`, `efSearch` each control, and what are the tradeoffs?
- [ ] Why does the neighbor-selection heuristic improve recall?
- [ ] What happens on insert when a neighbor's degree exceeds `M`?
- [ ] How do deletes work and why are they hard in graph indexes?
- [ ] Why is SIMD faster here, and what limits the speedup (memory bandwidth)?
- [ ] How did you make concurrent inserts safe? What are the remaining races?
- [ ] Where did you lose to hnswlib/Lucene and why?
- [ ] How would you scale this beyond one machine?
- [ ] What would you do differently?

---

## 11. Deliverables checklist

- [ ] GitHub repo with clean structure, license, and CI
- [ ] Tests (target 150+) passing in CI
- [ ] Benchmark scripts + results + charts committed
- [ ] Architecture doc with diagrams
- [ ] RepoPrism integration PR with before/after eval
- [ ] Published artifact (JitPack/Maven Central)
- [ ] Blog post (dev.to / personal site / Medium)
- [ ] Resume bullet with measured numbers
- [ ] LinkedIn post with one benchmark chart

---

## 12. Resources

- **Paper:** "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin)
- **Reference implementations to study (not copy):** hnswlib, Lucene's `HnswGraph`
- **Benchmarking:** ann-benchmarks (methodology and datasets), JMH, async-profiler
- **Datasets:** SIFT1M and GloVe from the ANN-Benchmarks / TEXMEX corpora
- **Java:** Vector API (`jdk.incubator.vector`), Foreign Function & Memory API, `java.util.concurrent`

---

## 13. Suggested README skeleton

```
# HNSWForge
One-line pitch + badges (build, license, Maven/JitPack)

## Why
## Quick start (5 lines of code)
## Features
## Benchmarks (charts + table + exact setup)
## Architecture (diagram)
## Design decisions & tradeoffs
## Limitations
## Roadmap
## Reproducing results
## License
```

---

## 14. Success criteria

**Minimum (hire-worthy):** correct HNSW, >=95% recall@10 on SIFT1M, SIMD + off-heap, honest benchmark vs at least one baseline, RepoPrism integration, clean README.

**Strong:** within ~2-3x of hnswlib QPS at equal recall, concurrent build, mmap persistence, published artifact, blog post.

**Exceptional:** quantization with measured tradeoffs, filtered search, external users or stars, a merged PR or citation from another project.
