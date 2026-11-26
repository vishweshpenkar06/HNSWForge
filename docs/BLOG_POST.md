# Building HNSW From Scratch in Java: What the Numbers Actually Say

Approximate nearest-neighbor search powers semantic search; HNSW (Hierarchical Navigable Small World) is the dominant graph index for it. Most engineers meet HNSW as a dependency; I wanted to meet it as an algorithm. This is an honest account of building it from scratch in Java, including what did not work.

TODO(human): why did you personally decide a from-scratch build was worth the time?

## 1. The problem, and why build it from scratch

An ANN index returns the k nearest vectors to a query, trading a little recall for speed. Using a library teaches its API; implementing one teaches its failure modes. My goal was not to beat hnswlib, a mature C++ implementation, but to understand every benchmark number, defend each design choice, and report losses as honestly as wins. No invented numbers, every claim measured — that constraint shaped the project. The result has zero runtime dependencies, SIMD kernels, concurrent inserts, and mmap persistence. It is good in places and 3.0 to 4.8 times slower than hnswlib at equal recall in others. Both are measured.

## 2. HNSW in plain language

HNSW is a stack of navigable maps over the same vectors. Layer 0 holds everything; each layer above is a sparser express lane. Search starts at the top, greedily walks to the closest neighbor until no improvement, then drops a layer and repeats. At layer 0, best-first search uses a candidate min-heap and a result max-heap bounded by `ef`, returning the best k.

Each node's level is drawn from an exponential distribution, `floor(-ln(U) * mL)` with `mL = 1/ln(M)`, so upper layers shrink geometrically. `M` caps neighbors per node above layer 0; raising it improves recall but costs memory and build time. `efConstruction` sizes the candidate list during build. `efSearch` sizes it during queries; raising it improves recall and lowers QPS. On 10,000 synthetic 32-dimensional vectors, raising `efSearch` from 20 to 160 at `M=8` lifted recall@10 from 0.846 to 0.974 but cut QPS from 46,793 to 9,264, about 5x. Raising `M` from 8 to 16 at `efSearch=20` lifted recall from 0.846 to 0.963, but build time more than doubled, from 0.94 s to 2.27 s.

The subtle part is neighbor selection. A naive index keeps the closest neighbors; HNSW's diversity heuristic prefers neighbors spread across regions, preserving connectivity between clusters. My first version had a bug — the most instructive part of the project.

TODO(human): how would you explain the diversity heuristic to a colleague who does not work on retrieval?

## 3. Build the exact baseline first

Before any graph code, I wrote `BruteForceIndex` and a recall evaluator against its ground truth. The first test asserts brute force compared with itself yields recall@10 = 1.0 on synthetic vectors. Trivial, and that is the point: a trusted reference. When recall looked suspicious later, the question was always "relative to what?" An exact oracle turned vague doubt into a measurement and made the synthetic-first decision safe, deferring large real datasets without blocking the build. With no runtime dependencies, a mystery result had nowhere to hide.

TODO(human): describe a specific moment when the brute-force baseline caught or ruled out a problem.

## 4. Three surprises

**The diversity-heuristic bug.** Phase 2 compared `dist(base, candidate)` against `dist(base, selected)` instead of the correct `dist(candidate, selected)` from Algorithm 4. That degenerates the heuristic into nearest-neighbor selection, producing a sparse graph, average degree 12.5, with recall@10 of 0.9265 on 100,000 synthetic 32-dimensional vectors at `efSearch=100`. After the fix the graph was dense and fully connected, average degree 22.6 with 100 percent reachability at layer 0, and recall@10 reached 0.9995 at the same `efSearch`. The buggy graph was faster, so optimizing QPS against a broken index rewards the wrong code. I kept the old numbers as the baseline; replacing them would have made the waterfall look better than reality.

TODO(human): what surprised you most about the heuristic bug?

**Uniform high-dimensional data is intrinsically hard.** On the same code, 100,000 uniform random vectors in 128 dimensions reached recall@10 = 0.4330 at `efSearch=100`, with a 167.70 s build and 1,364 QPS. In 32 dimensions the same problem reached 0.9775 at `efSearch=200`. Uniform points in 128 dimensions have little structure for a graph to exploit, so this is a property of the synthetic data, not a graph defect. Recording that caveat prevented a false alarm; a recall number is meaningless without its dataset.

TODO(human): what was your reaction when you first saw the 0.4330 figure, and how did you confirm it was the data?

**SIMD helped build time at least as much as query throughput.** Vector API kernels gave a 1.56x QPS gain over flat scalar, 714 to 1,112 QPS, and a slightly larger 1.59x build-time gain, 48.10 s to 30.18 s. I expected the win in search, but the build is also a long sequence of distance computations during neighbor selection, so it benefits from the same kernels.

TODO(human): did the build-time result change where you would invest optimization effort next?

## 5. The optimization waterfall, and the target we missed

All rows are real measurements: 100,000 vectors, dimension 32, `M=16`, `efConstruction=200`, `efSearch=100`, k=10, one thread, 200 queries, 5 timed passes with 3 warmups.

| Step | Configuration | build_s | QPS | recall@10 |
|---:|---|---:|---:|---:|
| 0 | Phase 2 baseline: object graph, scalar, degenerate heuristic | 52.35 | 3,262 | 0.9265 |
| 1 | Flat vectors + flat adjacency + epoch visited + scratch heaps (scalar, no prefetch) | 48.10 | 714 | 0.9995 |
| 2 | + SIMD distance kernels | 30.18 | 1,112 | 0.9995 |
| 3 | + off-heap vector storage (MemorySegment) | 38.42 | 981 | 0.9995 |
| 4 | + prefetch-friendly traversal (heap, SIMD, prefetch) | 29.08 | 1,203 | 0.9995 |
| 5 | Final combination: SIMD + off-heap + prefetch | 39.25 | 923 | 0.9995 |

**The 3x search speedup target was not met.** Cumulative QPS change from baseline is 923 / 3,262 = 0.28x, and even the fastest configuration, 1,203 QPS, is below baseline. QPS falls because step 1 fixed the heuristic and raised recall from 0.9265 to 0.9995; a denser graph costs more distance computations, so the baseline's speed was partly an artifact of lower recall. Quality never dropped. Build time improved from 52.35 s to 29.08 s, a 1.8x gain. Off-heap storage was slower than heap at 100,000 vectors, 981 vs 1,112 QPS, because FFM reads add overhead; its value is avoiding GC pressure on larger indexes, which this datapoint does not show. A JMH GC profile measured 320.867 B per search, under the 1 KB target.

## 6. The honest baseline comparison

Fixed setup: i7-12700H, OpenJDK 25.0.4.1, 15 GiB RAM, Linux (Nobara/Fedora), synthetic uniform vectors, n=10,000, dim 32, k=10, 200 queries, `M=16`, `efConstruction=200`, 3 warmups, 5 timed passes.

| System | efSearch | recall@10 | QPS | p50_us | p95_us | p99_us | build_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| hnswforge | 10 | 0.9995 | 3,026 | 333 | 373 | 416 | 2.90 |
| hnswforge | 160 | 1.0000 | 2,931 | 337 | 383 | 415 | 2.90 |
| hnswforge | 320 | 1.0000 | 4,760 | 208 | 227 | 239 | 2.90 |
| hnswlib | 10 | 0.9210 | 124,307 | 7 | 9 | 17 | 0.93 |
| hnswlib | 80 | 0.9940 | 29,644 | 32 | 41 | 47 | 0.93 |
| hnswlib | 160 | 0.9995 | 14,401 | 68 | 74 | 83 | 0.93 |
| Lucene | any | 0.9665 | 9,073-11,076 | 85-113 | 115-202 | 148-366 | 1.64 |
| pgvector | 40 | 0.9995 | 2,950 | 308 | 559 | 677 | 8.90 |
| pgvector | 320 | 1.0000 | 811 | 1,174 | 1,841 | 2,093 | 8.90 |

At equal recall of 0.9995, hnswlib delivers 14,401 QPS versus 3,026 at `efSearch=10` and 4,760 at `efSearch=320` for HNSWForge — a 3.0x to 4.8x gap, plus a 3.1x build-time gap (0.93 s vs 2.90 s). Lucene is faster at 9,000-11,000 QPS but recall is fixed at 0.9665 because it exposes no query-time `efSearch`. pgvector at equal recall is roughly on par, 2,950 vs 3,026 QPS, but builds in 8.90 s and pays network/JDBC overhead.

Four evidence-based hypotheses explain the loss. First, no native prefetch or autovectorization: hnswlib uses raw pointers and `-march=native`; this library reads through Vector API offsets with bounds checks. Second, a denser layer-0 graph: fixing the heuristic raised average degree from 12.5 to 22.6, improving recall at the cost of more distance computations. Third, per-neighbor validation and allocation: every candidate passes a published/tombstone atomic read and bounds check, and each query allocates a result list and up to k records. Fourth, JIT warmup and GC: QPS varied from 2,864 to 4,760 across `efSearch` values at constant recall, noise absent from the native baseline. No settings were cherry-picked; all values from 10 to 320 are reported.

TODO(human): which of these four hypotheses do you believe matters most, and how would you test that?

## 7. Concurrency and persistence

Parallel builds scale sublinearly because insertion contends on striped neighbor-list locks and the shared entry-point region.

| Threads | build_s | speedup vs 1 thread |
|---:|---:|---:|
| 1 | 74.23 | 1.00x |
| 2 | 44.46 | 1.67x |
| 4 | 31.94 | 2.32x |
| 8 | 27.93 | 2.66x |

Mapping a 1,000,000-vector index file of 264,000,128 bytes took 0.0776 s, versus 0.262 s for a full write. This measures mapping and header validation only; the synthetic file has no edges, so its queries are not comparable. Stress tests passed: 8 threads inserting 50,000 vectors preserved degree, self-loop, and duplicate invariants; parallel and sequential builds agreed on recall within 0.01; mixed insert/search threads ran 10 seconds with no failures.

TODO(human): what would you measure next to show the value of mmap on a real, fully built index?

## 8. What I would do next

The largest gap is native query throughput, so first I would reduce per-candidate validation and restructure traversal to exploit prefetch without bounds-check costs. The off-heap result suggests testing larger indexes where GC pressure dominates, since 100,000 vectors is too small to show that benefit. Then come int8 and product quantization, and filtered search.

Measurement rigor matters as much. All numbers here are synthetic; SIFT1M and GloVe were not downloaded, and the real-data rerun remains open. The RepoPrism adapter hit hit@6 = 1.0000 on a 200-chunk synthetic fixture against exact search, but the real 72-case evaluation and pgvector-only numbers still need a real export. Those tasks turn this synthetic result into a claim about real retrieval.

TODO(human): which single next step would you actually take first, given a free weekend?
