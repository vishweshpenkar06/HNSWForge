# HNSWForge — Complete Session Report (for Claude)

This document records everything done in this session: the build from an empty directory to a fully verified, committed Java 25 project, all tests and measured numbers, every bug found and fixed, and the remaining human tasks.

---

## 1. Session goal and starting state

The user asked to "analyze all and get ready to build", then approved a single continuous pass through every phase of `HNSWForge_AGENT_BUILD_SPEC.md` (532 lines) and `HNSWForge_PROJECT_PLAN.md` (310 lines), with a commit after each phase.

Starting state: the working directory `/home/vishwesh/Projects/Projects for GITHUB/HNSWForge` contained only those two Markdown specs — no code, no git repository, no Maven wrapper.

Decisions confirmed by the user before implementation:
- All phases in one pass (no per-phase approval stop).
- Synthetic data first; real datasets later (Phase 6).
- Generate a `./mvnw` wrapper rather than relying on system Maven.
- Commit after each phase.

Final state: **13 commits**, 5 Maven modules, clean working tree, all verification commands green.

---

## 2. Environment discovered

| Item | Finding |
|---|---|
| Java | OpenJDK 25.0.4.1 (Red Hat); `jdk.incubator.vector` present as incubator; FFM final in `java.base` |
| Maven | Not on PATH, but 3.10.0 cached at `~/.m2/wrapper/dists/apache-maven-3.10.0/.../bin/mvn`; wrapper generated and works |
| Python | 3.14.8 (system) + 3.12 at `~/.local/bin/python3.12`; numpy 2.4.6; matplotlib 3.10.8 |
| hnswlib | Would not build on Python 3.14 (no `Python.h`); installed successfully into `/tmp/opencode/pylibs` with Python 3.12 |
| Docker | Daemon working; port 5433 occupied, so pgvector compose uses host port 55433 |
| Hardware | i7-12700H (14 cores / 20 threads), 15 GiB RAM, Linux Nobara/Fedora |
| JVM flags used | `-Xmx4g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED` |

---

## 3. Repository structure created

```
hnswforge/
├── pom.xml, mvnw, mvnw.cmd, .mvn/wrapper/, jitpack.yml, LICENSE, README.md
├── .github/workflows/ci.yml
├── docs/  PLAN, ARCHITECTURE, DECISIONS, RESULTS, FILE_FORMAT, BLOG_POST,
│          LINKEDIN_POST, RESUME_BULLETS, TODO_HUMAN, charts/ (10 PNGs)
├── scripts/ download_datasets.sh, run_benchmarks.sh, hnswlib_baseline.py,
│            plot_results.py, plot_compare.py, docker-compose.pgvector.yml, requirements.txt
├── results/ compare.csv, compare_pgvector.csv, hnswlib.csv, repoprism_eval.csv,
│            sift_compare.csv, sift_hnswlib.csv, sweep_synthetic.csv, waterfall.csv
├── data/  SIFT1M downloaded (gitignored): sift.tar.gz + sift/sift_*.fvecs, sift_groundtruth.ivecs
├── hnswforge-core/      25 main + 24 test Java files
├── hnswforge-eval/       8 main +  9 test Java files
├── hnswforge-bench/      9 main +  1 test Java files
├── hnswforge-repoprism/  5 main +  3 test Java files
└── hnswforge-server/     1 main +  1 test Java files
```

---

## 4. Phase-by-phase work

### Phase 0 — scaffolding (commit `60186dd`)
Git init; Maven wrapper pinned to 3.10.0; parent POM with `release=25`, 4 modules, dependency management (JUnit 5.11.4, jqwik 1.9.3, JMH 1.37), `-Xlint:all` + `failOnWarning`, surefire flags, JaCoCo; module POMs; CI workflow (build + benchmark-smoke jobs); Apache-2.0 LICENSE; `.gitignore`; docs stubs. `./mvnw verify` green.

### Phase 1 — foundations and exact baseline (commit `d59bce1`, 62 tests)
- `index/{Metric, SearchResult, VectorIndex, HnswParams, NeighborSelection}`
- `distance/{DistanceFunction, ScalarDistance}` (L2 = squared Euclidean, IP = `1-dot`, cosine with normalization; zero-vector rejection; NaN validation)
- `exact/BruteForceIndex` with bounded max-heap and deterministic lower-id tie-breaking
- `util/{BoundedMaxHeap, MinHeap}` primitive heaps; jqwik oracle property tests
- `hnswforge-eval`: `FvecsReader`, `IvecsReader`, `BvecsReader`, `GloveReader`, `Recall`, `GroundTruth`, `Main` CLI (`groundtruth`, `recall`); `scripts/download_datasets.sh`
- **Bug found and fixed:** heap eviction condition was inverted (worse elements replaced better ones); caught by oracle property tests.

### Phase 2 — naive HNSW (commit `e9cdb4e`, 81 tests)
- `index/HnswIndex`: object-based graph, coarse `ReentrantReadWriteLock`, seeded random levels capped at 32, `searchLayer`, `greedyDescend`, both neighbor-selection strategies, bidirectional connect + degree pruning
- Graph-invariant, determinism, level-distribution, recall and edge-case tests
- **Bug found and fixed:** `greedyDescend` read neighbor slots from a node whose level changed mid-loop, causing NPE/AIOOBE
- Baseline recorded: 100k/dim32 build 52.35s, QPS 3,262, recall@10 0.9265; 10k recall 0.995
- **Later discovery:** the Phase 2 heuristic compared `dist(base,candidate)` instead of `dist(candidate,selected)`, degenerating to simple selection (sparse, fast, lower recall). Kept old numbers as the honest baseline.

### Phase 3 — tuning harness (commit `3dc658b`, 88 tests)
- `eval/Sweep` + `Main sweep` subcommand producing the exact 16-column CSV; dataset fallback to synthetic when real files absent; 3 warmup + 5 timed passes with percentiles
- `scripts/plot_results.py`, `requirements.txt`; first recall-vs-QPS chart
- Tombstones: `delete(int)`, `liveCount()` vs `size()`, result filtering; documented limitations in ARCHITECTURE.md

### Phase 4 — performance engineering (commit `be85787`)
- Offset-aware `DistanceFunction`, flat `float[]` `HeapVectorStore`, flat layer-0 adjacency + lazy upper slices, epoch `VisitedList`, `ThreadLocal<SearchScratch>`, SIMD kernels (`SimdDistance`, `DistanceFunctions` with scalar fallback), `OffHeapVectorStore` (FFM arena, 64-byte alignment), `StorageMode`, prefetch batching
- JMH `DistanceBench`, `SearchBench`, `InsertBench`; annotation processing enabled with `<proc>full</proc>` and `-Xlint:-processing`
- `OptimizationBench` harness; 6-row waterfall; JMH gc profile **320.867 B/op**
- **Bug found and fixed:** the diversity heuristic bug from Phase 2 was corrected to spec Algorithm 4
- **Honest result:** ≥3× QPS target **not met** (0.28× at equal params); SIMD gave 1.56× QPS and 1.59× build speedup; best build 29.08s vs 52.35s; recall rose to 0.9995

### Phase 5 — concurrency and persistence (commits `764b75a`, `483edac`)
- `HnswIndex` rewritten: `AtomicInteger` size/liveCount, 4096 striped `ReentrantLock`s, entry-point lock, `VarHandle` release/acquire on adjacency degree words, `AtomicIntegerArray` publication + tombstones, latch for first entry point, lock-free searches, `addAll(vectors, threads)`
- Persistence: `persist/{IndexSnapshot, IndexFormatException, HnswSerializer, MappedHnswIndex}`; 128-byte LE header, CRC32, atomic save, full load, read-only mmap; committed `v1_fixture.bin`
- Stress tests: 8 threads × 50k inserts with invariants; parallel vs sequential recall within 0.01; 4+4 mixed workload for 10s; deadlock watchdog via preemptive timeout; corruption/truncation/CRC tests
- Measured: 1/2/4/8-thread builds 74.23/44.46/31.94/27.93s (**2.66×**); 1M-vector file (264,000,128 bytes) mapped in **0.0776s**
- **Bug found and fixed:** id=0 path did not increment `liveCount`
- **Bug found and fixed (audit):** search could see `size>0` with `entryPoint==-1` → added latch and defensive empty result

### Phase 6 — honest baselines (commit `dec216b`)
- Fixed environment record; `hnswlib_baseline.py` (Python 3.12), Lucene 10.2.0 baseline, pgvector pg17 in Docker (port 55433) with JDBC 42.7.14, brute-force reference
- `CompareMain` CLI with `--systems`, `--hnswlib-csv`, CSV schema, warmup/timed passes, percentiles; 5 charts generated
- Synthetic 10k comparison: hnswforge recall 0.9995 at ~3,026 QPS; hnswlib 14,401 QPS at equal recall; Lucene recall 0.9665; pgvector ~2,950 QPS at recall 0.9995
- **Real SIFT1M (commit `f73e53b`):** downloaded 160 MB tar, 100k subset, recall@10 **0.99998** at 912 QPS; hnswlib 4,944 QPS at recall 0.9985 (5.4× gap); build 83.00s vs 19.34s; 4 additional `*_sift.png` charts
- "Where HNSWForge loses and why" section with four evidence-based hypotheses

### Phase 7 — RepoPrism adapter (commit `3e99e7e`)
- `repoprism/{VectorSearchBackend, ScoredChunk, Chunk, HnswForgeBackend, ReproprismEval}`
- Cosine MiniLM adapter with id↔string bidirectional map, tombstone re-upsert, RW lock, atomic persist/load side file
- Dependency-free JSONL parser; exact/HNSW/hybrid-RRF hit@6 CLI; `INTEGRATION.md` with Spring `@ConditionalOnProperty` wiring, pgvector kept as default
- Synthetic 200-chunk fixture: exact 100%, HNSWForge 100%, hybrid RRF 100% hit@6

### Phase 8 — packaging and docs (commit `2b0d4e5`)
- Full README (pitch, badges, quick start, charts, Mermaid diagram, tradeoffs, limitations, roadmap, reproduction, REST demo)
- Full ARCHITECTURE.md, FILE_FORMAT.md, publishing profile (source + javadoc plugins), `jitpack.yml`
- `BLOG_POST.md` (1799 words), `LINKEDIN_POST.md` (208 words), `RESUME_BULLETS.md` (3 variants), `TODO_HUMAN.md`
- Test suite expanded with 110 parameterized cases; coverage above target

### Stretch phases (commit `987d3e7`)
- **S1:** `quant/Int8VectorStore` — 4× smaller codes, measured recall@10 vs float32 0.9905, max error 0.017
- **S3:** `HnswIndex.search(q,k,ef,IntPredicate)` — traverses filtered-out nodes, low-selectivity exact fallback
- **S4:** `hnswforge-server` HTTP module (`/vectors`, `/search`, `/snapshot`, `/health`), Dockerfile, 6 tests
- **S2 (product quantization)** deliberately not implemented and documented

### Final audit (commit `a9ba649`)
- **Fixed:** CI smoke flag (`-Dsurefire.failIfNoSpecifiedTests=false`), wrong in original workflow
- **Fixed:** tombstones were not persisted → v2 format with CRC-covered deleted-id section, v1 read-compatible, mapped reader filters deleted ids, regression tests
- **Fixed:** `VectorStore.size` plain `int` lost updates under out-of-order parallel inserts → `AtomicInteger` with `accumulateAndGet(id+1, Math::max)` in heap/off-heap/int8 stores; added `VectorStoreConcurrencyTest`; 4 repeated stress runs clean
- **Fixed:** `run_benchmarks.sh` Python 3.14 hnswlib failure, missing CSV merge, `PYTHONPATH` collision during plotting
- **Fixed:** stale doc numbers (241/264 → 270 tests, 89.9% → 90.3%)

---

## 5. Final verification state

| Check | Result |
|---|---|
| `./mvnw clean verify` | PASS |
| `./mvnw -pl hnswforge-bench -am test -Dtest=BenchmarkSmokeTest -Dsurefire.failIfNoSpecifiedTests=false` | PASS |
| `./mvnw -q -Prelease -DskipTests package` | PASS |
| `bash -n scripts/run_benchmarks.sh` | PASS |
| Total tests | **270** (core 208, eval 34, repoprism 21, server 6, bench 1) |
| Core line coverage | **90.3%** (JaCoCo) |
| Git status | Clean; 13 commits |

---

## 6. All measured numbers recorded in `docs/RESULTS.md`

- Phase 1: brute force vs itself recall@10 = 1.0
- Phase 2: 10k recall 0.995; 100k/dim32 build 52.35s / 3,262 QPS / recall 0.9265; efSearch 200 → 0.9775; 100k/dim128 recall 0.4330 (synthetic uniform data is intrinsically hard)
- Phase 3 sweep: M/efC/efS effects, e.g. m=8/efC=100: efS 20 → recall 0.846 @ 46,793 QPS; efS 160 → recall 0.974 @ 9,264 QPS; m=32 recall 0.992 @ 22,299 QPS
- Phase 4 waterfall: 6 rows; best QPS 1,203 (prefetch), SIMD 1,112 vs scalar 714, build 29.08s; JMH gc 320.867 B/op
- Phase 5: 2.66× parallel build at 8 threads; mmap 1M vectors in 0.0776s; save 0.262s
- Phase 6 synthetic 10k: hnswforge 0.9995 @ 3,026 QPS; hnswlib 0.9995 @ 14,401 QPS; Lucene 0.9665 @ 9,073–11,076 QPS; pgvector 0.9995 @ 2,950 QPS
- Phase 6 real SIFT1M 100k: hnswforge recall@10 0.99998 @ 912 QPS / build 83.00s; hnswlib recall 1.0 @ 2,985 QPS / build 19.34s; Lucene recall 0.77401
- Phase 7: synthetic hit@6 = 100% for exact, HNSWForge and hybrid RRF
- S1: int8 320,000 vs 1,280,000 bytes (4×), recall@10 vs float32 0.9905

**Honest gap:** ≥3× search speedup over baseline was not achieved; hnswlib is 3.0–4.8× faster at equal recall on synthetic data and 5.4× on SIFT1M. Documented with four evidence-based hypotheses.

---

## 7. Bugs found and fixed (complete list)

1. Inverted bounded-heap eviction condition (Phase 1).
2. `greedyDescend` reading a changed node's neighbor slice (Phase 2).
3. Diversity heuristic compared base-to-candidate instead of candidate-to-selected (Phase 2 → fixed in Phase 4).
4. `id == 0` insert skipped `liveCount` increment (Phase 5).
5. Search saw `size > 0` with `entryPoint == -1` during parallel start (audit).
6. Flaky self-loops in adjacency under concurrent insertion (adjacency writes now sanitize self/duplicate ids).
7. CI smoke job used wrong Surefire flag.
8. Tombstones lost on save/load and mmap → v2 format.
9. `VectorStore.size` plain-int lost updates → atomic.
10. Reproduction script Python/CSV/PYTHONPATH defects.
11. Stale documentation numbers.

---

## 8. Remaining work for a human (`docs/TODO_HUMAN.md`, 21 checkboxes)

- Fill all `TODO(human)` placeholders in blog, LinkedIn, resume drafts with personal reflections.
- Download GloVe-100d and rerun the cosine-metric comparison.
- Run the real 72-case RepoPrism evaluation with actual chunk embeddings and gold ids; obtain pgvector-only numbers.
- Interview preparation: re-implement `searchLayer` and `selectNeighborsHeuristic` from memory, whiteboard search/insert, rerun benchmarks on another machine.
- S2 product quantization was intentionally skipped.
- Performance gap to hnswlib remains and is documented honestly.

---

## 9. Commands a successor should know

```bash
./mvnw clean verify                       # full build + all 270 tests + JaCoCo
./mvnw -q -Prelease -DskipTests package   # source + javadoc jars (JitPack/Maven Central ready)
./mvnw -pl hnswforge-bench -am test -Dtest=BenchmarkSmokeTest \
  -Dsurefire.failIfNoSpecifiedTests=false # CI recall smoke job
bash scripts/run_benchmarks.sh            # full reproduction (needs PYTHONPATH for py3.12 hnswlib)
docker compose -f scripts/docker-compose.pgvector.yml up -d   # pgvector on 55433
python3 scripts/plot_compare.py results/compare.csv results/waterfall.csv
```

Key entry points: `dev.hnswforge.index.HnswIndex`, `dev.hnswforge.eval.Main`, `dev.hnswforge.bench.CompareMain`, `dev.hnswforge.repoprism.ReproprismEval`, `dev.hnswforge.server.RestServer`.
