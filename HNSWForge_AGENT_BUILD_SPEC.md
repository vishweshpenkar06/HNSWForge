# HNSWForge: Complete Build Specification for an AI Coding Agent

You are building **HNSWForge**, a dependency-free Java library implementing HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor search, plus evaluation, benchmarking, and integration modules. Follow this document exactly. Work **phase by phase**. Do not start a phase until the previous phase's acceptance criteria pass. After each phase, run the full test suite, commit with a clear message, and print a short status report (what was built, test counts, measured numbers).

---

## 0. Ground rules

1. **Java 25**, Maven multi-module build. Use `--release 25`.
2. `hnswforge-core` must have **zero runtime dependencies**. Test dependencies only: JUnit 5, and jqwik (property tests).
3. Prefer clarity first, then optimize in the optimization phase. Keep a scalar reference implementation of every optimized kernel for testing.
4. Every public class and method has Javadoc stating contract and thread-safety.
5. Deterministic behavior: all randomness comes from an injected `SplittableRandom`/seed in `HnswParams`.
6. Never silently swallow exceptions. Validate inputs (dimension mismatch, NaN, k <= 0) and throw `IllegalArgumentException` with clear messages.
7. Every phase ends with: `mvn -q verify` green, README/docs updated, benchmark numbers recorded in `docs/RESULTS.md` where applicable.
8. Do not copy code from hnswlib or Lucene. Implement from the algorithm described here.
9. Record any design deviation in `docs/DECISIONS.md` (one short entry each: context, decision, reason).

---

## 1. Repository layout

```
hnswforge/
├── pom.xml                         # parent POM (packaging pom)
├── README.md
├── LICENSE                         # Apache-2.0
├── .github/workflows/ci.yml
├── docs/
│   ├── PLAN.md
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md
│   ├── RESULTS.md                  # benchmark tables + chart references
│   ├── FILE_FORMAT.md
│   └── charts/                     # generated PNGs
├── hnswforge-core/
│   └── src/{main,test}/java/dev/hnswforge/...
├── hnswforge-eval/                 # dataset loaders + recall + CLI
├── hnswforge-bench/                # JMH + end-to-end harness + baselines
├── hnswforge-repoprism/            # adapter module (interface + implementation)
├── hnswforge-server/               # (stretch) REST wrapper
├── scripts/
│   ├── download_datasets.sh
│   ├── run_benchmarks.sh
│   ├── hnswlib_baseline.py
│   ├── plot_results.py
│   └── docker-compose.pgvector.yml
└── data/                           # gitignored; datasets live here
```

Base package: `dev.hnswforge`. Sub-packages in core: `distance`, `storage`, `graph`, `index`, `persist`, `exact`, `util`.

---

## 2. Build configuration

### Parent `pom.xml` requirements
- `maven.compiler.release=25`
- Modules: `hnswforge-core`, `hnswforge-eval`, `hnswforge-bench`, `hnswforge-repoprism`, (`hnswforge-server` optional last).
- `dependencyManagement`: JUnit Jupiter 5.x, jqwik 1.x, JMH 1.37.
- Compiler args: `--add-modules jdk.incubator.vector`, `--enable-preview` NOT required unless a used API demands it (verify; if FFM is final in 25, no flag is needed).
- Surefire `argLine`: `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx4g`.
- Enable `-Xlint:all`; fail the build on warnings except the incubator-module warning.
- Plugins: maven-compiler, surefire, failsafe (optional), jacoco (report only), spotless or google-java-format check (optional).

### `.github/workflows/ci.yml`
- Trigger on push and PR. JDK 25 (Temurin). Steps: checkout, setup-java with Maven cache, `mvn -B verify`, upload surefire reports and jacoco report. Add a second job that runs a **small benchmark smoke test** (10k vectors) and fails if recall@10 < 0.95.

---

## 3. Core domain model (`hnswforge-core`)

### 3.1 Enums and records

```java
package dev.hnswforge.index;

public enum Metric { L2, COSINE, INNER_PRODUCT }

public record HnswParams(
    int dimension,       // > 0
    Metric metric,
    int m,               // default 16, range [2, 128]
    int efConstruction,  // default 200, must be >= m
    int efSearch,        // default 64, must be >= 1
    int maxElements,     // capacity; must be > 0 (index preallocates, grows optional)
    long seed            // default 42
) {
    public static Builder builder(int dimension, Metric metric) { ... }
    // validate in canonical constructor; derived values:
    public int maxM0() { return 2 * m; }     // max neighbors at layer 0
    public double levelMult() { return 1.0 / Math.log(m); }
}

public record SearchResult(int id, float distance) {}
```

### 3.2 Public interface

```java
public interface VectorIndex extends AutoCloseable {
    /** Adds a vector, returns the assigned id (0..size-1, insertion order). Thread-safe in HnswIndex. */
    int add(float[] vector);

    /** Returns up to k results sorted by ascending distance. */
    List<SearchResult> search(float[] query, int k);

    /** Same with explicit efSearch override (effective ef = max(efSearch, k)). */
    List<SearchResult> search(float[] query, int k, int efSearch);

    int size();
    int dimension();
    Metric metric();

    void save(Path path) throws IOException;
    @Override void close();
}
```

Implementations: `BruteForceIndex` (package `exact`), `HnswIndex` (package `index`). `HnswIndex.load(Path)` and `HnswIndex.loadMapped(Path)` are static factories.

### 3.3 Distance semantics (important, consistent everywhere)
Internally all metrics are **distances where smaller = closer**:
- `L2`: squared Euclidean distance (no sqrt).
- `INNER_PRODUCT`: `1 - dot(a, b)`... use `-dot(a,b)` internally is NOT allowed (negatives break nothing but confuse tests); use `1 - dot(a, b)`.
- `COSINE`: normalize vectors to unit length **at insert and query time** (store normalized copy; reject zero vectors with `IllegalArgumentException`), then use `1 - dot(a, b)`.
Returned `SearchResult.distance` is the internal distance (document this).

---

## 4. Phase 1: Foundations and exact baseline

### Deliverables
1. `distance/DistanceFunction` interface:
   ```java
   public interface DistanceFunction {
       float distance(float[] a, float[] b);
       // for later phases:
       float distance(float[] a, MemorySegment seg, long byteOffset, int dim);
   }
   ```
2. `distance/ScalarDistance` with `L2`, `InnerProduct`, `Cosine` (cosine = normalize + inner product) implementations. Plain loops, no unrolling tricks yet.
3. `exact/BruteForceIndex implements VectorIndex`: stores `float[][]` (or a flat array), linear scan, uses a bounded max-heap (size k) of `(id, dist)`. Not thread-safe for writes, safe for concurrent reads after build. `save` may throw `UnsupportedOperationException`.
4. `util/BoundedMaxHeap` and `util/MinHeap` backed by primitive arrays (parallel `int[] ids` and `float[] dists`). No boxing. API: `push`, `pop`, `peek`, `size`, `clear`. Needed repeatedly in later phases.
5. `hnswforge-eval` module:
   - `FvecsReader` / `IvecsReader` / `BvecsReader` (TEXMEX formats: each vector = int32 little-endian dim, then dim values).
   - `GloveReader` (text, `word f1 f2 ... fn`), with optional limit.
   - `Recall.recallAtK(int[][] groundTruth, List<List<SearchResult>> results, int k)` returns mean recall in [0,1].
   - `GroundTruth.compute(float[][] base, float[][] queries, Metric metric, int k)` using `BruteForceIndex`, parallelized with parallel streams, and writer/reader for `.ivecs` caching.
   - CLI main `dev.hnswforge.eval.Main` with subcommands: `groundtruth`, `recall`.
6. `scripts/download_datasets.sh`: downloads SIFT1M (`ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`) and GloVe-100d 1.2M from a stated URL into `data/` (idempotent, checksum optional). Document that the script may need updating if URLs change.

### Tests (Phase 1)
- Distance: known small vectors with hand-computed expected values; symmetric; `distance(a,a)==0` for L2; cosine invariant to scaling; zero-vector cosine throws.
- Heaps: random operations vs `PriorityQueue` oracle (jqwik property test).
- BruteForce: k > size returns size results; k == 1; ties deterministic by lower id; dimension mismatch throws; results sorted ascending.
- Readers: round-trip tiny generated fvecs/ivecs files.
- Recall: perfect results -> 1.0; empty overlap -> 0.0; partial overlap exact fraction.

### Acceptance
- `mvn verify` green, >= 40 tests.
- On a 10k subset of SIFT, brute force recall evaluated against its own ground truth = 1.0.

---

## 5. Phase 2: Naive HNSW (correctness first)

Implement in `index/HnswIndex` using straightforward Java structures: `float[][] vectors`, per-node `int[][] neighbors[layer]` (object-based is fine in this phase; flat layouts come in Phase 4). Single-threaded; guard with a coarse `ReentrantReadWriteLock` (readers = searches, writer = add) so it is safe, just not fast.

### 5.1 Data per node
- `level[node]` (int): highest layer the node appears in.
- `neighbors[node][layer]`: `int[]` + count. Max degree: `maxM0 = 2*m` at layer 0, `m` at layers >= 1.
- Global: `entryPoint` (int, -1 when empty), `maxLevel` (int), `size`.

### 5.2 Random level generation
```
level = (int) floor(-ln(U) * levelMult), U ~ Uniform(0,1], using seeded RNG
```
Cap at a safe maximum (e.g. 32).

### 5.3 Core routines (implement exactly)

**searchLayer(query, entryPoints, ef, layer)** returns up to `ef` closest nodes found, sorted ascending by distance.
```
visited = new visited-set containing entryPoints
candidates = MinHeap (closest first)     // frontier to expand
results    = BoundedMaxHeap(ef)          // best found, farthest on top
for each ep: d = dist(query, ep); candidates.push(ep,d); results.push(ep,d)
while candidates not empty:
    (c, dc) = candidates.pop()
    if dc > results.peekDistance() and results.size() == ef: break
    for each neighbor n of c at layer:
        if n in visited: continue
        visited.add(n)
        dn = dist(query, n)
        if results.size() < ef or dn < results.peekDistance():
            candidates.push(n, dn)
            results.push(n, dn)     // evicts farthest if over ef
return results sorted ascending
```

**greedyDescend(query, entry, fromLayer, toLayer)**: for each layer from `fromLayer` down to `toLayer` (exclusive of the layer where full search occurs), repeatedly move to the neighbor that is closer to the query until no improvement.

**selectNeighborsSimple(candidates, M)**: take the M closest.

**selectNeighborsHeuristic(base, candidates, M)** (the diversity heuristic, Algorithm 4 in the paper):
```
sort candidates ascending by distance to base
selected = []
for (c, dc) in candidates:
    if selected.size() == M: break
    good = true
    for s in selected:
        if dist(c, s) < dc:     // c is closer to an already-selected neighbor than to base
            good = false; break
    if good: selected.add(c)
// optional keepPrunedConnections: fill remaining slots with discarded candidates (make this a param, default false... but test both)
return selected
```
Make the strategy a switch in `HnswParams` (`NeighborSelection.SIMPLE | HEURISTIC`, default HEURISTIC) so benchmarks can compare.

**add(vector)**
```
id = size++; level = randomLevel(); store vector
if id == 0: entryPoint = 0; maxLevel = level; return
ep = [entryPoint]
for layer = maxLevel down to level+1: ep = greedyDescend(vector, ep, layer)   // ef=1
for layer = min(level, maxLevel) down to 0:
    W = searchLayer(vector, ep, efConstruction, layer)
    M_layer = (layer == 0) ? maxM0 : m
    neigh = selectNeighbors(id, W, m)             // connect to m (not maxM0) at insert time
    set neighbors[id][layer] = neigh
    for each n in neigh:                          // bidirectional
        add id to n's list at layer
        if n's degree > M_layer: shrink using selectNeighbors(n, current list + id, M_layer)
    ep = W
if level > maxLevel: entryPoint = id; maxLevel = level
```

**search(query, k, ef)**
```
if empty: return []
ep = greedy descend from maxLevel to layer 1 with ef=1
W = searchLayer(query, [ep], max(ef,k), 0)
return first k of W
```

### 5.4 Visited set
Phase 2 may use a `BitSet`/`boolean[]`. Phase 4 replaces it with an epoch-stamped visited list (`int[] stamp` + current epoch, reset by incrementing epoch).

### Tests (Phase 2)
- Graph invariants after building 5k random vectors: every node's degree <= bound; no self-loops; no duplicate neighbors; all neighbor ids valid; each non-entry node reachable from entry at layer 0 (BFS).
- Level distribution: histogram roughly geometric (loose statistical test with fixed seed).
- Determinism: same seed + same insertion order -> identical graph (compare adjacency).
- Recall on 10k SIFT subset (or synthetic Gaussian clusters when dataset absent): recall@10 >= 0.95 with m=16, efConstruction=200, efSearch=100.
- Edge cases: empty index search, single element, k > size, duplicate vectors, dimension mismatch, NaN rejected.
- Heuristic vs simple selection: on clustered data, heuristic recall >= simple recall (assert with tolerance).
- Property test: recall never exceeds 1.0; results sorted; ids unique.

### Acceptance
- >= 70 total tests passing; recall@10 >= 0.95 on 100k SIFT subset at efSearch <= 200 (document actual numbers in `docs/RESULTS.md`, section "Phase 2 baseline").
- Build and search latency for the naive version recorded as the **baseline for the optimization waterfall**.

---

## 6. Phase 3: Parameter study and tuning harness

### Deliverables
1. `hnswforge-eval` CLI subcommand `sweep`:
   `--dataset sift1m|glove100|synthetic --n 100000 --m 8,16,32 --efc 100,200,400 --efs 10,20,40,80,160,320 --k 10 --threads 1 --out results.csv`
   Output CSV columns: `dataset,n,dim,metric,m,efc,efs,k,recall,qps,p50_us,p95_us,p99_us,build_s,index_bytes_est`.
2. `scripts/plot_results.py` (matplotlib): reads CSV, produces `docs/charts/recall_vs_qps_<dataset>.png` with one line per (m, efc) combo (x = recall, y = QPS, log y). Include requirements.txt.
3. Optional: **tombstone deletes**: `boolean delete(int id)` marks deleted; searches skip deleted ids in results (still traverse them); `size()` vs `liveCount()`; document limitations in `ARCHITECTURE.md`.

### Tests
- Sweep on a tiny synthetic dataset runs end-to-end and writes a valid CSV.
- Deleted ids never appear in results; recall on remaining set stays >= 0.9 after deleting 10%.

### Acceptance
- First recall vs QPS charts committed. Short written analysis (5-10 lines) in `docs/RESULTS.md` explaining the effect of `M`, `efC`, `efS`.

---

## 7. Phase 4: Performance engineering (the headline phase)

Perform these optimizations **one at a time**, in order. After each, run the standard benchmark (SIFT1M 100k subset at fixed params: m=16, efC=200, efS=100, k=10, 1 thread) and append a row to the **optimization waterfall table** in `docs/RESULTS.md` (build time, QPS, recall, allocations/sec if measured). Recall must not drop by more than 0.002 versus the previous step; if it does, you introduced a bug.

### 4.1 Flat vector storage
Replace `float[][]` with a single contiguous `float[]` (or `MemorySegment` later) of size `maxElements * dimension`. Access vector i at offset `i * dimension`. Provide `VectorStore` interface with `float[] copyOf(int id)` and distance methods operating directly on the backing array with an offset (`dist(float[] q, int id)` and `dist(int a, int b)`).

### 4.2 Flat adjacency
Layer 0: one `int[] l0` of size `maxElements * (1 + maxM0)`; for node i, slot `i*(1+maxM0)` = degree, followed by neighbor ids. Upper layers: per-node `int[]` allocated lazily only for nodes with level >= 1, stored in `int[][] upper` indexed by node (null if level 0) with each entry sized `level * (1 + m)` (layer L occupies slice `(L-1)*(1+m)`).

### 4.3 Epoch-stamped visited list
`VisitedList { int[] stamp; int epoch; }` pooled per thread via `ThreadLocal` (or a bounded object pool). `reset()` = `epoch++` with full clear on overflow.

### 4.4 Primitive heaps with no allocation
Reuse per-thread scratch buffers for candidates/results heaps and temp arrays (`ThreadLocal<SearchScratch>`). Ensure steady-state search performs **zero allocations** aside from the final result list. Verify with a JMH `-prof gc` run (target: < 1 KB/op excluding output).

### 4.5 SIMD distance kernels
Implement `distance/SimdDistance` using `jdk.incubator.vector.FloatVector` with `FloatVector.SPECIES_PREFERRED`:
```
loop over dim in steps of species.length(): acc = a.fma(b_diff...) style
reduce with acc.reduceLanes(VectorOperators.ADD); scalar tail for remainder
```
- Provide L2 and dot variants (and cosine via dot on normalized data).
- `DistanceFunctions.forMetric(Metric, boolean preferSimd)` returns the best available; detect Vector API availability at class-load time and fall back to scalar with a logged warning.
- Property test: SIMD vs scalar within relative tolerance 1e-4 for random vectors of dims {1, 3, 7, 16, 100, 128, 960}.

### 4.6 Off-heap vector storage (Foreign Function & Memory API)
Implement `OffHeapVectorStore` using `Arena.ofShared()` and one big `MemorySegment` (float layout, aligned to 64 bytes). Distance kernels read via `FloatVector.fromMemorySegment(species, segment, byteOffset, ByteOrder.nativeOrder())`. Provide a switch (`StorageMode.HEAP | OFF_HEAP`) and compare in the waterfall. `close()` releases the arena; guard use-after-close.

### 4.7 Prefetch-friendly traversal (best effort)
Batch neighbor processing: first collect unvisited neighbors of the expanded node into a small int buffer, mark visited, then compute distances in a tight loop. Document the effect even if small.

### 4.8 Profiling artifacts
- `hnswforge-bench` JMH benchmarks: `DistanceBench` (scalar vs SIMD by dimension), `SearchBench` (QPS at fixed recall settings), `InsertBench`.
- Document how to run async-profiler and include one flame graph screenshot in `docs/charts/`.

### Acceptance
- Waterfall table with >= 6 rows. Cumulative search speedup >= 3x over the Phase 2 baseline (record the actual number; if lower, explain honestly in RESULTS.md).
- Zero-allocation steady-state search demonstrated by a JMH gc profile result in docs.

---

## 8. Phase 5: Concurrency and persistence

### 8.1 Concurrent insertion
Replace the coarse lock:
- `ReentrantLock[] nodeLocks` striped (e.g. 4096 stripes, lock = `locks[id & (stripes-1)]`) guarding a node's neighbor list modifications.
- A global `entryPointLock` (or `StampedLock`) for entry point / maxLevel updates when a new node exceeds the current max level.
- `size` managed with `AtomicInteger`; `add` reserves the id atomically, then initializes the vector and publishes it (use `VarHandle` release/acquire or `volatile` publication of a per-node "ready" flag so searchers never see half-written vectors).
- **Reads (search) take no locks.** Neighbor list reads must tolerate concurrent modification: read degree then neighbors with consistent snapshotting (e.g. copy the list slice under no lock but validate ids < size and skip invalid/unready nodes). Document the memory-ordering reasoning in `ARCHITECTURE.md`.
- Lock ordering rule: when locking two nodes (neighbor shrink), lock in ascending id order to avoid deadlock; never hold a node lock while acquiring the entry-point lock.
- Offer `addAll(float[][] vectors, int threads)` that inserts in parallel with a configurable thread count.

### 8.2 Persistence
Binary format, little-endian, specified in `docs/FILE_FORMAT.md`:
```
Header (fixed 128 bytes):
  magic        8 bytes   "HNSWFRG\0"
  version      int32     = 1
  endianness   int32     = 1 (little)
  dimension    int32
  metric       int32     (0=L2,1=COSINE,2=IP)
  m            int32
  efConstr     int32
  efSearch     int32
  maxElements  int32
  size         int32     (actual count)
  entryPoint   int32
  maxLevel     int32
  seed         int64
  vectorsOff   int64     (byte offset, 64-aligned)
  levelsOff    int64
  layer0Off    int64
  upperOff     int64
  upperLen     int64
  crc32        int32     (over everything after header)
  reserved     zero padding to 128 bytes
Sections:
  vectors : size * dimension floats
  levels  : size int32
  layer0  : size * (1 + maxM0) int32
  upper   : for each node with level >= 1: level * (1+m) int32, in node order (nodes with level 0 contribute nothing; offsets derivable from levels)
```
- `save(Path)`: write to a temp file then atomic rename.
- `HnswIndex.load(Path)`: read fully into heap/off-heap structures; verify magic, version, CRC; clear errors on mismatch (`IndexFormatException`).
- `HnswIndex.loadMapped(Path)`: memory-map the file read-only; vector and layer-0 sections accessed directly through `MemorySegment`; the mapped index is **read-only** (`add` throws `UnsupportedOperationException`).

### Tests (Phase 5)
- Concurrency: 8 threads insert 50k vectors concurrently; afterwards graph invariants hold, size == 50k, recall@10 within 0.01 of single-threaded build.
- Concurrent mixed workload: 4 inserting + 4 searching threads for 10 seconds; no exceptions, no results with invalid ids, no duplicate ids in a result list.
- Deadlock detection: run with a watchdog thread; test fails if not complete within timeout.
- Persistence: round-trip -> identical search results for 1000 random queries; truncated file, flipped byte, wrong magic, wrong version all raise `IndexFormatException`; mapped load returns identical results; mapped index rejects `add`.
- Backward compat: commit one small `v1` fixture file in test resources that must always load.

### Acceptance
- Parallel build speedup measured at 1, 2, 4, 8 threads (table in RESULTS.md).
- Load time of 1M-vector index via `loadMapped` recorded (should be near-instant versus full load).

---

## 9. Phase 6: Honest benchmarking against baselines

### 9.1 Fixed environment
Record in `docs/RESULTS.md`: CPU model, core count, RAM, OS, JDK version and flags, dataset versions. Use identical thread counts, `k=10`, and the same query set for all systems. Pin JVM: `-Xms4g -Xmx4g -XX:+UseParallelGC` (or document chosen GC).

### 9.2 Baselines
1. **hnswlib** via `scripts/hnswlib_baseline.py` (`pip install hnswlib numpy`): builds with identical `M`, `efConstruction`; sweeps `efSearch`; outputs the same CSV schema. Single-thread query loop for fairness.
2. **Lucene HNSW**: in `hnswforge-bench`, add `lucene-core` dependency (bench module only). Implement `LuceneHnswBaseline` using `KnnFloatVectorField` + `KnnFloatVectorQuery`, matching `M` (`maxConn`) and `beamWidth` (= efConstruction). Force single segment (`forceMerge(1)`) before querying.
3. **pgvector**: `scripts/docker-compose.pgvector.yml` (image `pgvector/pgvector:pg17` or current), table with `vector(dim)`, `CREATE INDEX ... USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200)`, query with `SET hnsw.ef_search = X`. Java harness `PgvectorBaseline` over JDBC (bench-only dependency). Note network/driver overhead honestly; additionally report in-DB `EXPLAIN ANALYZE` timings if feasible.
4. **Brute force** (exact) as a sanity reference.

### 9.3 Metrics (every system, every setting)
Recall@10, QPS (single thread) and QPS (all threads), p50/p95/p99 latency, build time (single and multi-thread), index size on disk, peak RSS or heap+off-heap estimate.

### 9.4 Harness
`hnswforge-bench` main class `dev.hnswforge.bench.CompareMain`:
`--dataset sift1m --systems hnswforge,lucene,pgvector,hnswlib(csv) --efs 10,20,40,80,160,320 --out results/compare.csv`.
Warm up each system (>= 3 passes over queries) before timing; report median of 5 timed runs.

### 9.5 Charts (generated by `plot_results.py`)
1. Recall@10 vs QPS, all systems, per dataset (log-y).
2. Build time vs dataset size (10k, 100k, 1M).
3. Memory per 1M vectors.
4. Latency percentiles at recall ~ 0.95 and 0.99.
5. Optimization waterfall bar chart from Phase 4.

### 9.6 Reporting rules
`docs/RESULTS.md` must include a "Where HNSWForge loses and why" section with at least 3 concrete, evidence-based hypotheses (for example: no prefetch intrinsics, JIT warmup, bounds checks, GC settings). Never cherry-pick settings. If HNSWForge is within 3x of hnswlib at equal recall, state it plainly; if not, state the real gap.

---

## 10. Phase 7: RepoPrism integration (`hnswforge-repoprism`)

The RepoPrism codebase is a separate Spring Boot project (Java, pgvector, ONNX MiniLM embeddings, hybrid retrieval with RRF). This module provides a clean adapter, and instructions for wiring it in.

### Deliverables
1. In this module define (or mirror, if RepoPrism already has an equivalent) the abstraction:
   ```java
   public interface VectorSearchBackend {
       void upsert(String chunkId, float[] embedding);
       List<ScoredChunk> search(float[] queryEmbedding, int k);
       void rebuild(Iterable<Chunk> chunks);   // full reindex
       void persist();                         // save snapshot
   }
   public record ScoredChunk(String chunkId, float score) {}
   ```
2. `HnswForgeBackend implements VectorSearchBackend`: maps `chunkId <-> int id` (bidirectional map persisted next to the index file as a small side file), uses cosine metric for MiniLM (384-dim), thread-safe, snapshot save/load on startup/shutdown.
3. `INTEGRATION.md` in this module: step-by-step instructions for the human to add the dependency to RepoPrism, add a Spring `@ConditionalOnProperty(name="retrieval.backend", havingValue="hnswforge")` configuration, keep `pgvector` as the default, and keep the existing RRF fusion unchanged (HNSWForge replaces only the dense/vector leg).
4. A standalone **evaluation script/class** that, given a JSON/CSV export of RepoPrism chunk embeddings plus the 72-case eval queries and gold chunk ids, computes hit@6 for: pgvector-only, HNSWForge-only, and hybrid with each backend, and prints a before/after table. Because the agent may not have RepoPrism data, build this against a synthetic corpus fixture in tests, and make the real-data path a documented CLI: `java -jar ... repoprism-eval --embeddings chunks.jsonl --queries eval72.jsonl --gold gold.jsonl`.

### Tests
- Upsert/search round-trip with string ids; unknown ids; re-upsert updates vector (implement via tombstone + reinsert, and document).
- Persist and restore preserves id mapping and results.
- Concurrent upserts and searches are safe.

### Acceptance
- Adapter tests pass; INTEGRATION.md reviewed for completeness; synthetic eval shows HNSWForge hit@6 within 0.5 percentage points of exact search at the default settings.

---

## 11. Phase 8: Packaging, docs, release

1. **README.md** with: one-line pitch, badges (CI, license), 10-line quick start code sample, feature list, benchmark chart embeds with exact setup, architecture diagram (Mermaid), design tradeoffs, limitations (no distributed mode, mapped index read-only, deletes via tombstones), roadmap, reproduction instructions (`scripts/run_benchmarks.sh` runs everything), contributing, license.
2. **docs/ARCHITECTURE.md**: components diagram (Mermaid), data layouts (vector store, adjacency), search and insert flow diagrams, concurrency model with memory-ordering argument, persistence format summary, and a "How HNSW works in 1 page" primer written for newcomers.
3. **Publishing**: configure `maven-source-plugin`, `maven-javadoc-plugin`, and a `release` profile (GPG sign optional) so the artifact can be published via JitPack (tag-based) immediately and Maven Central later. Add `jitpack.yml` specifying JDK 25 and the incubator-module flag note.
4. **Blog post draft** `docs/BLOG_POST.md` (1200-1800 words): problem, what HNSW is in plain language, three biggest surprises during implementation, the optimization waterfall, honest benchmark results including losses, and what you would do next. Leave clearly marked `TODO(human)` placeholders for personal reflections.
5. **LinkedIn post draft** `docs/LINKEDIN_POST.md` (under 250 words) with the best chart and one headline number.
6. **Resume bullets** `docs/RESUME_BULLETS.md`: three variants, with measured numbers filled in from `RESULTS.md` (do not invent numbers; use placeholders if not measured).

---

## 12. Stretch phases (only after Phase 8 is complete and green)

### S1: int8 scalar quantization
Per-index or per-dimension min/max calibration; store `byte[]` codes (4x smaller); asymmetric distance (float query vs int8 base) with SIMD; optional re-rank of top `efSearch` candidates with full-precision vectors. Report recall/QPS/memory tradeoff vs float32.

### S2: Product quantization (optional, advanced)
k-means codebooks (8 subvectors x 256 centroids), ADC lookup tables, SIMD-friendly layout. Report recall/memory.

### S3: Filtered search
`search(float[] q, int k, int ef, IntPredicate filter)`; traverse through filtered-out nodes but only return/admit matching ones; handle low-selectivity filters by falling back to brute-force over matching ids when selectivity < configurable threshold. Test with 1%, 10%, 50% selectivity.

### S4: REST server (`hnswforge-server`)
Minimal HTTP server (JDK `com.sun.net.httpserver` or a tiny framework): `POST /vectors`, `POST /search`, `POST /snapshot`, `GET /health`, with a Dockerfile and a 5-line curl demo in the README.

---

## 13. Complete test inventory (target >= 200 tests by the end)

| Area | Count target | Key cases |
|---|---|---|
| Distance (scalar + SIMD) | 20 | dims 1..960, tails, normalization, zero vectors |
| Heaps/util | 15 | oracle vs PriorityQueue, property tests |
| BruteForce | 15 | ties, k edge cases, sorted |
| Eval/readers/recall | 15 | format round trips |
| HNSW graph invariants | 25 | degree bounds, reachability, determinism |
| HNSW recall | 20 | per metric, per selection strategy, per dataset |
| Storage (heap/off-heap) | 15 | alignment, close semantics, bounds |
| Concurrency | 20 | parallel build, mixed workload, watchdog |
| Persistence | 25 | round trip, corruption, mapped, fixture |
| Deletes | 10 | tombstones, recall after deletes |
| Adapter (RepoPrism) | 15 | id mapping, persist, upsert semantics |
| Stretch (if built) | 15 | quantization, filters |

Use jqwik for randomized properties (fixed seeds on failure reproduction). Keep unit tests fast (< 2 min total); mark heavy tests with `@Tag("slow")` and run them in a separate CI job or locally.

---

## 14. Definition of done (checklist for the agent)

- [ ] All modules build with `mvn -B verify` on JDK 25
- [ ] >= 200 tests passing, jacoco line coverage >= 80% on core
- [ ] Recall@10 >= 0.95 on SIFT1M at documented parameters
- [ ] Optimization waterfall with >= 6 steps and honest numbers in `docs/RESULTS.md`
- [ ] Concurrent build and mapped load work and are tested
- [ ] Comparison charts vs hnswlib, Lucene, pgvector (or documented reason if one could not be run)
- [ ] RepoPrism adapter + INTEGRATION.md + synthetic eval
- [ ] README, ARCHITECTURE, FILE_FORMAT, DECISIONS, RESULTS complete
- [ ] CI workflow green, including the benchmark smoke job
- [ ] Blog, LinkedIn, and resume drafts created with measured numbers or clear placeholders
- [ ] No TODOs left in code except `TODO(human)` markers listed in a single `docs/TODO_HUMAN.md`

---

## 15. Instructions for the human (so you can defend this in interviews)

The agent can write the code, but you must be able to explain it. Before applying anywhere:
1. Read the HNSW paper and `docs/ARCHITECTURE.md` and explain search and insert on a whiteboard.
2. Re-implement `searchLayer` and `selectNeighborsHeuristic` yourself once from memory, and compare to the agent's version.
3. Rerun the benchmarks on your own machine and understand each chart.
4. Be able to answer: why layers, what `M`/`efC`/`efS` do, why the heuristic helps, how concurrent insert stays safe, why SIMD helped and where it stopped helping, and where you lose to hnswlib and why.
5. Fill in every `TODO(human)` marker with your own words.

---

## 16. How to run this with an agent (prompt template)

Paste this at the start of the agent session, then feed phases one at a time:

```
You are implementing the project described in HNSWForge_AGENT_BUILD_SPEC.md (attached).
Rules: follow the spec exactly, work one phase at a time, write tests first or alongside code,
run `mvn -q verify` before declaring a phase done, commit after each phase, and print a status
report with test counts and measured numbers. Do not invent benchmark numbers; only report
what you measured. Ask me before deviating from the spec; record approved deviations in
docs/DECISIONS.md. Start with Phase 1 now.
```

Then for each next phase: `Proceed to Phase N. Re-read its section of the spec first.`
