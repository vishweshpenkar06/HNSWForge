package dev.hnswforge.index;

import dev.hnswforge.distance.DistanceFunctions;
import dev.hnswforge.persist.HnswSerializer;
import dev.hnswforge.persist.IndexSnapshot;
import dev.hnswforge.persist.MappedHnswIndex;
import dev.hnswforge.storage.HeapVectorStore;
import dev.hnswforge.storage.OffHeapVectorStore;
import dev.hnswforge.storage.StorageMode;
import dev.hnswforge.storage.VectorStore;
import dev.hnswforge.util.SearchScratch;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntPredicate;

/**
 * Concurrent HNSW index with flat storage and disk persistence.
 *
 * <p>Inserts reserve ids atomically, publish each node with a release store after
 * its vector and edges are complete, and guard neighbor-list writes with striped
 * locks. Searches take no locks: they acquire-read degree words, validate neighbor
 * ids against the current size and skip unpublished nodes. Entry point updates use
 * a dedicated lock. Lock ordering is trivial because at most one node lock is held
 * at a time; the entry-point lock is never acquired while holding a node lock.
 *
 * <p>Vectors are stored contiguously on heap or off-heap; layer 0 uses one flat
 * {@code int[]}, upper layers are allocated lazily. All randomness derives from
 * the seed in {@link HnswParams}. Deletes are tombstones.
 *
 * <p>Thread-safety: thread-safe for concurrent inserts, deletes and searches.
 */
public final class HnswIndex implements VectorIndex {
    private static final int MAX_LEVEL_CAP = 32;
    private static final int LOCK_STRIPES = 4096;
    private static final int LOCK_MASK = LOCK_STRIPES - 1;
    private static final VarHandle INT_HANDLE = MethodHandles.arrayElementVarHandle(int[].class);

    private final HnswParams params;
    private final VectorStore store;
    private final StorageMode storageMode;
    private final boolean prefetch;
    private final SplittableRandom random;
    private final Object randomLock = new Object();
    private final ThreadLocal<SearchScratch> scratch;
    private final ReentrantLock[] nodeLocks = new ReentrantLock[LOCK_STRIPES];
    private final ReentrantLock entryPointLock = new ReentrantLock();
    private final AtomicInteger size = new AtomicInteger();
    private final AtomicInteger liveCount = new AtomicInteger();
    private final AtomicIntegerArray published;
    private final AtomicIntegerArray tombstones;
    private final CountDownLatch firstPublished = new CountDownLatch(1);

    private final int[] levels;
    private final int[] layer0;
    private final int[][] upper;
    private final int layer0Stride;
    private final int upperStride;
    private volatile int entryPoint = -1;
    private volatile int maxLevel = -1;

    /** Creates an index with SIMD kernels, on-heap storage and prefetch batching. */
    public HnswIndex(HnswParams params) {
        this(params, true, StorageMode.HEAP, true);
    }

    /**
     * Creates an index with explicit kernel and storage choices and prefetch batching.
     *
     * @param params      construction parameters
     * @param preferSimd  whether to use the Vector API when available
     * @param storageMode heap or off-heap vector storage
     */
    public HnswIndex(HnswParams params, boolean preferSimd, StorageMode storageMode) {
        this(params, preferSimd, storageMode, true);
    }

    /**
     * Creates an index with explicit kernel, storage and traversal choices.
     *
     * @param params      construction parameters
     * @param preferSimd  whether to use the Vector API when available
     * @param storageMode heap or off-heap vector storage
     * @param prefetch    whether to batch neighbor discovery before computing distances
     */
    public HnswIndex(HnswParams params, boolean preferSimd, StorageMode storageMode,
                     boolean prefetch) {
        this.params = Objects.requireNonNull(params, "params must not be null");
        this.storageMode = Objects.requireNonNull(storageMode, "storageMode must not be null");
        this.prefetch = prefetch;
        var kernel = DistanceFunctions.forMetric(params.metric(), preferSimd);
        this.store = switch (storageMode) {
            case HEAP -> new HeapVectorStore(params.dimension(), params.maxElements(), kernel);
            case OFF_HEAP -> new OffHeapVectorStore(params.dimension(), params.maxElements(), kernel);
        };
        this.random = new SplittableRandom(params.seed());
        int capacity = params.maxElements();
        this.levels = new int[capacity];
        this.layer0Stride = 1 + params.maxM0();
        this.upperStride = 1 + params.m();
        this.layer0 = new int[Math.multiplyExact(capacity, layer0Stride)];
        this.upper = new int[capacity][];
        this.published = new AtomicIntegerArray(capacity);
        this.tombstones = new AtomicIntegerArray(capacity);
        for (int i = 0; i < LOCK_STRIPES; i++) {
            nodeLocks[i] = new ReentrantLock();
        }
        this.scratch = ThreadLocal.withInitial(() -> new SearchScratch(
                params.dimension(), capacity,
                Math.max(params.efConstruction(), params.efSearch()), params.maxM0()));
    }

    @Override
    public int add(float[] vector) {
        SearchScratch local = scratch.get();
        float[] prepared = prepare(vector, local.query);
        int id = size.getAndIncrement();
        if (id >= params.maxElements()) {
            throw new IllegalStateException(
                    "index capacity exceeded: " + params.maxElements() + " elements");
        }
        try {
            store.set(id, prepared);
            int level = randomLevel();
            levels[id] = level;
            if (level > 0) {
                upper[id] = new int[level * upperStride];
            }
            published.set(id, 1);

            if (id == 0) {
                entryPointLock.lock();
                try {
                    entryPoint = 0;
                    maxLevel = level;
                } finally {
                    entryPointLock.unlock();
                }
                liveCount.incrementAndGet();
                firstPublished.countDown();
                return id;
            }

            try {
                firstPublished.await();
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for the first node",
                        exception);
            }
            int[] entryPoints = {entryPoint};
            for (int layer = maxLevel; layer > level; layer--) {
                entryPoints = new int[] {greedyDescend(prepared, entryPoints[0], layer, layer - 1)};
            }

            for (int layer = Math.min(level, maxLevel); layer >= 0; layer--) {
                int found = searchLayer(prepared, entryPoints, entryPoints.length,
                        params.efConstruction(), layer, local);
                int[] foundIds = Arrays.copyOf(local.foundIds, found);
                float[] foundDistances = Arrays.copyOf(local.foundDistances, found);
                int chosenCount = selectByStrategy(id, foundIds, foundDistances, found,
                        params.m(), local.chosen);
                int[] chosen = Arrays.copyOf(local.chosen, chosenCount);
                int bound = layer == 0 ? params.maxM0() : params.m();

                ReentrantLock selfLock = nodeLocks[id & LOCK_MASK];
                selfLock.lock();
                try {
                    setNeighbors(id, layer, chosen, chosenCount);
                } finally {
                    selfLock.unlock();
                }

                for (int neighbor : chosen) {
                    ReentrantLock lock = nodeLocks[neighbor & LOCK_MASK];
                    lock.lock();
                    try {
                        int count = degree(neighbor, layer);
                        if (count < bound && !hasNeighbor(neighbor, layer, id)) {
                            setNeighborAt(neighbor, layer, count, id);
                            setDegree(neighbor, layer, count + 1);
                        } else if (count >= bound) {
                            int combinedCount = gatherWithExtra(neighbor, layer, id, local);
                            sortCandidates(local.combinedIds, local.combinedDistances, combinedCount);
                            int prunedCount = selectByStrategy(neighbor, local.combinedIds,
                                    local.combinedDistances, combinedCount, bound, local.chosen);
                            setNeighbors(neighbor, layer, local.chosen, prunedCount);
                        }
                    } finally {
                        lock.unlock();
                    }
                }
                entryPoints = foundIds;
            }

            if (level > maxLevel) {
                entryPointLock.lock();
                try {
                    if (level > maxLevel) {
                        entryPoint = id;
                        maxLevel = level;
                    }
                } finally {
                    entryPointLock.unlock();
                }
            }
            liveCount.incrementAndGet();
            return id;
        } catch (RuntimeException | Error failure) {
            tombstones.set(id, 1);
            throw failure;
        }
    }

    /**
     * Inserts all vectors in parallel using a fixed number of worker threads.
     *
     * @param vectors array of vectors to add
     * @param threads number of worker threads, must be positive
     */
    public void addAll(float[][] vectors, int threads) {
        Objects.requireNonNull(vectors, "vectors must not be null");
        if (threads <= 0) {
            throw new IllegalArgumentException("threads must be > 0, got " + threads);
        }
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int worker = 0; worker < threads; worker++) {
                final int first = worker;
                futures.add(pool.submit(() -> {
                    for (int i = first; i < vectors.length; i += threads) {
                        add(vectors[i]);
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("parallel insertion interrupted", exception);
        } catch (java.util.concurrent.ExecutionException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof RuntimeException runtime) {
                throw runtime;
            }
            if (cause instanceof Error error) {
                throw error;
            }
            throw new IllegalStateException("parallel insertion failed", cause);
        } finally {
            pool.shutdown();
            try {
                pool.awaitTermination(1, TimeUnit.MINUTES);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Override
    public List<SearchResult> search(float[] query, int k) {
        return search(query, k, params.efSearch());
    }

    @Override
    public List<SearchResult> search(float[] query, int k, int efSearch) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        if (efSearch < 1) {
            throw new IllegalArgumentException("efSearch must be >= 1, got " + efSearch);
        }
        SearchScratch local = scratch.get();
        float[] prepared = prepare(query, local.query);
        int currentSize = size.get();
        int start = entryPoint;
        if (currentSize == 0 || start < 0) {
            return List.of();
        }
        int entry = greedyDescend(prepared, start, maxLevel, 1);
        local.entryBuffer[0] = entry;
        int found = searchLayer(prepared, local.entryBuffer, 1, Math.max(efSearch, k), 0, local);
        List<SearchResult> results = new ArrayList<>(Math.min(k, found));
        for (int i = 0; i < found && results.size() < k; i++) {
            int id = local.foundIds[i];
            if (tombstones.get(id) == 0) {
                results.add(new SearchResult(id, local.foundDistances[i]));
            }
        }
        return results;
    }

    /**
     * Searches with a metadata filter. Filtered-out nodes are still traversed but
     * are never admitted to the result set. When the filter selects few nodes,
     * the search falls back to an exact scan over the matching ids.
     *
     * @param query  query vector
     * @param k      number of results, must be positive
     * @param ef     search effort, must be positive
     * @param filter predicate over internal ids, must not be null
     * @return up to {@code k} matching results sorted by ascending distance
     * @throws IllegalArgumentException if inputs are invalid
     */
    public List<SearchResult> search(float[] query, int k, int ef, IntPredicate filter) {
        if (k <= 0 || ef < 1) {
            throw new IllegalArgumentException("k and ef must be positive");
        }
        Objects.requireNonNull(filter, "filter must not be null");
        SearchScratch local = scratch.get();
        float[] prepared = prepare(query, local.query);
        int currentSize = size.get();
        int start = entryPoint;
        if (currentSize == 0 || start < 0) {
            return List.of();
        }
        int[] matching = collectMatching(filter, currentSize);
        if (matching.length <= Math.max(64, currentSize / 20)) {
            List<SearchResult> exact = new ArrayList<>(matching.length);
            for (int id : matching) {
                exact.add(new SearchResult(id, store.dist(prepared, id)));
            }
            exact.sort(Comparator.comparingDouble(SearchResult::distance)
                    .thenComparingInt(SearchResult::id));
            return exact.size() <= k ? exact : new ArrayList<>(exact.subList(0, k));
        }
        int entry = greedyDescend(prepared, start, maxLevel, 1);
        local.entryBuffer[0] = entry;
        int found = searchLayer(prepared, local.entryBuffer, 1, Math.max(ef, k), 0, local, filter);
        List<SearchResult> results = new ArrayList<>(Math.min(k, found));
        for (int i = 0; i < found && results.size() < k; i++) {
            int id = local.foundIds[i];
            if (tombstones.get(id) == 0 && filter.test(id)) {
                results.add(new SearchResult(id, local.foundDistances[i]));
            }
        }
        return results;
    }

    private int[] collectMatching(IntPredicate filter, int currentSize) {
        int[] ids = new int[currentSize];
        int count = 0;
        for (int id = 0; id < currentSize; id++) {
            if (tombstones.get(id) == 0 && filter.test(id)) {
                ids[count++] = id;
            }
        }
        return Arrays.copyOf(ids, count);
    }

    @Override
    public int size() {
        return size.get();
    }

    /**
     * Marks a vector as deleted. Deleted nodes are still traversed during search
     * but are never returned. The graph is not physically repaired.
     *
     * @param id id of an existing vector
     * @return true if the vector was live and is now deleted, false if already deleted
     * @throws IllegalArgumentException if {@code id} is outside {@code [0, size())}
     */
    public boolean delete(int id) {
        if (id < 0 || id >= size.get()) {
            throw new IllegalArgumentException("id outside [0, size()): " + id);
        }
        if (tombstones.getAndSet(id, 1) != 0) {
            return false;
        }
        liveCount.decrementAndGet();
        return true;
    }

    /** @return number of live vectors, excluding tombstones */
    public int liveCount() {
        return liveCount.get();
    }

    boolean isDeleted(int id) {
        return tombstones.get(id) != 0;
    }

    @Override
    public int dimension() {
        return params.dimension();
    }

    @Override
    public Metric metric() {
        return params.metric();
    }

    /** @return the parameters this index was built with */
    public HnswParams params() {
        return params;
    }

    /** @return the storage mode used for vectors */
    public StorageMode storageMode() {
        return storageMode;
    }

    @Override
    public void save(Path path) throws IOException {
        HnswSerializer.save(snapshot(), path);
    }

    /** Loads a full in-memory index from a v1 file. */
    public static HnswIndex load(Path path) throws IOException {
        return fromSnapshot(HnswSerializer.load(path));
    }

    /** Opens a read-only memory-mapped index. */
    public static VectorIndex loadMapped(Path path) throws IOException {
        return MappedHnswIndex.open(path);
    }

    /** Returns a consistent-enough snapshot of the current graph state. */
    public IndexSnapshot snapshot() {
        int count = size.get();
        float[] vectors = new float[count * params.dimension()];
        for (int id = 0; id < count; id++) {
            float[] vector = store.copyOf(id);
            System.arraycopy(vector, 0, vectors, id * params.dimension(), params.dimension());
        }
        int[] levelCopy = Arrays.copyOf(levels, count);
        int[] layer0Copy = Arrays.copyOf(layer0, count * layer0Stride);
        int[][] upperCopy = new int[params.maxElements()][];
        for (int id = 0; id < count; id++) {
            if (upper[id] != null) {
                upperCopy[id] = upper[id].clone();
            }
        }
        return new IndexSnapshot(params, count, entryPoint, maxLevel, vectors,
                levelCopy, layer0Copy, upperCopy);
    }

    /** Rebuilds an index from a validated snapshot. */
    public static HnswIndex fromSnapshot(IndexSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        HnswIndex index = new HnswIndex(snapshot.params());
        int count = snapshot.size();
        if (count < 0 || count > snapshot.params().maxElements()) {
            throw new IllegalArgumentException("invalid snapshot size: " + count);
        }
        int dimension = snapshot.params().dimension();
        if (snapshot.vectors().length != count * dimension) {
            throw new IllegalArgumentException("snapshot vector array has the wrong length");
        }
        for (int id = 0; id < count; id++) {
            float[] vector = Arrays.copyOfRange(snapshot.vectors(),
                    id * dimension, (id + 1) * dimension);
            index.store.set(id, vector);
            index.levels[id] = snapshot.levels()[id];
            if (index.levels[id] > 0) {
                index.upper[id] = new int[index.levels[id] * index.upperStride];
                if (snapshot.upper()[id] != null) {
                    System.arraycopy(snapshot.upper()[id], 0, index.upper[id],
                            0, Math.min(index.upper[id].length, snapshot.upper()[id].length));
                }
            }
            index.published.set(id, 1);
        }
        System.arraycopy(snapshot.layer0(), 0, index.layer0, 0, snapshot.layer0().length);
        index.size.set(count);
        index.liveCount.set(count);
        index.entryPoint = snapshot.entryPoint();
        index.maxLevel = snapshot.maxLevel();
        if (count > 0) {
            index.firstPublished.countDown();
        }
        return index;
    }

    @Override
    public void close() {
        store.close();
    }

    int level(int id) {
        return levels[id];
    }

    float[] vector(int id) {
        return store.copyOf(id);
    }

    int entryPoint() {
        return entryPoint;
    }

    int maxLevel() {
        return maxLevel;
    }

    int neighborCount(int id, int layer) {
        return degree(id, layer);
    }

    int neighborAt(int id, int layer, int index) {
        if (layer == 0) {
            return (int) INT_HANDLE.getAcquire(layer0, id * layer0Stride + 1 + index);
        }
        return (int) INT_HANDLE.getAcquire(upper[id],
                (layer - 1) * upperStride + 1 + index);
    }

    int[] selectHeuristic(int baseId, List<Candidate> sorted, int limit, boolean keepPruned) {
        int count = sorted.size();
        int[] ids = new int[count];
        float[] distances = new float[count];
        for (int i = 0; i < count; i++) {
            ids[i] = sorted.get(i).id();
            distances[i] = sorted.get(i).distance();
        }
        int[] out = new int[limit];
        int selected = selectHeuristic(ids, distances, count, limit, out, keepPruned);
        return Arrays.copyOf(out, selected);
    }

    private int degree(int id, int layer) {
        if (layer == 0) {
            return (int) INT_HANDLE.getAcquire(layer0, id * layer0Stride);
        }
        int[] slice = upper[id];
        return slice == null ? 0 : (int) INT_HANDLE.getAcquire(slice, (layer - 1) * upperStride);
    }

    private boolean hasNeighbor(int id, int layer, int neighbor) {
        int count = degree(id, layer);
        for (int i = 0; i < count; i++) {
            if (neighborAt(id, layer, i) == neighbor) {
                return true;
            }
        }
        return false;
    }

    private void setDegree(int id, int layer, int count) {
        if (layer == 0) {
            INT_HANDLE.setRelease(layer0, id * layer0Stride, count);
        } else {
            INT_HANDLE.setRelease(upper[id], (layer - 1) * upperStride, count);
        }
    }

    private void setNeighbors(int id, int layer, int[] ids, int count) {
        int[] target = layer == 0 ? layer0 : upper[id];
        int offset = layer == 0 ? id * layer0Stride : (layer - 1) * upperStride;
        int written = 0;
        for (int i = 0; i < count && written < params.maxM0(); i++) {
            int neighbor = ids[i];
            if (neighbor == id || neighbor < 0 || neighbor >= params.maxElements()) {
                continue;
            }
            boolean duplicate = false;
            for (int j = 0; j < written; j++) {
                if (target[offset + 1 + j] == neighbor) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                target[offset + 1 + written++] = neighbor;
            }
        }
        INT_HANDLE.setRelease(target, offset, written);
    }

    private void setNeighborAt(int id, int layer, int index, int neighbor) {
        if (layer == 0) {
            INT_HANDLE.setRelease(layer0, id * layer0Stride + 1 + index, neighbor);
        } else {
            INT_HANDLE.setRelease(upper[id], (layer - 1) * upperStride + 1 + index, neighbor);
        }
    }

    private int gatherWithExtra(int baseId, int layer, int extraId, SearchScratch local) {
        int count = degree(baseId, layer);
        for (int i = 0; i < count; i++) {
            int id = neighborAt(baseId, layer, i);
            local.combinedIds[i] = id;
            local.combinedDistances[i] = store.dist(baseId, id);
        }
        local.combinedIds[count] = extraId;
        local.combinedDistances[count] = store.dist(baseId, extraId);
        return count + 1;
    }

    private static void sortCandidates(int[] ids, float[] distances, int count) {
        for (int i = 1; i < count; i++) {
            int id = ids[i];
            float distance = distances[i];
            int j = i - 1;
            while (j >= 0 && (distances[j] > distance
                    || (distances[j] == distance && ids[j] > id))) {
                ids[j + 1] = ids[j];
                distances[j + 1] = distances[j];
                j--;
            }
            ids[j + 1] = id;
            distances[j + 1] = distance;
        }
    }

    private int selectByStrategy(int baseId, int[] ids, float[] distances, int count,
                                 int limit, int[] out) {
        if (params.neighborSelection() == NeighborSelection.SIMPLE) {
            int result = Math.min(limit, count);
            System.arraycopy(ids, 0, out, 0, result);
            return result;
        }
        return selectHeuristic(ids, distances, count, limit, out, false);
    }

    private int selectHeuristic(int[] ids, float[] distances, int count, int limit,
                                int[] out, boolean keepPruned) {
        int selected = 0;
        boolean[] used = new boolean[count];
        for (int i = 0; i < count && selected < limit; i++) {
            boolean diverse = true;
            for (int j = 0; j < selected; j++) {
                if (store.dist(ids[i], out[j]) < distances[i]) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) {
                out[selected++] = ids[i];
                used[i] = true;
            }
        }
        if (keepPruned) {
            for (int i = 0; i < count && selected < limit; i++) {
                if (!used[i]) {
                    out[selected++] = ids[i];
                }
            }
        }
        return selected;
    }

    private int randomLevel() {
        double uniform;
        synchronized (randomLock) {
            uniform = 1.0 - random.nextDouble();
        }
        int level = (int) Math.floor(-Math.log(uniform) * params.levelMult());
        return Math.min(level, MAX_LEVEL_CAP);
    }

    private int searchLayer(float[] query, int[] entryPoints, int entryCount, int ef,
                            int layer, SearchScratch local) {
        return searchLayer(query, entryPoints, entryCount, ef, layer, local, null);
    }

    private int searchLayer(float[] query, int[] entryPoints, int entryCount, int ef,
                            int layer, SearchScratch local, IntPredicate filter) {
        local.visited.begin();
        local.candidates.clear();
        local.results.reset(ef);
        local.ensureFound(ef);
        for (int i = 0; i < entryCount; i++) {
            int entry = entryPoints[i];
            if (entry < 0 || entry >= size.get() || published.get(entry) == 0
                    || !local.visited.mark(entry)) {
                continue;
            }
            float distanceToEntry = store.dist(query, entry);
            local.candidates.push(entry, distanceToEntry);
            if (filter == null || filter.test(entry)) {
                local.results.push(entry, distanceToEntry);
            }
        }
        while (local.candidates.size() > 0) {
            int current = local.candidates.pop();
            float currentDistance = local.candidates.popDistance();
            if (local.results.size() == ef && currentDistance > local.results.peekDistance()) {
                break;
            }
            if (layer > levels[current]) {
                continue;
            }
            int count = degree(current, layer);
            if (prefetch) {
                int batch = 0;
                for (int i = 0; i < count; i++) {
                    int neighbor = neighborAt(current, layer, i);
                    if (neighbor >= 0 && neighbor < size.get() && published.get(neighbor) == 1
                            && local.visited.mark(neighbor)) {
                        local.neighborBatch[batch++] = neighbor;
                    }
                }
                for (int i = 0; i < batch; i++) {
                    int neighbor = local.neighborBatch[i];
                    float distanceToNeighbor = store.dist(query, neighbor);
                    local.candidates.push(neighbor, distanceToNeighbor);
                    if ((filter == null || filter.test(neighbor))
                            && (local.results.size() < ef
                            || distanceToNeighbor < local.results.peekDistance())) {
                        local.results.push(neighbor, distanceToNeighbor);
                    }
                }
            } else {
                for (int i = 0; i < count; i++) {
                    int neighbor = neighborAt(current, layer, i);
                    if (neighbor < 0 || neighbor >= size.get() || published.get(neighbor) == 0
                            || !local.visited.mark(neighbor)) {
                        continue;
                    }
                    float distanceToNeighbor = store.dist(query, neighbor);
                    local.candidates.push(neighbor, distanceToNeighbor);
                    if ((filter == null || filter.test(neighbor))
                            && (local.results.size() < ef
                            || distanceToNeighbor < local.results.peekDistance())) {
                        local.results.push(neighbor, distanceToNeighbor);
                    }
                }
            }
        }
        int found = local.results.size();
        for (int index = found - 1; index >= 0; index--) {
            local.foundIds[index] = local.results.pop();
            local.foundDistances[index] = local.results.popDistance();
        }
        return found;
    }

    private int greedyDescend(float[] query, int entry, int fromLayer, int toLayer) {
        int current = entry;
        float currentDistance = store.dist(query, current);
        for (int layer = fromLayer; layer > toLayer; layer--) {
            if (layer > levels[current]) {
                continue;
            }
            boolean improved;
            do {
                improved = false;
                int node = current;
                int count = degree(node, layer);
                for (int i = 0; i < count; i++) {
                    int neighbor = neighborAt(node, layer, i);
                    if (neighbor < 0 || neighbor >= size.get() || published.get(neighbor) == 0) {
                        continue;
                    }
                    float distanceToNeighbor = store.dist(query, neighbor);
                    if (distanceToNeighbor < currentDistance) {
                        current = neighbor;
                        currentDistance = distanceToNeighbor;
                        improved = true;
                    }
                }
            } while (improved && layer <= levels[current]);
        }
        return current;
    }

    private float[] prepare(float[] vector, float[] target) {
        Objects.requireNonNull(vector, "vector must not be null");
        if (vector.length != params.dimension()) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + params.dimension() + ", got " + vector.length);
        }
        for (float value : vector) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException("vector values must not be NaN");
            }
        }
        System.arraycopy(vector, 0, target, 0, params.dimension());
        if (params.metric() == Metric.COSINE) {
            normalize(target);
        }
        return target;
    }

    private static void normalize(float[] vector) {
        double sum = 0.0;
        for (float value : vector) {
            sum += (double) value * value;
        }
        if (sum == 0.0) {
            throw new IllegalArgumentException("cosine distance is undefined for a zero vector");
        }
        double norm = Math.sqrt(sum);
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (vector[i] / norm);
        }
    }

    /** A candidate id paired with its distance, used by the heuristic test hook. */
    record Candidate(int id, float distance) {}
}
