package dev.hnswforge.index;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.DistanceFunctions;
import dev.hnswforge.storage.HeapVectorStore;
import dev.hnswforge.storage.OffHeapVectorStore;
import dev.hnswforge.storage.StorageMode;
import dev.hnswforge.storage.VectorStore;
import dev.hnswforge.util.SearchScratch;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * HNSW index with flat vector and adjacency storage, epoch-stamped visited sets
 * and reusable per-thread search buffers.
 *
 * <p>Vectors live in a contiguous {@link VectorStore} (heap or off-heap); layer 0
 * adjacency lives in one flat {@code int[]} with a degree word followed by neighbor
 * ids per node, and upper layers are allocated lazily. Searches take a read lock
 * and allocate only the returned result list; inserts take a write lock. Random
 * levels derive entirely from the seed in {@link HnswParams}.
 *
 * <p>Deletes are tombstones: deleted nodes remain in the graph and are traversed
 * but never returned. {@link #size()} counts all inserts, {@link #liveCount()}
 * excludes tombstones.
 *
 * <p>Thread-safety: thread-safe for concurrent inserts and searches.
 */
public final class HnswIndex implements VectorIndex {
    private static final int MAX_LEVEL_CAP = 32;

    private final HnswParams params;
    private final VectorStore store;
    private final StorageMode storageMode;
    private final boolean prefetch;
    private final SplittableRandom random;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private final ThreadLocal<SearchScratch> scratch;

    private final int[] levels;
    private final int[] layer0;
    private final int[][] upper;
    private final int layer0Stride;
    private final int upperStride;
    private final boolean[] deleted;
    private int entryPoint = -1;
    private int maxLevel = -1;
    private int size;
    private int liveCount;

    /** Creates an index with SIMD kernels, on-heap storage and prefetch batching. */
    public HnswIndex(HnswParams params) {
        this(params, true, StorageMode.HEAP, true);
    }

    /**
     * Creates an index with explicit kernel and storage choices and prefetch batching.
     *
     * @param params       construction parameters
     * @param preferSimd   whether to use the Vector API when available
     * @param storageMode  heap or off-heap vector storage
     */
    public HnswIndex(HnswParams params, boolean preferSimd, StorageMode storageMode) {
        this(params, preferSimd, storageMode, true);
    }

    /**
     * Creates an index with explicit kernel, storage and traversal choices.
     *
     * @param params       construction parameters
     * @param preferSimd   whether to use the Vector API when available
     * @param storageMode  heap or off-heap vector storage
     * @param prefetch     whether to batch neighbor discovery before computing distances
     */
    public HnswIndex(HnswParams params, boolean preferSimd, StorageMode storageMode,
                     boolean prefetch) {
        this.params = Objects.requireNonNull(params, "params must not be null");
        this.storageMode = Objects.requireNonNull(storageMode, "storageMode must not be null");
        this.prefetch = prefetch;
        DistanceFunction kernel = DistanceFunctions.forMetric(params.metric(), preferSimd);
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
        this.deleted = new boolean[capacity];
        this.scratch = ThreadLocal.withInitial(() -> new SearchScratch(
                params.dimension(), capacity,
                Math.max(params.efConstruction(), params.efSearch()), params.maxM0()));
    }

    @Override
    public int add(float[] vector) {
        float[] prepared = prepare(vector, scratch.get().query);
        SearchScratch local = scratch.get();
        lock.writeLock().lock();
        try {
            if (size >= params.maxElements()) {
                throw new IllegalStateException(
                        "index capacity exceeded: " + params.maxElements() + " elements");
            }
            int id = size++;
            liveCount++;
            store.set(id, prepared);
            int level = randomLevel();
            levels[id] = level;
            if (level > 0) {
                upper[id] = new int[level * upperStride];
            }

            if (id == 0) {
                entryPoint = 0;
                maxLevel = level;
                return id;
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
                setNeighbors(id, layer, chosen, chosenCount);
                for (int neighbor : chosen) {
                    int count = neighborCount(neighbor, layer);
                    if (count < bound) {
                        setNeighborAt(neighbor, layer, count, id);
                        setNeighborCount(neighbor, layer, count + 1);
                    } else {
                        int combinedCount = gatherWithExtra(neighbor, layer, id, local);
                        sortCandidates(local.combinedIds, local.combinedDistances, combinedCount);
                        int prunedCount = selectByStrategy(neighbor, local.combinedIds,
                                local.combinedDistances, combinedCount, bound, local.chosen);
                        setNeighbors(neighbor, layer, local.chosen, prunedCount);
                    }
                }
                entryPoints = foundIds;
            }

            if (level > maxLevel) {
                entryPoint = id;
                maxLevel = level;
            }
            return id;
        } finally {
            lock.writeLock().unlock();
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
        lock.readLock().lock();
        try {
            if (size == 0) {
                return List.of();
            }
            int entry = greedyDescend(prepared, entryPoint, maxLevel, 1);
            local.entryBuffer[0] = entry;
            int found = searchLayer(prepared, local.entryBuffer, 1, Math.max(efSearch, k),
                    0, local);
            List<SearchResult> results = new ArrayList<>(Math.min(k, found));
            for (int i = 0; i < found && results.size() < k; i++) {
                int id = local.foundIds[i];
                if (!deleted[id]) {
                    results.add(new SearchResult(id, local.foundDistances[i]));
                }
            }
            return results;
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public int size() {
        lock.readLock().lock();
        try {
            return size;
        } finally {
            lock.readLock().unlock();
        }
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
        lock.writeLock().lock();
        try {
            if (id < 0 || id >= size) {
                throw new IllegalArgumentException("id outside [0, size()): " + id);
            }
            if (deleted[id]) {
                return false;
            }
            deleted[id] = true;
            liveCount--;
            return true;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** @return number of live vectors, excluding tombstones */
    public int liveCount() {
        lock.readLock().lock();
        try {
            return liveCount;
        } finally {
            lock.readLock().unlock();
        }
    }

    boolean isDeleted(int id) {
        return deleted[id];
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
        throw new UnsupportedOperationException("Persistence is implemented in Phase 5");
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
        if (layer == 0) {
            return layer0[id * layer0Stride];
        }
        int[] slice = upper[id];
        return slice == null ? 0 : slice[(layer - 1) * upperStride];
    }

    int neighborAt(int id, int layer, int index) {
        if (layer == 0) {
            return layer0[id * layer0Stride + 1 + index];
        }
        return upper[id][(layer - 1) * upperStride + 1 + index];
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
        int selected = selectNeighbors(baseId, ids, distances, count, limit, out, keepPruned);
        return Arrays.copyOf(out, selected);
    }

    private void setNeighbors(int id, int layer, int[] ids, int count) {
        if (layer == 0) {
            int offset = id * layer0Stride;
            layer0[offset] = count;
            System.arraycopy(ids, 0, layer0, offset + 1, count);
        } else {
            int[] slice = upper[id];
            int offset = (layer - 1) * upperStride;
            slice[offset] = count;
            System.arraycopy(ids, 0, slice, offset + 1, count);
        }
    }

    private void setNeighborCount(int id, int layer, int count) {
        if (layer == 0) {
            layer0[id * layer0Stride] = count;
        } else {
            upper[id][(layer - 1) * upperStride] = count;
        }
    }

    private void setNeighborAt(int id, int layer, int index, int neighbor) {
        if (layer == 0) {
            layer0[id * layer0Stride + 1 + index] = neighbor;
        } else {
            upper[id][(layer - 1) * upperStride + 1 + index] = neighbor;
        }
    }

    private int gatherWithExtra(int baseId, int layer, int extraId, SearchScratch local) {
        int count = neighborCount(baseId, layer);
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

    private int selectNeighbors(int baseId, int[] ids, float[] distances, int count,
                                int limit, int[] out, boolean keepPruned) {
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

    private int selectByStrategy(int baseId, int[] ids, float[] distances, int count,
                                 int limit, int[] out) {
        if (params.neighborSelection() == NeighborSelection.SIMPLE) {
            int result = Math.min(limit, count);
            System.arraycopy(ids, 0, out, 0, result);
            return result;
        }
        return selectNeighbors(baseId, ids, distances, count, limit, out, false);
    }

    private int randomLevel() {
        double uniform = 1.0 - random.nextDouble();
        int level = (int) Math.floor(-Math.log(uniform) * params.levelMult());
        return Math.min(level, MAX_LEVEL_CAP);
    }

    private int searchLayer(float[] query, int[] entryPoints, int entryCount, int ef,
                            int layer, SearchScratch local) {
        local.visited.begin();
        local.candidates.clear();
        local.results.reset(ef);
        local.ensureFound(ef);
        for (int i = 0; i < entryCount; i++) {
            int entry = entryPoints[i];
            if (entry < 0 || entry >= size || !local.visited.mark(entry)) {
                continue;
            }
            float distanceToEntry = store.dist(query, entry);
            local.candidates.push(entry, distanceToEntry);
            local.results.push(entry, distanceToEntry);
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
            int count = neighborCount(current, layer);
            if (prefetch) {
                int batch = 0;
                for (int i = 0; i < count; i++) {
                    int neighbor = neighborAt(current, layer, i);
                    if (neighbor >= 0 && neighbor < size && local.visited.mark(neighbor)) {
                        local.neighborBatch[batch++] = neighbor;
                    }
                }
                for (int i = 0; i < batch; i++) {
                    int neighbor = local.neighborBatch[i];
                    float distanceToNeighbor = store.dist(query, neighbor);
                    if (local.results.size() < ef
                            || distanceToNeighbor < local.results.peekDistance()) {
                        local.candidates.push(neighbor, distanceToNeighbor);
                        local.results.push(neighbor, distanceToNeighbor);
                    }
                }
            } else {
                for (int i = 0; i < count; i++) {
                    int neighbor = neighborAt(current, layer, i);
                    if (neighbor < 0 || neighbor >= size || !local.visited.mark(neighbor)) {
                        continue;
                    }
                    float distanceToNeighbor = store.dist(query, neighbor);
                    if (local.results.size() < ef
                            || distanceToNeighbor < local.results.peekDistance()) {
                        local.candidates.push(neighbor, distanceToNeighbor);
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
                int count = neighborCount(node, layer);
                for (int i = 0; i < count; i++) {
                    int neighbor = neighborAt(node, layer, i);
                    if (neighbor < 0 || neighbor >= size) {
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
