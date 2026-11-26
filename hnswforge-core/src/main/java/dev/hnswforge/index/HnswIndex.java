package dev.hnswforge.index;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.util.BoundedMaxHeap;
import dev.hnswforge.util.MinHeap;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Naive but correct HNSW index using object-based storage and a coarse read-write lock.
 *
 * <p>Insertion follows the paper: a random level is drawn, the graph is descended
 * greedily through upper layers, and each visited layer gets a best-first search
 * whose results are used to connect the new node bidirectionally. Searches use the
 * same best-first procedure at layer 0 after a greedy descent from the top layer.
 * All randomness comes from the seed in {@link HnswParams}, so a given seed and
 * insertion order produce an identical graph.
 *
 * <p>Thread-safety: thread-safe. Inserts take the write lock and searches take the
 * read lock; reads therefore never observe a partially inserted node.
 */
public final class HnswIndex implements VectorIndex {
    private static final int MAX_LEVEL_CAP = 32;

    private final HnswParams params;
    private final DistanceFunction distance;
    private final SplittableRandom random;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    private final float[][] vectors;
    private final int[] levels;
    private final int[][][] neighbors;
    private final int[][] neighborCounts;
    private int entryPoint = -1;
    private int maxLevel = -1;
    private int size;

    /** Creates an empty index with the capacity declared in {@code params}. */
    public HnswIndex(HnswParams params) {
        this.params = Objects.requireNonNull(params, "params must not be null");
        this.distance = ScalarDistance.forMetric(params.metric());
        this.random = new SplittableRandom(params.seed());
        int capacity = params.maxElements();
        this.vectors = new float[capacity][];
        this.levels = new int[capacity];
        this.neighbors = new int[capacity][][];
        this.neighborCounts = new int[capacity][];
    }

    @Override
    public int add(float[] vector) {
        float[] stored = prepare(vector);
        lock.writeLock().lock();
        try {
            if (size >= vectors.length) {
                throw new IllegalStateException(
                        "index capacity exceeded: " + vectors.length + " elements");
            }
            int id = size++;
            vectors[id] = stored;
            int level = randomLevel();
            levels[id] = level;
            neighbors[id] = new int[level + 1][];
            neighborCounts[id] = new int[level + 1];
            for (int layer = 0; layer <= level; layer++) {
                neighbors[id][layer] = new int[layer == 0 ? params.maxM0() : params.m()];
            }
            if (id == 0) {
                entryPoint = 0;
                maxLevel = level;
                return id;
            }

            int[] entryPoints = {entryPoint};
            for (int layer = maxLevel; layer > level; layer--) {
                entryPoints = new int[] {greedyDescend(stored, entryPoints[0], layer, layer - 1)};
            }

            for (int layer = Math.min(level, maxLevel); layer >= 0; layer--) {
                List<Candidate> found = searchLayer(stored, entryPoints, params.efConstruction(), layer);
                int[] chosen = selectNeighbors(id, found, params.m());
                int bound = layer == 0 ? params.maxM0() : params.m();
                connect(id, layer, chosen, bound);
                for (int neighbor : chosen) {
                    int count = neighborCounts[neighbor][layer];
                    if (count < bound) {
                        neighbors[neighbor][layer][count] = id;
                        neighborCounts[neighbor][layer] = count + 1;
                    } else {
                        List<Candidate> combined = candidatesFor(neighbor, layer, count, id);
                        int[] pruned = selectNeighbors(neighbor, combined, bound);
                        System.arraycopy(pruned, 0, neighbors[neighbor][layer], 0, pruned.length);
                        neighborCounts[neighbor][layer] = pruned.length;
                    }
                }
                entryPoints = found.stream().mapToInt(Candidate::id).toArray();
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
        float[] prepared = prepare(query);
        lock.readLock().lock();
        try {
            if (size == 0) {
                return List.of();
            }
            int entry = greedyDescend(prepared, entryPoint, maxLevel, 1);
            List<Candidate> found = searchLayer(prepared, new int[] {entry},
                    Math.max(efSearch, k), 0);
            int limit = Math.min(k, found.size());
            List<SearchResult> results = new ArrayList<>(limit);
            for (int i = 0; i < limit; i++) {
                Candidate candidate = found.get(i);
                results.add(new SearchResult(candidate.id(), candidate.distance()));
            }
            return List.copyOf(results);
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

    @Override
    public void save(Path path) throws IOException {
        throw new UnsupportedOperationException("Persistence is implemented in Phase 5");
    }

    @Override
    public void close() {
    }

    int level(int id) {
        return levels[id];
    }

    float[] vector(int id) {
        return vectors[id];
    }

    int entryPoint() {
        return entryPoint;
    }

    int maxLevel() {
        return maxLevel;
    }

    int neighborCount(int id, int layer) {
        return neighborCounts[id][layer];
    }

    int neighborAt(int id, int layer, int index) {
        return neighbors[id][layer][index];
    }

    int[] selectHeuristic(int baseId, List<Candidate> sorted, int limit, boolean keepPruned) {
        List<Candidate> selected = new ArrayList<>();
        List<Candidate> pruned = new ArrayList<>();
        for (Candidate candidate : sorted) {
            if (selected.size() == limit) {
                break;
            }
            boolean diverse = true;
            for (Candidate chosen : selected) {
                if (distance.distance(vectors[baseId], vectors[candidate.id()])
                        < candidate.distance()) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) {
                selected.add(candidate);
            } else {
                pruned.add(candidate);
            }
        }
        if (keepPruned) {
            for (Candidate candidate : pruned) {
                if (selected.size() == limit) {
                    break;
                }
                selected.add(candidate);
            }
        }
        return idsOf(selected, limit);
    }

    private void connect(int id, int layer, int[] chosen, int bound) {
        if (chosen.length > bound) {
            throw new IllegalStateException("neighbor selection exceeded the degree bound");
        }
        System.arraycopy(chosen, 0, neighbors[id][layer], 0, chosen.length);
        neighborCounts[id][layer] = chosen.length;
    }

    private List<Candidate> candidatesFor(int baseId, int layer, int count, int extraId) {
        List<Candidate> candidates = new ArrayList<>(count + 1);
        for (int i = 0; i < count; i++) {
            int id = neighbors[baseId][layer][i];
            candidates.add(new Candidate(id, distance.distance(vectors[baseId], vectors[id])));
        }
        if (extraId >= 0) {
            candidates.add(new Candidate(extraId,
                    distance.distance(vectors[baseId], vectors[extraId])));
        }
        candidates.sort(Comparator.comparingDouble(Candidate::distance)
                .thenComparingInt(Candidate::id));
        return candidates;
    }

    private int[] selectNeighbors(int baseId, List<Candidate> sorted, int limit) {
        if (params.neighborSelection() == NeighborSelection.SIMPLE) {
            return idsOf(sorted, limit);
        }
        return selectHeuristic(baseId, sorted, limit, false);
    }

    private static int[] idsOf(List<Candidate> candidates, int limit) {
        int count = Math.min(limit, candidates.size());
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) {
            ids[i] = candidates.get(i).id();
        }
        return ids;
    }

    private int randomLevel() {
        double uniform = 1.0 - random.nextDouble();
        int level = (int) Math.floor(-Math.log(uniform) * params.levelMult());
        return Math.min(level, MAX_LEVEL_CAP);
    }

    private List<Candidate> searchLayer(float[] query, int[] entryPoints, int ef, int layer) {
        boolean[] visited = new boolean[size];
        MinHeap candidates = new MinHeap();
        BoundedMaxHeap results = new BoundedMaxHeap(ef);
        for (int entry : entryPoints) {
            if (entry < 0 || entry >= size || visited[entry]) {
                continue;
            }
            visited[entry] = true;
            float distanceToEntry = distance.distance(query, vectors[entry]);
            candidates.push(entry, distanceToEntry);
            results.push(entry, distanceToEntry);
        }
        while (candidates.size() > 0) {
            int current = candidates.pop();
            float currentDistance = candidates.popDistance();
            if (results.size() == ef && currentDistance > results.peekDistance()) {
                break;
            }
            if (layer > levels[current]) {
                continue;
            }
            int[] adjacent = neighbors[current][layer];
            int count = neighborCounts[current][layer];
            for (int i = 0; i < count; i++) {
                int neighbor = adjacent[i];
                if (neighbor < 0 || neighbor >= size || visited[neighbor]) {
                    continue;
                }
                visited[neighbor] = true;
                float distanceToNeighbor = distance.distance(query, vectors[neighbor]);
                if (results.size() < ef || distanceToNeighbor < results.peekDistance()) {
                    candidates.push(neighbor, distanceToNeighbor);
                    results.push(neighbor, distanceToNeighbor);
                }
            }
        }
        return sortedCandidates(results);
    }

    private static List<Candidate> sortedCandidates(BoundedMaxHeap heap) {
        int count = heap.size();
        Candidate[] sorted = new Candidate[count];
        for (int index = count - 1; index >= 0; index--) {
            sorted[index] = new Candidate(heap.pop(), heap.popDistance());
        }
        return List.of(sorted);
    }

    private int greedyDescend(float[] query, int entry, int fromLayer, int toLayer) {
        int current = entry;
        float currentDistance = distance.distance(query, vectors[current]);
        for (int layer = fromLayer; layer > toLayer; layer--) {
            if (layer > levels[current]) {
                continue;
            }
            boolean improved;
            do {
                improved = false;
                int[] adjacent = neighbors[current][layer];
                int count = neighborCounts[current][layer];
                for (int i = 0; i < count; i++) {
                    int neighbor = adjacent[i];
                    if (neighbor < 0 || neighbor >= size) {
                        continue;
                    }
                    float distanceToNeighbor = distance.distance(query, vectors[neighbor]);
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

    private float[] prepare(float[] vector) {
        Objects.requireNonNull(vector, "vector must not be null");
        if (vector.length != params.dimension()) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + params.dimension() + ", got " + vector.length);
        }
        float[] copy = vector.clone();
        for (float value : copy) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException("vector values must not be NaN");
            }
        }
        if (params.metric() == Metric.COSINE) {
            copy = normalize(copy);
        }
        return copy;
    }

    private static float[] normalize(float[] vector) {
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
        return vector;
    }

    /** A candidate id paired with its distance from the query or base vector. */
    record Candidate(int id, float distance) {}
}
