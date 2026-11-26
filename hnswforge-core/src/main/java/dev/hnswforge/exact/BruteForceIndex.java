package dev.hnswforge.exact;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.index.VectorIndex;
import dev.hnswforge.util.BoundedMaxHeap;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Exact brute-force k-nearest-neighbor index.
 *
 * <p>Vectors are copied on insertion; cosine vectors are additionally normalized.
 * Search is a linear scan with a bounded max-heap of size {@code k}, so results
 * are exact and ties are broken by ascending id. It is used as the ground-truth
 * oracle and as a correctness reference for the approximate index.
 *
 * <p>Thread-safety: not safe for concurrent writes. After construction and all
 * insertions are complete, concurrent searches are safe because nothing is mutated.
 * {@link #save(Path)} is unsupported.
 */
public final class BruteForceIndex implements VectorIndex {
    private final int dimension;
    private final Metric metric;
    private final DistanceFunction distance;
    private float[][] vectors;
    private int size;

    /** Creates an empty exact index for the given dimension and metric. */
    public BruteForceIndex(int dimension, Metric metric) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, got " + dimension);
        }
        this.dimension = dimension;
        this.metric = Objects.requireNonNull(metric, "metric must not be null");
        this.distance = ScalarDistance.forMetric(metric);
        this.vectors = new float[16][];
    }

    @Override
    public int add(float[] vector) {
        float[] stored = prepare(vector);
        if (size == vectors.length) {
            float[][] grown = new float[vectors.length * 2][];
            System.arraycopy(vectors, 0, grown, 0, size);
            vectors = grown;
        }
        int id = size++;
        vectors[id] = stored;
        return id;
    }

    @Override
    public List<SearchResult> search(float[] query, int k) {
        return search(query, k, 0);
    }

    @Override
    public List<SearchResult> search(float[] query, int k, int efSearch) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        float[] prepared = prepare(query);
        if (size == 0) {
            return List.of();
        }
        BoundedMaxHeap heap = new BoundedMaxHeap(Math.min(k, size));
        for (int id = 0; id < size; id++) {
            heap.push(id, distance.distance(prepared, vectors[id]));
        }
        int count = heap.size();
        int[] ids = new int[count];
        float[] distances = new float[count];
        for (int index = count - 1; index >= 0; index--) {
            ids[index] = heap.pop();
            distances[index] = heap.popDistance();
        }
        List<SearchResult> results = new ArrayList<>(count);
        for (int index = 0; index < count; index++) {
            results.add(new SearchResult(ids[index], distances[index]));
        }
        return List.copyOf(results);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public Metric metric() {
        return metric;
    }

    @Override
    public void save(Path path) throws IOException {
        throw new UnsupportedOperationException("BruteForceIndex cannot be saved");
    }

    @Override
    public void close() {
        vectors = null;
        size = 0;
    }

    private float[] prepare(float[] vector) {
        Objects.requireNonNull(vector, "vector must not be null");
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "dimension mismatch: expected " + dimension + ", got " + vector.length);
        }
        float[] copy = vector.clone();
        for (float value : copy) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException("vector values must not be NaN");
            }
        }
        return metric == Metric.COSINE ? normalize(copy) : copy;
    }

    private float[] normalize(float[] vector) {
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
}
