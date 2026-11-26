package dev.hnswforge.storage;

import dev.hnswforge.distance.DistanceFunction;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Objects;

/**
 * Heap-based {@link VectorStore} using one contiguous {@code float[]}.
 *
 * <p>Vector {@code id} occupies floats {@code [id * dimension, (id + 1) * dimension)}.
 * Distances are computed directly against a slice of the backing array with no
 * per-call allocation.
 *
 * <p>Thread-safety: not thread-safe for writes; concurrent reads are safe after
 * all writes are published.
 */
public final class HeapVectorStore implements VectorStore {
    private final int dimension;
    private final int capacity;
    private final float[] data;
    private final DistanceFunction kernel;
    private final AtomicInteger size = new AtomicInteger();

    /** Creates a store with the given capacity and distance kernel. */
    public HeapVectorStore(int dimension, int capacity, DistanceFunction kernel) {
        if (dimension <= 0 || capacity <= 0) {
            throw new IllegalArgumentException("dimension and capacity must be positive");
        }
        this.dimension = dimension;
        this.capacity = capacity;
        this.kernel = Objects.requireNonNull(kernel, "kernel must not be null");
        this.data = new float[Math.multiplyExact(dimension, capacity)];
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int capacity() {
        return capacity;
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public void set(int id, float[] vector) {
        Objects.requireNonNull(vector, "vector must not be null");
        if (id < 0 || id >= capacity) {
            throw new IllegalArgumentException("id outside [0, capacity()): " + id);
        }
        if (vector.length != dimension) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + dimension + ", got " + vector.length);
        }
        System.arraycopy(vector, 0, data, id * dimension, dimension);
        size.accumulateAndGet(id + 1, Math::max);
    }

    @Override
    public float[] copyOf(int id) {
        checkId(id);
        return Arrays.copyOfRange(data, id * dimension, (id + 1) * dimension);
    }

    @Override
    public float dist(float[] query, int id) {
        checkId(id);
        return kernel.distance(query, 0, data, id * dimension, dimension);
    }

    @Override
    public float dist(int a, int b) {
        checkId(a);
        checkId(b);
        return kernel.distance(data, a * dimension, data, b * dimension, dimension);
    }

    @Override
    public void close() {
    }

    private void checkId(int id) {
        if (id < 0 || id >= size.get()) {
            throw new IllegalArgumentException("id outside [0, size()): " + id);
        }
    }
}
