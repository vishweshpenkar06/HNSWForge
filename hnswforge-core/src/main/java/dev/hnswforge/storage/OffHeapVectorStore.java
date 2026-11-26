package dev.hnswforge.storage;

import dev.hnswforge.distance.DistanceFunction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Off-heap {@link VectorStore} backed by one 64-byte-aligned shared memory segment.
 *
 * <p>Vector {@code id} occupies floats {@code [id * dimension, (id + 1) * dimension)}
 * at byte offset {@code id * dimension * Float.BYTES}. Distances are computed by the
 * supplied kernel directly against that segment, without a per-call copy, except
 * {@code dist(int, int)} which copies one operand because the kernel takes one array
 * and one segment.
 *
 * <p>Thread-safety: the shared arena permits cross-thread access; writes are not
 * thread-safe, while concurrent reads are safe after writes are published. Calling
 * any method except {@link #close()} after closure throws {@code IllegalStateException}.
 */
public final class OffHeapVectorStore implements VectorStore {

    private final int dimension;
    private final int capacity;
    private final DistanceFunction kernel;
    private final Arena arena;
    private final MemorySegment data;
    private volatile boolean closed;
    private int size;

    /**
     * Creates an empty off-heap store with the given capacity and distance kernel.
     *
     * <p>Thread-safe: construction is not shared with other threads.
     *
     * @param dimension number of floats per vector
     * @param capacity  maximum number of vectors
     * @param kernel    distance kernel used for all distance calls
     * @throws IllegalArgumentException if dimension or capacity is not positive
     * @throws NullPointerException     if {@code kernel} is null
     */
    public OffHeapVectorStore(int dimension, int capacity, DistanceFunction kernel) {
        if (dimension <= 0 || capacity <= 0) {
            throw new IllegalArgumentException("dimension and capacity must be positive");
        }
        this.dimension = dimension;
        this.capacity = capacity;
        this.kernel = Objects.requireNonNull(kernel, "kernel must not be null");
        this.arena = Arena.ofShared();
        long byteSize = Math.multiplyExact(
                Math.multiplyExact((long) dimension, capacity), (long) Float.BYTES);
        this.data = arena.allocate(byteSize, 64L);
    }

    /**
     * Returns the vector dimension.
     *
     * <p>Thread-safe: reads an immutable field.
     *
     * @throws IllegalStateException if the store is closed
     */
    @Override
    public int dimension() {
        ensureOpen();
        return dimension;
    }

    /**
     * Returns the maximum number of vectors.
     *
     * <p>Thread-safe: reads an immutable field.
     *
     * @throws IllegalStateException if the store is closed
     */
    @Override
    public int capacity() {
        ensureOpen();
        return capacity;
    }

    /**
     * Returns the number of stored vectors.
     *
     * <p>Thread-safe for reads after writes are published.
     *
     * @throws IllegalStateException if the store is closed
     */
    @Override
    public int size() {
        ensureOpen();
        return size;
    }

    /**
     * Stores a vector under {@code id}, growing the visible size to {@code id + 1}.
     *
     * <p>Not thread-safe for writes.
     *
     * @throws IllegalArgumentException if the id is outside {@code [0, capacity())} or
     *                                  the vector length differs from {@link #dimension()}
     * @throws IllegalStateException    if the store is closed
     * @throws NullPointerException     if {@code vector} is null
     */
    @Override
    public void set(int id, float[] vector) {
        ensureOpen();
        Objects.requireNonNull(vector, "vector must not be null");
        if (id < 0 || id >= capacity) {
            throw new IllegalArgumentException("id outside [0, capacity()): " + id);
        }
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "dimension mismatch: expected " + dimension + ", got " + vector.length);
        }
        MemorySegment.copy(vector, 0, data, ValueLayout.JAVA_FLOAT, byteOffset(id), dimension);
        if (id >= size) {
            size = id + 1;
        }
    }

    /**
     * Returns a copy of the vector at {@code id}.
     *
     * <p>Thread-safe for reads after writes are published.
     *
     * @throws IllegalArgumentException if {@code id} is outside {@code [0, size())}
     * @throws IllegalStateException    if the store is closed
     */
    @Override
    public float[] copyOf(int id) {
        ensureOpen();
        checkId(id);
        float[] result = new float[dimension];
        MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, byteOffset(id), result, 0, dimension);
        return result;
    }

    /**
     * Returns the distance between a query and the vector at {@code id}.
     *
     * <p>Thread-safe for reads after writes are published.
     *
     * @throws IllegalArgumentException if {@code id} is outside {@code [0, size())}
     * @throws IllegalStateException    if the store is closed
     */
    @Override
    public float dist(float[] query, int id) {
        ensureOpen();
        checkId(id);
        return kernel.distance(query, data, byteOffset(id), dimension);
    }

    /**
     * Returns the distance between the vectors at {@code a} and {@code b}.
     *
     * <p>Thread-safe for reads after writes are published; one operand is copied into
     * a fresh array so no mutable scratch state is shared.
     *
     * @throws IllegalArgumentException if either id is outside {@code [0, size())}
     * @throws IllegalStateException    if the store is closed
     */
    @Override
    public float dist(int a, int b) {
        ensureOpen();
        checkId(a);
        checkId(b);
        return kernel.distance(copyOf(a), data, byteOffset(b), dimension);
    }

    /**
     * Returns the backing segment for persistence and tests.
     *
     * <p>Thread-safe: returns the same immutable segment reference while open.
     *
     * @throws IllegalStateException if the store is closed
     */
    public MemorySegment segment() {
        ensureOpen();
        return data;
    }

    /**
     * Releases the backing arena; repeated calls have no effect.
     *
     * <p>Thread-safe: idempotent, but must be called only after access has finished.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            arena.close();
        }
    }

    private long byteOffset(int id) {
        return (long) id * dimension * Float.BYTES;
    }

    private void checkId(int id) {
        if (id < 0 || id >= size) {
            throw new IllegalArgumentException("id outside [0, size()): " + id);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("store is closed");
        }
    }
}
