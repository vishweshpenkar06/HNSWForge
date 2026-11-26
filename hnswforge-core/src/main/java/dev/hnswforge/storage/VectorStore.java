package dev.hnswforge.storage;

/**
 * Storage for vectors addressed by integer id, with distance computation directly
 * against the backing storage.
 *
 * <p>Implementations keep vectors in insertion order at id {@code 0..size-1} and
 * provide distances without creating an intermediate {@code float[]} copy.
 * {@code copyOf} is provided for persistence, debugging and exact-search callers.
 *
 * <p>Thread-safety: implementations are not thread-safe for writes; concurrent
 * reads are safe after all writes are published.
 */
public interface VectorStore extends AutoCloseable {

    /** @return dimension of every stored vector */
    int dimension();

    /** @return maximum number of vectors that can be stored */
    int capacity();

    /** @return number of stored vectors */
    int size();

    /**
     * Stores a vector under {@code id}, growing the visible size to {@code id + 1}.
     *
     * @param id     target id in {@code [0, capacity())}
     * @param vector vector of length {@link #dimension()}
     * @throws IllegalArgumentException if the id or vector is invalid
     */
    void set(int id, float[] vector);

    /** @return a copy of the vector at {@code id} */
    float[] copyOf(int id);

    /** @return distance between a query and the vector at {@code id} */
    float dist(float[] query, int id);

    /** @return distance between the vectors at {@code a} and {@code b} */
    float dist(int a, int b);

    /** Releases resources; calling more than once has no effect. */
    @Override
    void close();
}
