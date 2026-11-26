package dev.hnswforge.index;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Common contract for exact and approximate vector indexes.
 *
 * <p>Implementations validate all inputs: a null vector, a length different from
 * {@link #dimension()}, NaN values, and {@code k <= 0} are rejected with
 * {@link IllegalArgumentException}. Search results are sorted by ascending
 * internal distance, with ties broken by ascending id.
 *
 * <p>Thread-safety: implementation-specific. {@code HnswIndex} is thread-safe for
 * concurrent {@code add} and {@code search}; {@code BruteForceIndex} supports
 * concurrent reads only after its build is complete.
 */
public interface VectorIndex extends AutoCloseable {

    /**
     * Adds a vector and returns its assigned id (insertion order, starting at zero).
     *
     * @param vector vector of length {@link #dimension()}
     * @return the id assigned to the vector
     * @throws IllegalArgumentException if the vector is null, has the wrong length, or contains NaN
     */
    int add(float[] vector);

    /**
     * Returns up to {@code k} closest vectors, sorted by ascending distance.
     *
     * @param query query vector
     * @param k     maximum number of results, must be positive
     * @return results, never null; an empty list when the index is empty
     * @throws IllegalArgumentException if inputs are invalid
     */
    List<SearchResult> search(float[] query, int k);

    /**
     * Returns up to {@code k} closest vectors using an explicit search-effort override.
     * The effective effort is {@code max(efSearch, k)}. Implementations with a fixed
     * search effort may ignore the override.
     *
     * @param query    query vector
     * @param k        maximum number of results, must be positive
     * @param efSearch search effort, must be positive
     * @return results sorted by ascending distance
     * @throws IllegalArgumentException if inputs are invalid
     */
    List<SearchResult> search(float[] query, int k, int efSearch);

    /** @return number of vectors currently in the index */
    int size();

    /** @return dimension of every vector in the index */
    int dimension();

    /** @return the metric used by this index */
    Metric metric();

    /**
     * Writes a snapshot to {@code path}.
     *
     * @param path destination path
     * @throws IOException              if writing fails
     * @throws UnsupportedOperationException if the implementation cannot be saved
     */
    void save(Path path) throws IOException;

    /** Releases resources. Calling more than once has no effect. */
    @Override
    void close();
}
