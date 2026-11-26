package dev.hnswforge.index;

/**
 * A single nearest-neighbor result: the internal id of a vector and its distance.
 *
 * <p>The distance is the index's internal distance as described in {@link Metric};
 * it is never a similarity score and is not normalized across metrics.
 *
 * <p>Thread-safety: immutable record; inherently thread-safe.
 *
 * @param id       id of the matching vector, always non-negative
 * @param distance internal distance to the query, smaller is closer
 */
public record SearchResult(int id, float distance) {
    /** Creates a result, rejecting invalid ids and NaN distances. */
    public SearchResult {
        if (id < 0) {
            throw new IllegalArgumentException("id must be >= 0, got " + id);
        }
        if (Float.isNaN(distance)) {
            throw new IllegalArgumentException("distance must not be NaN");
        }
    }
}
