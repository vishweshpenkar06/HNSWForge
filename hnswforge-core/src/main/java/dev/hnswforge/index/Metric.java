package dev.hnswforge.index;

/**
 * Distance metric used by an index.
 *
 * <p>Internally every metric is expressed as a distance where smaller means closer:
 * {@code L2} is squared Euclidean distance, {@code COSINE} and {@code INNER_PRODUCT}
 * are {@code 1 - dot(a, b)}. Cosine vectors are normalized on insertion and query.
 *
 * <p>Thread-safety: immutable enum constants; inherently thread-safe.
 */
public enum Metric {
    /** Squared Euclidean distance (the square root is intentionally not taken). */
    L2,
    /** Cosine distance; vectors are normalized to unit length before storage and query. */
    COSINE,
    /** Inner-product distance, computed as {@code 1 - dot(a, b)}. */
    INNER_PRODUCT
}
