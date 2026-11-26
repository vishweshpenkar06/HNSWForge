package dev.hnswforge.index;

/**
 * Strategy used to choose neighbors when connecting a new node during insertion.
 *
 * <p>Thread-safety: immutable enum constants; inherently thread-safe.
 */
public enum NeighborSelection {
    /** Keep the {@code M} geometrically closest candidates. */
    SIMPLE,
    /** Keep neighbors that are diverse (the HNSW Algorithm 4 heuristic). */
    HEURISTIC
}
