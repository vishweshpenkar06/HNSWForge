package dev.hnswforge.storage;

/**
 * Selects the backing storage for vectors in an index.
 *
 * <p>{@link #HEAP} keeps vectors in a contiguous {@code float[]};
 * {@link #OFF_HEAP} keeps them in a shared arena-allocated segment.
 *
 * <p>Thread-safety: immutable enum constants; inherently thread-safe.
 */
public enum StorageMode {
    /** Store vectors in a contiguous heap {@code float[]}. */
    HEAP,
    /** Store vectors in a shared off-heap {@link java.lang.foreign.MemorySegment}. */
    OFF_HEAP
}
