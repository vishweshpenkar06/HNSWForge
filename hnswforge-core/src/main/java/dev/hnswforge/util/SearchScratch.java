package dev.hnswforge.util;

import java.util.Arrays;

/**
 * Reusable per-search buffers: a query vector, heaps, visited stamps and scratch
 * arrays for neighbor selection.
 *
 * <p>Reusing these buffers keeps steady-state search allocation-free apart from
 * the returned result list. Each thread must own its own instance.
 *
 * <p>Thread-safety: not thread-safe; pool instances with {@code ThreadLocal}.
 */
public final class SearchScratch {
    /** Query buffer of length {@code dimension}. */
    public final float[] query;
    /** Candidate frontier heap. */
    public final MinHeap candidates;
    /** Bounded results heap. */
    public final BoundedMaxHeap results;
    /** Epoch-stamped visited set sized for the whole index. */
    public final VisitedList visited;
    /** Single-entry buffer for search entry points. */
    public final int[] entryBuffer;
    /** Buffer for selected neighbor ids. */
    public final int[] chosen;
    /** Buffer for combined neighbor ids during prune. */
    public final int[] combinedIds;
    /** Distances corresponding to {@link #combinedIds}. */
    public final float[] combinedDistances;
    /** Buffer for batching unvisited neighbors before distance computation. */
    public final int[] neighborBatch;
    /** Sorted ids found by the last layer search. */
    public int[] foundIds;
    /** Distances corresponding to {@link #foundIds}. */
    public float[] foundDistances;

    /** Creates scratch buffers sized for the given index and search parameters. */
    public SearchScratch(int dimension, int capacity, int maxEf, int maxDegree) {
        if (dimension <= 0 || capacity <= 0 || maxEf <= 0 || maxDegree <= 0) {
            throw new IllegalArgumentException("all sizes must be positive");
        }
        this.query = new float[dimension];
        this.candidates = new MinHeap();
        this.results = new BoundedMaxHeap(maxEf);
        this.visited = new VisitedList(capacity);
        this.foundIds = new int[maxEf];
        this.foundDistances = new float[maxEf];
        this.entryBuffer = new int[1];
        this.chosen = new int[maxDegree];
        this.combinedIds = new int[maxDegree + 1];
        this.combinedDistances = new float[maxDegree + 1];
        this.neighborBatch = new int[maxDegree + 1];
    }

    /** Ensures the found-result buffers can hold {@code capacity} entries. */
    public void ensureFound(int capacity) {
        if (capacity > foundIds.length) {
            int newLength = Math.max(capacity, foundIds.length * 2);
            foundIds = Arrays.copyOf(foundIds, newLength);
            foundDistances = Arrays.copyOf(foundDistances, newLength);
        }
    }
}
