package dev.hnswforge.persist;

import dev.hnswforge.index.HnswParams;
import java.util.Objects;

/**
 * Immutable point-in-time copy of an HNSW index.
 *
 * <p>The arrays are stored by reference and are never copied by this record. Callers
 * must not mutate them after construction; doing so breaks the immutability contract
 * and may corrupt any index restored from the snapshot. The vectors are flat and
 * ordered by id, {@code levels} holds one level per id, {@code layer0} holds
 * {@code size} records of {@code [degree, neighborId...]} with stride
 * {@code 1 + params.maxM0()}, and {@code upper} has length {@code params.maxElements()}
 * with {@code null} at level-0 ids and {@code int[level * (1 + params.m())]} elsewhere.
 * {@code deleted} holds the internal ids that are tombstoned: every id must lie in
 * {@code [0, size)}, the array must be sorted ascending and ids must be distinct.
 * Tombstoned nodes remain in the graph and are still traversed during search; they
 * are only withheld from returned results.
 *
 * <p>The canonical constructor validates that {@code deleted} is non-null, sorted,
 * distinct and within {@code [0, size)}. The seven-argument constructor is retained
 * for source compatibility and delegates with an empty {@code deleted} array.
 *
 * <p>Thread-safety: immutable once created, provided callers honor the no-mutation
 * contract; safe for concurrent reads.
 *
 * @param params     parameters of the source index
 * @param size       number of vectors, in {@code [0, params.maxElements()]}
 * @param entryPoint entry-point id, or {@code -1} when {@code size == 0}
 * @param maxLevel   highest level present, or {@code -1} when {@code size == 0}
 * @param vectors    flat vector data of length {@code size * params.dimension()}
 * @param levels     level of each id, of length {@code size}
 * @param layer0     layer-0 adjacency of length {@code size * (1 + params.maxM0())}
 * @param upper      upper-layer adjacency of length {@code params.maxElements()}
 * @param deleted    sorted, distinct tombstoned ids, each in {@code [0, size)}
 */
public record IndexSnapshot(
        HnswParams params,
        int size,
        int entryPoint,
        int maxLevel,
        float[] vectors,
        int[] levels,
        int[] layer0,
        int[][] upper,
        int[] deleted) {

    /**
     * Creates a snapshot without tombstones.
     *
     * <p>Equivalent to the canonical constructor with an empty {@code deleted} array.
     *
     * @param params     parameters of the source index
     * @param size       number of vectors
     * @param entryPoint entry-point id
     * @param maxLevel   highest level present
     * @param vectors    flat vector data
     * @param levels     level of each id
     * @param layer0     layer-0 adjacency
     * @param upper      upper-layer adjacency
     */
    public IndexSnapshot(HnswParams params, int size, int entryPoint, int maxLevel,
                         float[] vectors, int[] levels, int[] layer0, int[][] upper) {
        this(params, size, entryPoint, maxLevel, vectors, levels, layer0, upper, new int[0]);
    }

    /**
     * Validates the tombstone ids after assigning all record components.
     *
     * @throws IllegalArgumentException if an id is outside {@code [0, size)} or the
     *                                  ids are not strictly ascending
     * @throws NullPointerException     if {@code deleted} is null
     */
    public IndexSnapshot {
        Objects.requireNonNull(deleted, "deleted must not be null");
        int previous = -1;
        for (int id : deleted) {
            if (id < 0 || id >= size) {
                throw new IllegalArgumentException("deleted id out of range: " + id
                        + ", size " + size);
            }
            if (id <= previous) {
                throw new IllegalArgumentException("deleted ids must be sorted and distinct: "
                        + id + " after " + previous);
            }
            previous = id;
        }
    }
}
