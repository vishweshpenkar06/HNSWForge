package dev.hnswforge.persist;

import dev.hnswforge.index.HnswParams;

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
 */
public record IndexSnapshot(
        HnswParams params,
        int size,
        int entryPoint,
        int maxLevel,
        float[] vectors,
        int[] levels,
        int[] layer0,
        int[][] upper) {
}
