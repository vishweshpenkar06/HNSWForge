package dev.hnswforge.repoprism;

import java.util.List;

/**
 * Contract for the dense vector leg of a retrieval pipeline.
 *
 * <p>Implementations associate opaque string chunk identifiers with embeddings and
 * return at most {@code k} scored chunk identifiers for a query embedding. Scores
 * are higher-is-better similarities. Every implementation must reject null inputs,
 * malformed embeddings and non-positive {@code k} with
 * {@link IllegalArgumentException}, and must document its own concurrency rules.
 *
 * <p>Thread-safety: implementation-specific; callers must consult the concrete class.
 */
public interface VectorSearchBackend {

    /**
     * Inserts a chunk embedding, replacing the previous embedding when the chunk is
     * already present.
     *
     * <p>Thread-safety: implementation-specific.
     *
     * @param chunkId   non-blank chunk identifier without tab or newline characters
     * @param embedding non-null embedding whose length matches the backend dimension
     * @throws IllegalArgumentException if either argument is invalid
     */
    void upsert(String chunkId, float[] embedding);

    /**
     * Returns the closest chunks for a query embedding.
     *
     * <p>Thread-safety: implementation-specific.
     *
     * @param queryEmbedding non-null embedding whose length matches the backend dimension
     * @param k              maximum number of results, must be positive
     * @return results sorted by descending score, never null and never longer than {@code k}
     * @throws IllegalArgumentException if the query is invalid or {@code k} is not positive
     */
    List<ScoredChunk> search(float[] queryEmbedding, int k);

    /**
     * Replaces all current state with the supplied chunks, remapping internal ids from
     * scratch. The previous state is discarded only after the replacement is built
     * successfully.
     *
     * <p>Thread-safety: implementation-specific.
     *
     * @param chunks non-null iterable of non-null, uniquely identified chunks
     * @throws IllegalArgumentException if the iterable, a chunk, an embedding or a
     *                                  chunk identifier is invalid or duplicated
     */
    void rebuild(Iterable<Chunk> chunks);

    /**
     * Persists the index and the chunk-id map to the configured storage location.
     * Any runtime failure is reported to the caller and never silently ignored.
     *
     * <p>Thread-safety: implementation-specific.
     *
     * @throws IllegalStateException if the backend has been closed
     */
    void persist();
}
