package dev.hnswforge.repoprism;

import java.util.Objects;

/**
 * A chunk identifier and the embedding to index for it.
 *
 * <p>The embedding is defensively copied in the compact constructor, so later
 * mutation of the caller's array cannot change this value.
 *
 * <p>Thread-safety: immutable record (apart from the internal defensive copy);
 * inherently thread-safe.
 *
 * @param chunkId   non-blank chunk identifier
 * @param embedding non-null embedding of the backend's configured dimension
 */
public record Chunk(String chunkId, float[] embedding) {

    /** Validates the identifier and defensively copies the embedding. */
    public Chunk {
        if (chunkId == null || chunkId.isBlank()) {
            throw new IllegalArgumentException("chunkId must not be null or blank");
        }
        Objects.requireNonNull(embedding, "embedding must not be null");
        embedding = embedding.clone();
    }
}
