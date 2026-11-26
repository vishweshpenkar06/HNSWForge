package dev.hnswforge.repoprism;

/**
 * A retrieved chunk paired with its similarity score.
 *
 * <p>Scores are higher-is-better similarities in {@code [0, 1]} for cosine-backed
 * implementations. Instances are value objects and can be shared freely.
 *
 * <p>Thread-safety: immutable record; inherently thread-safe.
 *
 * @param chunkId non-blank chunk identifier
 * @param score   finite similarity score, must not be NaN
 */
public record ScoredChunk(String chunkId, float score) {

    /** Validates the identifier and score. */
    public ScoredChunk {
        if (chunkId == null || chunkId.isBlank()) {
            throw new IllegalArgumentException("chunkId must not be null or blank");
        }
        if (Float.isNaN(score)) {
            throw new IllegalArgumentException("score must not be NaN");
        }
    }
}
