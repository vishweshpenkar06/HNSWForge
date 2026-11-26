package dev.hnswforge.persist;

import java.io.IOException;

/**
 * Reports that a serialized HNSW index is malformed, truncated, or internally
 * inconsistent.
 *
 * <p>Thrown for a wrong magic, an unsupported version or endianness, a CRC
 * mismatch, invalid dimensions or metrics, inconsistent sizes, and invalid section
 * offsets. The message identifies the violated constraint.
 *
 * <p>Thread-safety: immutable; inherently thread-safe.
 */
public final class IndexFormatException extends IOException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a diagnostic message.
     *
     * @param message description of the violated format constraint
     */
    public IndexFormatException(String message) {
        super(message);
    }

    /**
     * Creates an exception with a diagnostic message and an underlying cause.
     *
     * @param message description of the violated format constraint
     * @param cause   underlying failure, for example an arithmetic overflow
     */
    public IndexFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
