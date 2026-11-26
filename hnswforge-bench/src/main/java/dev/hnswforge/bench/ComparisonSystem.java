package dev.hnswforge.bench;

import dev.hnswforge.index.SearchResult;
import java.util.List;

/**
 * Common query contract implemented by every comparison system.
 *
 * <p>Contract: {@code setEfSearch} establishes the effort used by subsequent
 * {@code search} calls; implementations that cannot change effort accept and ignore
 * the value. {@code indexBytesEstimate} returns a non-negative estimate or {@code -1}
 * when the footprint is unknown.
 *
 * <p>Thread-safety: implementation-specific; the harness uses one thread per system.
 */
interface ComparisonSystem extends AutoCloseable {
    List<SearchResult> search(float[] query, int k);

    void setEfSearch(int ef);

    long indexBytesEstimate();

    @Override
    void close();
}
