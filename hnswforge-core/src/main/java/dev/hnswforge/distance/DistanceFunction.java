package dev.hnswforge.distance;

import java.lang.foreign.MemorySegment;

/**
 * Computes a distance between a query vector and a stored vector.
 *
 * <p>Two overloads are required: one for heap arrays and one for a vector stored
 * in a {@link MemorySegment} at a given byte offset (used by off-heap and mapped
 * storage). Both return a smaller-is-closer distance as documented in
 * {@code dev.hnswforge.index.Metric}.
 *
 * <p>Thread-safety: implementations must be stateless and safe for concurrent use.
 */
public interface DistanceFunction {

    /**
     * Computes the distance between two heap arrays of equal length.
     *
     * @param a first vector
     * @param b second vector
     * @return the distance, smaller is closer
     * @throws IllegalArgumentException if lengths differ or values are NaN
     */
    float distance(float[] a, float[] b);

    /**
     * Computes the distance between a heap array and a vector in a memory segment.
     *
     * @param a          first vector, of length {@code dim}
     * @param segment    segment containing the second vector
     * @param byteOffset byte offset of the second vector within the segment
     * @param dim        number of float values in the second vector
     * @return the distance, smaller is closer
     * @throws IllegalArgumentException if {@code dim} differs from {@code a.length},
     *                                  is non-positive, or values are NaN
     */
    float distance(float[] a, MemorySegment segment, long byteOffset, int dim);
}
