package dev.hnswforge.distance;

import java.lang.foreign.MemorySegment;

/**
 * Computes a distance between a query vector and a stored vector.
 *
 * <p>Overloads support heap arrays, offset-based heap arrays (for flat vector stores),
 * and vectors in a {@link MemorySegment} (for off-heap and mapped storage). All
 * return a smaller-is-closer distance as documented in {@code dev.hnswforge.index.Metric}.
 *
 * <p>Thread-safety: implementations must be stateless and safe for concurrent use.
 */
public interface DistanceFunction {

    /**
     * Computes the distance between two heap arrays.
     *
     * @param a first vector
     * @param b second vector
     * @return the distance, smaller is closer
     * @throws IllegalArgumentException if lengths differ or values are NaN
     */
    float distance(float[] a, float[] b);

    /**
     * Computes the distance between two slices of heap arrays.
     *
     * @param a       first array
     * @param aOffset float index of the first vector in {@code a}
     * @param b       second array
     * @param bOffset float index of the second vector in {@code b}
     * @param dim     number of float values in each vector
     * @return the distance, smaller is closer
     * @throws IllegalArgumentException if offsets or dimension are invalid, or values are NaN
     */
    float distance(float[] a, int aOffset, float[] b, int bOffset, int dim);

    /**
     * Computes the distance between a heap array and a vector in a memory segment.
     *
     * @param a          first vector of length {@code dim}
     * @param segment    segment containing the second vector
     * @param byteOffset byte offset of the second vector
     * @param dim        number of float values in the second vector
     * @return the distance, smaller is closer
     * @throws IllegalArgumentException if inputs are invalid
     */
    float distance(float[] a, MemorySegment segment, long byteOffset, int dim);
}
