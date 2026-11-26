package dev.hnswforge.distance;

import dev.hnswforge.index.Metric;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD distance kernels implemented with {@link FloatVector} and the preferred species.
 *
 * <p>Each metric mirrors {@link ScalarDistance}: {@code L2} is squared Euclidean
 * distance, {@code INNER_PRODUCT} is {@code 1 - dot}, and {@code COSINE} normalizes
 * both operands before computing {@code 1 - dot}. Lanes are accumulated with fused
 * multiply-add, reduced with {@link VectorOperators#ADD}, and finished with a scalar
 * tail for dimensions not divisible by the species length.
 *
 * <p>Thread-safety: stateless and immutable; safe for concurrent use.
 */
public final class SimdDistance implements DistanceFunction {

    /** Squared Euclidean distance kernel. */
    public static final SimdDistance L2 = new SimdDistance(Metric.L2);
    /** Inner-product distance kernel ({@code 1 - dot}). */
    public static final SimdDistance INNER_PRODUCT = new SimdDistance(Metric.INNER_PRODUCT);
    /** Cosine distance kernel (normalize, then {@code 1 - dot}). */
    public static final SimdDistance COSINE = new SimdDistance(Metric.COSINE);

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final Metric metric;

    private SimdDistance(Metric metric) {
        this.metric = metric;
    }

    /**
     * Returns the SIMD kernel for a metric.
     *
     * <p>Thread-safe: immutable singleton lookup.
     *
     * @param metric metric to compute
     * @return the matching SIMD kernel
     * @throws NullPointerException if {@code metric} is null
     */
    public static DistanceFunction forMetric(Metric metric) {
        Objects.requireNonNull(metric, "metric must not be null");
        return switch (metric) {
            case L2 -> L2;
            case INNER_PRODUCT -> INNER_PRODUCT;
            case COSINE -> COSINE;
        };
    }

    /**
     * Computes the distance between two full heap arrays.
     *
     * <p>Thread-safe: reads only its arguments.
     *
     * @throws IllegalArgumentException if lengths differ or values are NaN
     */
    @Override
    public float distance(float[] a, float[] b) {
        Objects.requireNonNull(a, "a must not be null");
        Objects.requireNonNull(b, "b must not be null");
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "dimension mismatch: " + a.length + " vs " + b.length);
        }
        return distance(a, 0, b, 0, a.length);
    }

    /**
     * Computes the distance between two array slices without reading outside them.
     *
     * <p>Thread-safe: reads only its arguments.
     *
     * @throws IllegalArgumentException if offsets or dimension are invalid, or values are NaN
     */
    @Override
    public float distance(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        Objects.requireNonNull(a, "a must not be null");
        Objects.requireNonNull(b, "b must not be null");
        validate(dim, aOffset, a.length, bOffset, b.length);
        return switch (metric) {
            case L2, INNER_PRODUCT -> combine(a, aOffset, b, bOffset, dim);
            case COSINE -> cosine(a, aOffset, b, bOffset, dim);
        };
    }

    /**
     * Computes the distance between a heap array and a vector inside a memory segment.
     *
     * <p>Thread-safe: reads only its arguments.
     *
     * @throws IllegalArgumentException if inputs are invalid or values are NaN
     */
    @Override
    public float distance(float[] a, MemorySegment segment, long byteOffset, int dim) {
        Objects.requireNonNull(a, "a must not be null");
        Objects.requireNonNull(segment, "segment must not be null");
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be > 0, got " + dim);
        }
        if (a.length != dim) {
            throw new IllegalArgumentException("dimension mismatch: " + a.length + " vs " + dim);
        }
        if (byteOffset < 0 || byteOffset % Float.BYTES != 0
                || byteOffset + (long) dim * Float.BYTES > segment.byteSize()) {
            throw new IllegalArgumentException("byte offset " + byteOffset + " with " + dim
                    + " floats is outside a segment of " + segment.byteSize() + " bytes");
        }
        return switch (metric) {
            case L2, INNER_PRODUCT -> combine(a, segment, byteOffset, dim);
            case COSINE -> cosine(a, segment, byteOffset, dim);
        };
    }

    private static void validate(int dim, int aOffset, int aLength, int bOffset, int bLength) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be > 0, got " + dim);
        }
        if (aOffset < 0 || bOffset < 0
                || aOffset + dim > aLength || bOffset + dim > bLength) {
            throw new IllegalArgumentException("offset/dim outside array bounds");
        }
    }

    private float combine(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(dim);
        int index = 0;
        for (; index < limit; index += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOffset + index);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOffset + index);
            requireFinite(va);
            requireFinite(vb);
            acc = accumulate(acc, va, vb);
        }
        double tail = 0.0;
        for (; index < dim; index++) {
            float x = requireFinite(a[aOffset + index]);
            float y = requireFinite(b[bOffset + index]);
            tail = accumulateTail(tail, x, y);
        }
        return finish((double) acc.reduceLanes(VectorOperators.ADD) + tail);
    }

    private float combine(float[] a, MemorySegment segment, long byteOffset, int dim) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(dim);
        int index = 0;
        for (; index < limit; index += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, index);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, segment,
                    byteOffset + (long) index * Float.BYTES, ByteOrder.nativeOrder());
            requireFinite(va);
            requireFinite(vb);
            acc = accumulate(acc, va, vb);
        }
        double tail = 0.0;
        for (; index < dim; index++) {
            float x = requireFinite(a[index]);
            float y = requireFinite(segment.get(ValueLayout.JAVA_FLOAT,
                    byteOffset + (long) index * Float.BYTES));
            tail = accumulateTail(tail, x, y);
        }
        return finish((double) acc.reduceLanes(VectorOperators.ADD) + tail);
    }

    private FloatVector accumulate(FloatVector acc, FloatVector va, FloatVector vb) {
        if (metric == Metric.L2) {
            FloatVector diff = va.sub(vb);
            return diff.fma(diff, acc);
        }
        return va.fma(vb, acc);
    }

    private double accumulateTail(double acc, float x, float y) {
        if (metric == Metric.L2) {
            double diff = (double) x - (double) y;
            return acc + diff * diff;
        }
        return acc + (double) x * (double) y;
    }

    private float finish(double sum) {
        return metric == Metric.L2 ? (float) sum : 1.0f - (float) sum;
    }

    private float cosine(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        double normA = norm(a, aOffset, dim);
        double normB = norm(b, bOffset, dim);
        requireNonZero(normA, normB);
        float scaleA = (float) normA;
        float scaleB = (float) normB;
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(dim);
        int index = 0;
        for (; index < limit; index += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOffset + index);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOffset + index);
            acc = va.div(scaleA).fma(vb.div(scaleB), acc);
        }
        double dot = acc.reduceLanes(VectorOperators.ADD);
        for (; index < dim; index++) {
            dot += (double) (a[aOffset + index] / scaleA) * (b[bOffset + index] / scaleB);
        }
        return (float) (1.0 - dot);
    }

    private float cosine(float[] a, MemorySegment segment, long byteOffset, int dim) {
        double normA = norm(a, 0, dim);
        double normB = norm(segment, byteOffset, dim);
        requireNonZero(normA, normB);
        float scaleA = (float) normA;
        float scaleB = (float) normB;
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(dim);
        int index = 0;
        for (; index < limit; index += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, index);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, segment,
                    byteOffset + (long) index * Float.BYTES, ByteOrder.nativeOrder());
            acc = va.div(scaleA).fma(vb.div(scaleB), acc);
        }
        double dot = acc.reduceLanes(VectorOperators.ADD);
        for (; index < dim; index++) {
            float y = segment.get(ValueLayout.JAVA_FLOAT,
                    byteOffset + (long) index * Float.BYTES);
            dot += (double) (a[index] / scaleA) * (y / scaleB);
        }
        return (float) (1.0 - dot);
    }

    private static double norm(float[] a, int offset, int dim) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(dim);
        int index = 0;
        for (; index < limit; index += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, offset + index);
            requireFinite(va);
            acc = va.fma(va, acc);
        }
        double sum = acc.reduceLanes(VectorOperators.ADD);
        for (; index < dim; index++) {
            float value = requireFinite(a[offset + index]);
            sum += (double) value * value;
        }
        return Math.sqrt(sum);
    }

    private static double norm(MemorySegment segment, long byteOffset, int dim) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(dim);
        int index = 0;
        for (; index < limit; index += SPECIES.length()) {
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, segment,
                    byteOffset + (long) index * Float.BYTES, ByteOrder.nativeOrder());
            requireFinite(vb);
            acc = vb.fma(vb, acc);
        }
        double sum = acc.reduceLanes(VectorOperators.ADD);
        for (; index < dim; index++) {
            float value = requireFinite(segment.get(ValueLayout.JAVA_FLOAT,
                    byteOffset + (long) index * Float.BYTES));
            sum += (double) value * value;
        }
        return Math.sqrt(sum);
    }

    private static void requireFinite(FloatVector vector) {
        if (vector.test(VectorOperators.IS_NAN).anyTrue()) {
            throw new IllegalArgumentException("vector values must not be NaN");
        }
    }

    private static float requireFinite(float value) {
        if (Float.isNaN(value)) {
            throw new IllegalArgumentException("vector values must not be NaN");
        }
        return value;
    }

    private static void requireNonZero(double normA, double normB) {
        if (normA == 0.0 || normB == 0.0) {
            throw new IllegalArgumentException("cosine distance is undefined for a zero vector");
        }
    }
}
