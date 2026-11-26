package dev.hnswforge.distance;

import dev.hnswforge.index.Metric;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Scalar reference implementations of all distance kernels.
 *
 * <p>Each metric is computed as a smaller-is-closer distance. Cosine distance
 * normalizes both operands, so it is invariant to scaling and rejects zero vectors.
 * This class is the correctness oracle for the optimized SIMD kernels.
 *
 * <p>Thread-safety: stateless and immutable; safe for concurrent use.
 */
public final class ScalarDistance implements DistanceFunction {

    /** Squared Euclidean distance kernel. */
    public static final ScalarDistance L2 = new ScalarDistance(Metric.L2);
    /** Inner-product distance kernel ({@code 1 - dot}). */
    public static final ScalarDistance INNER_PRODUCT = new ScalarDistance(Metric.INNER_PRODUCT);
    /** Cosine distance kernel (normalize, then {@code 1 - dot}). */
    public static final ScalarDistance COSINE = new ScalarDistance(Metric.COSINE);

    private final Metric metric;

    private ScalarDistance(Metric metric) {
        this.metric = metric;
    }

    /** Returns the scalar kernel for a metric. */
    public static DistanceFunction forMetric(Metric metric) {
        Objects.requireNonNull(metric, "metric must not be null");
        return switch (metric) {
            case L2 -> L2;
            case INNER_PRODUCT -> INNER_PRODUCT;
            case COSINE -> COSINE;
        };
    }

    /** @return the metric implemented by this kernel */
    public Metric metric() {
        return metric;
    }

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

    @Override
    public float distance(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        Objects.requireNonNull(a, "a must not be null");
        Objects.requireNonNull(b, "b must not be null");
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be > 0, got " + dim);
        }
        if (aOffset < 0 || bOffset < 0 || aOffset + dim > a.length || bOffset + dim > b.length) {
            throw new IllegalArgumentException("offset/dim outside array bounds");
        }
        return switch (metric) {
            case L2 -> (float) squaredL2(a, aOffset, b, bOffset, dim);
            case INNER_PRODUCT -> 1.0f - (float) dot(a, aOffset, b, bOffset, dim);
            case COSINE -> cosine(a, aOffset, b, bOffset, dim);
        };
    }

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
        if (byteOffset < 0 || byteOffset + (long) dim * Float.BYTES > segment.byteSize()) {
            throw new IllegalArgumentException(
                    "byte offset " + byteOffset + " with " + dim
                            + " floats is outside a segment of " + segment.byteSize() + " bytes");
        }
        return switch (metric) {
            case L2 -> (float) squaredL2(segment, a, byteOffset, dim);
            case INNER_PRODUCT -> 1.0f - (float) dot(segment, a, byteOffset, dim);
            case COSINE -> cosine(segment, a, byteOffset, dim);
        };
    }

    private float cosine(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        double normA = norm(a, aOffset, dim);
        double normB = norm(b, bOffset, dim);
        if (normA == 0.0 || normB == 0.0) {
            throw new IllegalArgumentException("cosine distance is undefined for a zero vector");
        }
        return (float) (1.0 - dot(a, aOffset, b, bOffset, dim) / (normA * normB));
    }

    private float cosine(MemorySegment segment, float[] a, long byteOffset, int dim) {
        double normA = norm(a, 0, dim);
        double normB = norm(segment, byteOffset, dim);
        if (normA == 0.0 || normB == 0.0) {
            throw new IllegalArgumentException("cosine distance is undefined for a zero vector");
        }
        return (float) (1.0 - dot(segment, a, byteOffset, dim) / (normA * normB));
    }

    private static double norm(float[] a, int offset, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            float value = requireFinite(a[offset + i]);
            sum += (double) value * value;
        }
        return Math.sqrt(sum);
    }

    private static double norm(MemorySegment segment, long byteOffset, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            float value = requireFinite(read(segment, byteOffset, i));
            sum += (double) value * value;
        }
        return Math.sqrt(sum);
    }

    private static double dot(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += (double) requireFinite(a[aOffset + i]) * requireFinite(b[bOffset + i]);
        }
        return sum;
    }

    private static double dot(MemorySegment segment, float[] a, long byteOffset, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += (double) read(segment, byteOffset, i) * requireFinite(a[i]);
        }
        return sum;
    }

    private static double squaredL2(float[] a, int aOffset, float[] b, int bOffset, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double difference = (double) requireFinite(a[aOffset + i])
                    - requireFinite(b[bOffset + i]);
            sum += difference * difference;
        }
        return sum;
    }

    private static double squaredL2(MemorySegment segment, float[] a, long byteOffset, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double difference = (double) read(segment, byteOffset, i) - requireFinite(a[i]);
            sum += difference * difference;
        }
        return sum;
    }

    private static float read(MemorySegment segment, long byteOffset, int index) {
        return segment.get(ValueLayout.JAVA_FLOAT, byteOffset + (long) index * Float.BYTES);
    }

    private static float requireFinite(float value) {
        if (Float.isNaN(value)) {
            throw new IllegalArgumentException("vector values must not be NaN");
        }
        return value;
    }
}
