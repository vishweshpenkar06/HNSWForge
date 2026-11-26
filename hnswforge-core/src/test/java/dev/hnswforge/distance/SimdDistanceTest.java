package dev.hnswforge.distance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.hnswforge.index.Metric;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Tests that SIMD kernels agree with the scalar reference kernels. */
class SimdDistanceTest {

    private static final int[] DIMS = {1, 3, 7, 16, 100, 128, 960};

    @Test
    void matchesScalarForEveryMetricAndDimension() {
        requireSimd();
        Random random = new Random(42L);
        for (int dim : DIMS) {
            for (Metric metric : Metric.values()) {
                float[] a = randomVector(random, dim);
                float[] b = randomVector(random, dim);
                assertClose(scalar(metric).distance(a, b), simd(metric).distance(a, b), dim, metric);
            }
        }
    }

    @Test
    void offsetVariantMatchesScalarAndIgnoresPadding() {
        requireSimd();
        Random random = new Random(7L);
        for (int dim : DIMS) {
            for (Metric metric : Metric.values()) {
                float[] a = padded(random, dim, 5);
                float[] b = padded(random, dim, 3);
                assertClose(scalar(metric).distance(a, 5, b, 3, dim),
                        simd(metric).distance(a, 5, b, 3, dim), dim, metric);
            }
        }
    }

    @Test
    void memorySegmentVariantMatchesScalarAndRespectsByteOffset() {
        requireSimd();
        Random random = new Random(11L);
        try (Arena arena = Arena.ofConfined()) {
            for (int dim : DIMS) {
                for (Metric metric : Metric.values()) {
                    float[] a = randomVector(random, dim);
                    float[] b = randomVector(random, dim);
                    long byteOffset = 16L;
                    MemorySegment segment = arena.allocate(byteOffset + (long) dim * Float.BYTES);
                    MemorySegment.copy(b, 0, segment, ValueLayout.JAVA_FLOAT, byteOffset, dim);
                    assertClose(scalar(metric).distance(a, segment, byteOffset, dim),
                            simd(metric).distance(a, segment, byteOffset, dim), dim, metric);
                }
            }
        }
    }

    @Test
    void cosineRejectsZeroVector() {
        requireSimd();
        DistanceFunction cosine = simd(Metric.COSINE);
        assertThrows(IllegalArgumentException.class,
                () -> cosine.distance(new float[] {0f, 0f}, new float[] {1f, 2f}));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(2L * Float.BYTES);
            assertThrows(IllegalArgumentException.class,
                    () -> cosine.distance(new float[] {1f, 2f}, segment, 0L, 2));
        }
    }

    @Test
    void dimensionMismatchThrows() {
        requireSimd();
        DistanceFunction l2 = simd(Metric.L2);
        assertThrows(IllegalArgumentException.class,
                () -> l2.distance(new float[] {1f}, new float[] {1f, 2f}));
        assertThrows(IllegalArgumentException.class,
                () -> l2.distance(new float[] {1f, 2f}, 0, new float[] {1f}, 0, 2));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(2L * Float.BYTES);
            assertThrows(IllegalArgumentException.class,
                    () -> l2.distance(new float[] {1f, 2f}, segment, 0L, 1));
        }
    }

    private static void requireSimd() {
        Assumptions.assumeTrue(DistanceFunctions.simdAvailable(), "SIMD module unavailable");
    }

    private static DistanceFunction simd(Metric metric) {
        return DistanceFunctions.forMetric(metric, true);
    }

    private static DistanceFunction scalar(Metric metric) {
        return ScalarDistance.forMetric(metric);
    }

    private static void assertClose(float expected, float actual, int dim, Metric metric) {
        float delta = Math.max(1e-6f, 1e-4f * Math.abs(expected));
        assertEquals(expected, actual, delta,
                () -> metric + " dim=" + dim + " expected=" + expected + " actual=" + actual);
    }

    private static float[] randomVector(Random random, int dim) {
        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = random.nextFloat() * 4f - 2f;
        }
        return vector;
    }

    private static float[] padded(Random random, int dim, int offset) {
        float[] vector = new float[dim + offset + 3];
        Arrays.fill(vector, 1234f);
        for (int i = 0; i < dim; i++) {
            vector[offset + i] = random.nextFloat() * 4f - 2f;
        }
        return vector;
    }
}
