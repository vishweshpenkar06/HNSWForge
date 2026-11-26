package dev.hnswforge.distance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.hnswforge.index.Metric;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/** Tests for the scalar reference distance kernels. */
class ScalarDistanceTest {

    @Test
    void l2ComputesSquaredEuclideanDistance() {
        assertEquals(25.0f,
                ScalarDistance.L2.distance(new float[] {1, 2}, new float[] {4, 6}), 1e-6f);
    }

    @Test
    void l2OfIdenticalVectorIsZero() {
        float[] vector = {1.5f, -2f, 3f};
        assertEquals(0.0f, ScalarDistance.L2.distance(vector, vector), 1e-6f);
    }

    @Test
    void l2IsSymmetric() {
        float[] a = {1, 2, 3};
        float[] b = {4, -1, 2};
        assertEquals(ScalarDistance.L2.distance(a, b), ScalarDistance.L2.distance(b, a), 1e-6f);
    }

    @Test
    void innerProductIsOneMinusDot() {
        assertEquals(-10.0f,
                ScalarDistance.INNER_PRODUCT.distance(new float[] {1, 2}, new float[] {3, 4}),
                1e-6f);
    }

    @Test
    void cosineOfOrthogonalVectorsIsOne() {
        assertEquals(1.0f,
                ScalarDistance.COSINE.distance(new float[] {1, 0}, new float[] {0, 1}), 1e-6f);
    }

    @Test
    void cosineIsInvariantToScaling() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        float[] scaledB = {40, 50, 60};
        assertEquals(ScalarDistance.COSINE.distance(a, b),
                ScalarDistance.COSINE.distance(a, scaledB), 1e-6f);
    }

    @Test
    void cosineOfIdenticalDirectionIsZero() {
        assertEquals(0.0f,
                ScalarDistance.COSINE.distance(new float[] {2, 0}, new float[] {5, 0}), 1e-6f);
    }

    @Test
    void cosineRejectsZeroVector() {
        assertThrows(IllegalArgumentException.class,
                () -> ScalarDistance.COSINE.distance(new float[] {0, 0}, new float[] {1, 2}));
    }

    @Test
    void dimensionMismatchThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> ScalarDistance.L2.distance(new float[] {1}, new float[] {1, 2}));
    }

    @Test
    void nanThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> ScalarDistance.L2.distance(new float[] {Float.NaN}, new float[] {1}));
    }

    @Test
    void forMetricReturnsExpectedKernel() {
        assertEquals(ScalarDistance.L2, ScalarDistance.forMetric(Metric.L2));
        assertEquals(ScalarDistance.INNER_PRODUCT, ScalarDistance.forMetric(Metric.INNER_PRODUCT));
        assertEquals(ScalarDistance.COSINE, ScalarDistance.forMetric(Metric.COSINE));
    }

    @Test
    void memorySegmentDistanceMatchesArrayDistance() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, -1f, 2f};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) b.length * Float.BYTES);
            for (int i = 0; i < b.length; i++) {
                segment.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, b[i]);
            }
            assertEquals(ScalarDistance.L2.distance(a, b),
                    ScalarDistance.L2.distance(a, segment, 0, b.length), 1e-6f);
            assertEquals(ScalarDistance.INNER_PRODUCT.distance(a, b),
                    ScalarDistance.INNER_PRODUCT.distance(a, segment, 0, b.length), 1e-6f);
            assertEquals(ScalarDistance.COSINE.distance(a, b),
                    ScalarDistance.COSINE.distance(a, segment, 0, b.length), 1e-6f);
        }
    }

    @Test
    void memorySegmentDistanceRespectsByteOffset() {
        float[] a = {1f, 2f};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(4L * Float.BYTES);
            segment.set(ValueLayout.JAVA_FLOAT, 0, 99f);
            segment.set(ValueLayout.JAVA_FLOAT, Float.BYTES, 3f);
            segment.set(ValueLayout.JAVA_FLOAT, 2L * Float.BYTES, 4f);
            assertEquals(ScalarDistance.L2.distance(a, new float[] {3f, 4f}),
                    ScalarDistance.L2.distance(a, segment, Float.BYTES, 2), 1e-6f);
        }
    }

    @Test
    void memorySegmentBoundsAreValidated() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) Float.BYTES);
            assertThrows(IllegalArgumentException.class,
                    () -> ScalarDistance.L2.distance(new float[] {1f}, segment, Float.BYTES, 1));
        }
    }
}
