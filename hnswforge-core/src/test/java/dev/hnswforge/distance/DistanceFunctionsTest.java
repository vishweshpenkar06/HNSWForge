package dev.hnswforge.distance;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.index.Metric;
import org.junit.jupiter.api.Test;

/** Tests runtime selection between SIMD and scalar distance kernels. */
class DistanceFunctionsTest {

    @Test
    void simdAvailableInThisEnvironment() {
        assertTrue(DistanceFunctions.simdAvailable());
    }

    @Test
    void forMetricChoosesExpectedImplementation() {
        assertSame(SimdDistance.L2, DistanceFunctions.forMetric(Metric.L2, true));
        assertSame(SimdDistance.INNER_PRODUCT,
                DistanceFunctions.forMetric(Metric.INNER_PRODUCT, true));
        assertSame(SimdDistance.COSINE, DistanceFunctions.forMetric(Metric.COSINE, true));

        assertSame(ScalarDistance.L2, DistanceFunctions.forMetric(Metric.L2, false));
        assertSame(ScalarDistance.INNER_PRODUCT,
                DistanceFunctions.forMetric(Metric.INNER_PRODUCT, false));
        assertSame(ScalarDistance.COSINE, DistanceFunctions.forMetric(Metric.COSINE, false));
    }
}
