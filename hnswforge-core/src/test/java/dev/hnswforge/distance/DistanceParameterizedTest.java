package dev.hnswforge.distance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.hnswforge.index.Metric;
import dev.hnswforge.util.SyntheticData;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

class DistanceParameterizedTest {

    private static final int[] DIMENSIONS = {1, 2, 3, 7, 16, 32, 64, 128};

    @ParameterizedTest
    @MethodSource("squaredMetricDimPairs")
    void squaredDistanceIsSymmetric(Metric metric, int dimension) {
        float[][] vectors = SyntheticData.uniform(2, dimension, dimension * 13L);
        DistanceFunction kernel = ScalarDistance.forMetric(metric);
        float forward = kernel.distance(vectors[0], vectors[1]);
        float backward = kernel.distance(vectors[1], vectors[0]);
        assertEquals(forward, backward, 1e-4f);
    }

    @ParameterizedTest
    @MethodSource("offsetSamples")
    void offsetOverloadMatchesFullDistance(Metric metric, int dimension) {
        float[][] vectors = SyntheticData.uniform(2, dimension, dimension * 17L);
        float[] first = vectors[0];
        float[] second = vectors[1];
        float[] paddedFirst = new float[dimension + 4];
        float[] paddedSecond = new float[dimension + 4];
        java.util.Arrays.fill(paddedFirst, 123f);
        java.util.Arrays.fill(paddedSecond, -77f);
        System.arraycopy(first, 0, paddedFirst, 2, dimension);
        System.arraycopy(second, 0, paddedSecond, 3, dimension);
        DistanceFunction kernel = ScalarDistance.forMetric(metric);
        float expected = kernel.distance(first, second);
        float actual = kernel.distance(paddedFirst, 2, paddedSecond, 3, dimension);
        assertEquals(expected, actual, 1e-4f);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 16, 128})
    void cosineRejectsZeroVector(int dimension) {
        float[] zero = new float[dimension];
        float[] other = SyntheticData.uniform(1, dimension, 7L)[0];
        assertThrows(IllegalArgumentException.class,
                () -> ScalarDistance.COSINE.distance(zero, other));
        assertThrows(IllegalArgumentException.class,
                () -> ScalarDistance.COSINE.distance(other, zero));
    }

    @ParameterizedTest
    @EnumSource(Metric.class)
    void nanVectorIsRejected(Metric metric) {
        float[] first = {1f, 2f, Float.NaN, 4f};
        float[] second = {1f, 2f, 3f, 4f};
        DistanceFunction kernel = ScalarDistance.forMetric(metric);
        assertThrows(IllegalArgumentException.class, () -> kernel.distance(first, second));
    }

    private static Stream<Arguments> squaredMetricDimPairs() {
        List<Arguments> pairs = new ArrayList<>();
        for (Metric metric : new Metric[] {Metric.L2, Metric.INNER_PRODUCT}) {
            for (int dimension : DIMENSIONS) {
                pairs.add(Arguments.of(metric, dimension));
            }
        }
        return pairs.stream();
    }

    private static Stream<Arguments> offsetSamples() {
        return Stream.of(
                Arguments.of(Metric.L2, 1),
                Arguments.of(Metric.L2, 7),
                Arguments.of(Metric.L2, 128),
                Arguments.of(Metric.INNER_PRODUCT, 3),
                Arguments.of(Metric.COSINE, 16),
                Arguments.of(Metric.COSINE, 32));
    }
}
