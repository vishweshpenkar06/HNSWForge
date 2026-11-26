package dev.hnswforge.storage;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.index.Metric;
import dev.hnswforge.util.SyntheticData;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

class VectorStoreParameterizedTest {

    @ParameterizedTest
    @MethodSource("dimensionsAndModes")
    void setCopyAndDistanceAgree(int dimension, boolean offHeap) {
        DistanceFunction kernel = ScalarDistance.forMetric(Metric.L2);
        float[][] vectors = SyntheticData.uniform(2, dimension, dimension * 29L);
        float[] first = vectors[0];
        float[] second = vectors[1];
        try (VectorStore store = offHeap
                ? new OffHeapVectorStore(dimension, 4, kernel)
                : new HeapVectorStore(dimension, 4, kernel)) {
            store.set(0, first);
            store.set(1, second);
            assertEquals(dimension, store.dimension());
            assertEquals(4, store.capacity());
            assertEquals(2, store.size());
            assertArrayEquals(first, store.copyOf(0), 1e-6f);
            assertArrayEquals(second, store.copyOf(1), 1e-6f);
            assertEquals(0f, store.dist(first, 0), 1e-5f);
            assertEquals(kernel.distance(first, second), store.dist(first, 1), 1e-4f);
            assertEquals(kernel.distance(first, second), store.dist(0, 1), 1e-4f);
            assertEquals(0f, store.dist(0, 0), 1e-5f);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 3, 8, 16, 32})
    void sizeAndCapacityTrackSets(int dimension) {
        DistanceFunction kernel = ScalarDistance.forMetric(Metric.L2);
        float[][] vectors = SyntheticData.uniform(5, dimension, dimension * 31L);
        try (HeapVectorStore store = new HeapVectorStore(dimension, 10, kernel)) {
            assertEquals(dimension, store.dimension());
            assertEquals(10, store.capacity());
            assertEquals(0, store.size());
            store.set(0, vectors[0]);
            store.set(4, vectors[4]);
            assertEquals(5, store.size());
            assertArrayEquals(vectors[4], store.copyOf(4), 1e-6f);
        }
    }

    private static Stream<Arguments> dimensionsAndModes() {
        List<Arguments> cases = new ArrayList<>();
        for (int dimension : new int[] {1, 3, 8, 16, 32}) {
            cases.add(Arguments.of(dimension, false));
            cases.add(Arguments.of(dimension, true));
        }
        return cases.stream();
    }
}
