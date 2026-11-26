package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.util.SyntheticData;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

class HnswParameterizedTest {

    @ParameterizedTest
    @MethodSource("searchConfigurations")
    void searchIsSortedUniqueAndAccurate(Metric metric, NeighborSelection selection, int k) {
        int dimension = 8;
        int count = 200;
        float[][] base = SyntheticData.uniform(count, dimension, 1234L);
        float[][] queries = SyntheticData.queries(base, 3, 5678L);
        HnswParams params = HnswParams.builder(dimension, metric)
                .m(8).efConstruction(80).efSearch(100)
                .maxElements(count).seed(42L)
                .neighborSelection(selection).build();
        try (HnswIndex index = HnswTestUtil.build(base, params)) {
            for (float[] query : queries) {
                List<SearchResult> results = index.search(query, k);
                assertTrue(results.size() <= k);
                float previous = Float.NEGATIVE_INFINITY;
                Set<Integer> ids = new HashSet<>();
                for (SearchResult result : results) {
                    assertTrue(result.distance() >= previous);
                    previous = result.distance();
                    assertTrue(ids.add(result.id()));
                    assertTrue(result.id() >= 0 && result.id() < count);
                }
            }
            double recall = HnswTestUtil.recallAtK(index, queries, k, 100);
            assertTrue(recall >= 0.8, metric + "/" + selection + "/k=" + k + " recall " + recall);
        }
    }

    @ParameterizedTest
    @EnumSource(Metric.class)
    void deleteUpdatesLiveCount(Metric metric) {
        int dimension = 4;
        float[][] base = SyntheticData.uniform(20, dimension, 99L);
        HnswParams params = HnswParams.builder(dimension, metric)
                .m(4).efConstruction(40).efSearch(40)
                .maxElements(base.length).seed(7L).build();
        try (HnswIndex index = HnswTestUtil.build(base, params)) {
            assertEquals(20, index.liveCount());
            assertTrue(index.delete(5));
            assertEquals(19, index.liveCount());
            assertFalse(index.delete(5));
            assertEquals(19, index.liveCount());
            for (SearchResult result : index.search(base[5], 20)) {
                assertFalse(result.id() == 5);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 4})
    void addAllInsertsEveryVector(int threads) {
        int dimension = 4;
        float[][] base = SyntheticData.uniform(30, dimension, 314L);
        HnswParams params = HnswParams.builder(dimension, Metric.L2)
                .m(8).efConstruction(80).efSearch(50)
                .maxElements(base.length).seed(11L).build();
        try (HnswIndex index = new HnswIndex(params)) {
            index.addAll(base, threads);
            assertEquals(base.length, index.size());
            assertEquals(base.length, index.liveCount());
            List<SearchResult> results = index.search(base[5], 1);
            assertEquals(1, results.size());
            assertTrue(results.get(0).id() >= 0 && results.get(0).id() < base.length);
        }
    }

    private static Stream<Arguments> searchConfigurations() {
        List<Arguments> cases = new ArrayList<>();
        for (Metric metric : new Metric[] {Metric.L2, Metric.COSINE}) {
            for (NeighborSelection selection : NeighborSelection.values()) {
                for (int k : new int[] {1, 3, 5, 10}) {
                    cases.add(Arguments.of(metric, selection, k));
                }
            }
        }
        return cases.stream();
    }
}
