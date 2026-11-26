package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.util.SyntheticData;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Tests for filtered search and its low-selectivity fallback. */
class FilteredSearchTest {

    @Test
    void filteredResultsAllSatisfyPredicate() {
        float[][] base = SyntheticData.uniform(1_000, 16, 123L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(16, Metric.L2).m(8).efConstruction(100)
                        .efSearch(100).maxElements(1_000).seed(42L).build());
        float[] query = base[7];
        List<SearchResult> results = index.search(query, 10, 100, id -> id % 10 == 0);
        assertEquals(10, results.size());
        float previous = Float.NEGATIVE_INFINITY;
        for (SearchResult result : results) {
            assertEquals(0, result.id() % 10);
            assertTrue(result.distance() >= previous);
            previous = result.distance();
        }
    }

    @Test
    void lowSelectivityFallbackReturnsExactMatches() {
        float[][] base = SyntheticData.uniform(200, 16, 9L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(16, Metric.L2).m(8).efConstruction(100)
                        .maxElements(200).seed(9L).build());
        List<SearchResult> results = index.search(base[3], 10, 100, id -> id < 5);
        assertEquals(5, results.size());
        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).distance() <= results.get(i).distance());
        }
    }

    @Test
    void nullFilterThrows() {
        HnswIndex index = new HnswIndex(HnswParams.builder(4, Metric.L2).build());
        index.add(new float[4]);
        try {
            index.search(new float[4], 1, 10, null);
            throw new AssertionError("expected NullPointerException");
        } catch (NullPointerException expected) {
            assertTrue(expected.getMessage().contains("filter"));
        }
    }
}
