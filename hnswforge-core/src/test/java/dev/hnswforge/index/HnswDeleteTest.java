package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.util.SyntheticData;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Tests for tombstone deletion. */
class HnswDeleteTest {

    @Test
    void deleteUpdatesLiveCountButNotSize() {
        HnswIndex index = build(100);
        assertEquals(100, index.size());
        assertEquals(100, index.liveCount());
        assertTrue(index.delete(5));
        assertEquals(100, index.size());
        assertEquals(99, index.liveCount());
        assertTrue(index.isDeleted(5));
    }

    @Test
    void secondDeleteReturnsFalse() {
        HnswIndex index = build(20);
        assertTrue(index.delete(3));
        assertFalse(index.delete(3));
        assertEquals(19, index.liveCount());
    }

    @Test
    void invalidIdThrows() {
        HnswIndex index = build(10);
        assertThrows(IllegalArgumentException.class, () -> index.delete(-1));
        assertThrows(IllegalArgumentException.class, () -> index.delete(10));
    }

    @Test
    void deletedIdsNeverAppearInResults() {
        HnswIndex index = build(500);
        Set<Integer> removed = new HashSet<>();
        for (int id = 0; id < 50; id++) {
            assertTrue(index.delete(id));
            removed.add(id);
        }
        float[][] queries = SyntheticData.queries(SyntheticData.uniform(500, 16, 1L), 50, 2L);
        for (float[] query : queries) {
            for (SearchResult result : index.search(query, 20, 100)) {
                assertFalse(removed.contains(result.id()), "deleted id returned: " + result.id());
            }
        }
    }

    @Test
    void recallOnRemainingSetStaysAbovePointNineAfterTenPercentDeletes() {
        float[][] base = SyntheticData.uniform(2_000, 16, 123L);
        HnswParams params = HnswParams.builder(16, Metric.L2)
                .m(16).efConstruction(200).efSearch(100)
                .maxElements(2_000).seed(42L).build();
        HnswIndex index = HnswTestUtil.build(base, params);
        for (int id = 0; id < 200; id++) {
            index.delete(id);
        }
        BruteForceIndex exact = new BruteForceIndex(16, Metric.L2);
        for (int id = 200; id < index.size(); id++) {
            exact.add(index.vector(id));
        }
        float[][] queries = SyntheticData.queries(base, 100, 456L);
        double recall = 0.0;
        for (float[] query : queries) {
            Set<Integer> truth = new HashSet<>();
            for (SearchResult result : exact.search(query, 10)) {
                truth.add(result.id() + 200);
            }
            int hits = 0;
            List<SearchResult> actual = index.search(query, 10, 100);
            for (SearchResult result : actual) {
                if (truth.contains(result.id())) {
                    hits++;
                }
            }
            recall += hits / 10.0;
        }
        recall /= queries.length;
        assertTrue(recall >= 0.9, "recall after deletes was " + recall);
    }

    private static HnswIndex build(int count) {
        float[][] base = SyntheticData.uniform(count, 16, 7L);
        return HnswTestUtil.build(base,
                HnswParams.builder(16, Metric.L2).m(8).efConstruction(100)
                        .maxElements(count).seed(7L).build());
    }
}
