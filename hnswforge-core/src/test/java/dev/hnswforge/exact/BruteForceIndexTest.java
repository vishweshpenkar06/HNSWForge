package dev.hnswforge.exact;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Tests for the exact brute-force index. */
class BruteForceIndexTest {

    @Test
    void addAssignsSequentialIds() {
        BruteForceIndex index = new BruteForceIndex(2, Metric.L2);
        assertEquals(0, index.add(new float[] {0, 0}));
        assertEquals(1, index.add(new float[] {1, 1}));
        assertEquals(2, index.size());
    }

    @Test
    void searchReturnsResultsSortedByDistance() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        index.add(new float[] {10});
        index.add(new float[] {1});
        index.add(new float[] {5});
        List<SearchResult> results = index.search(new float[] {0}, 3);
        assertEquals(List.of(new SearchResult(1, 1f), new SearchResult(2, 25f),
                new SearchResult(0, 100f)), results);
    }

    @Test
    void kLargerThanSizeReturnsAllElements() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        index.add(new float[] {1});
        index.add(new float[] {2});
        assertEquals(2, index.search(new float[] {0}, 100).size());
    }

    @Test
    void kEqualsOneReturnsClosest() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        index.add(new float[] {5});
        index.add(new float[] {1});
        assertEquals(1, index.search(new float[] {0}, 1).get(0).id());
    }

    @Test
    void tiesAreBrokenByLowerId() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        index.add(new float[] {-1});
        index.add(new float[] {1});
        index.add(new float[] {2});
        List<SearchResult> results = index.search(new float[] {0}, 3);
        assertEquals(0, results.get(0).id());
        assertEquals(1, results.get(1).id());
    }

    @Test
    void emptyIndexReturnsNoResults() {
        BruteForceIndex index = new BruteForceIndex(3, Metric.L2);
        assertTrue(index.search(new float[] {1, 2, 3}, 5).isEmpty());
    }

    @Test
    void invalidKThrows() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[] {1}, 0));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[] {1}, -1));
    }

    @Test
    void dimensionMismatchThrows() {
        BruteForceIndex index = new BruteForceIndex(2, Metric.L2);
        assertThrows(IllegalArgumentException.class, () -> index.add(new float[] {1}));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[] {1}, 1));
    }

    @Test
    void nanThrows() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        assertThrows(IllegalArgumentException.class, () -> index.add(new float[] {Float.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> index.search(new float[] {Float.NaN}, 1));
    }

    @Test
    void cosineRejectsZeroVector() {
        BruteForceIndex index = new BruteForceIndex(2, Metric.COSINE);
        assertThrows(IllegalArgumentException.class, () -> index.add(new float[] {0, 0}));
    }

    @Test
    void cosineIsScalingInvariant() {
        BruteForceIndex index = new BruteForceIndex(2, Metric.COSINE);
        index.add(new float[] {1, 0});
        index.add(new float[] {0, 5});
        List<SearchResult> a = index.search(new float[] {1, 0}, 2);
        List<SearchResult> b = index.search(new float[] {1000, 0}, 2);
        assertEquals(a, b);
    }

    @Test
    void innerProductMetricIsUsed() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.INNER_PRODUCT);
        index.add(new float[] {2});
        assertEquals(-1.0f, index.search(new float[] {1}, 1).get(0).distance(), 1e-6f);
    }

    @Test
    void vectorsAreCopiedOnInsert() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        float[] vector = {1};
        index.add(vector);
        vector[0] = 100;
        assertEquals(1.0f, index.search(new float[] {0}, 1).get(0).distance(), 1e-6f);
    }

    @Test
    void saveIsUnsupported() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        assertThrows(UnsupportedOperationException.class,
                () -> index.save(Path.of("ignored.bin")));
    }

    @Test
    void closeCanBeCalledTwice() {
        BruteForceIndex index = new BruteForceIndex(1, Metric.L2);
        index.close();
        index.close();
    }

    @Test
    void metricAndDimensionAreExposed() {
        BruteForceIndex index = new BruteForceIndex(7, Metric.COSINE);
        assertEquals(7, index.dimension());
        assertEquals(Metric.COSINE, index.metric());
        assertNotEquals(Metric.L2, index.metric());
    }
}
