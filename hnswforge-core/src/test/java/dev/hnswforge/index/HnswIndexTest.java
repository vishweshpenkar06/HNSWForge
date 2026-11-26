package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.util.SyntheticData;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Edge cases and input validation for {@link HnswIndex}. */
class HnswIndexTest {

    @Test
    void emptyIndexReturnsNoResults() {
        HnswIndex index = new HnswIndex(HnswParams.builder(4, Metric.L2).build());
        assertTrue(index.search(new float[] {1, 2, 3, 4}, 10).isEmpty());
        assertEquals(0, index.size());
    }

    @Test
    void singleElementIndexReturnsThatElement() {
        HnswIndex index = new HnswIndex(HnswParams.builder(2, Metric.L2).build());
        index.add(new float[] {1, 2});
        List<SearchResult> results = index.search(new float[] {0, 0}, 10);
        assertEquals(1, results.size());
        assertEquals(0, results.get(0).id());
    }

    @Test
    void kLargerThanSizeReturnsAllElements() {
        float[][] vectors = SyntheticData.clusters(100, 4, 5, 1L);
        HnswIndex index = HnswTestUtil.build(vectors,
                HnswParams.builder(4, Metric.L2).maxElements(100).build());
        assertEquals(100, index.search(new float[] {0, 0, 0, 0}, 500).size());
    }

    @Test
    void duplicateVectorsAreHandled() {
        HnswIndex index = new HnswIndex(HnswParams.builder(2, Metric.L2).build());
        index.add(new float[] {1, 1});
        index.add(new float[] {1, 1});
        index.add(new float[] {1, 1});
        List<SearchResult> results = index.search(new float[] {1, 1}, 3);
        assertEquals(3, results.size());
        assertEquals(0.0f, results.get(0).distance(), 1e-6f);
    }

    @Test
    void dimensionMismatchThrows() {
        HnswIndex index = new HnswIndex(HnswParams.builder(3, Metric.L2).build());
        assertThrows(IllegalArgumentException.class, () -> index.add(new float[] {1, 2}));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[] {1, 2}, 1));
    }

    @Test
    void nanVectorThrows() {
        HnswIndex index = new HnswIndex(HnswParams.builder(2, Metric.L2).build());
        assertThrows(IllegalArgumentException.class,
                () -> index.add(new float[] {Float.NaN, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> index.search(new float[] {Float.NaN, 1}, 1));
    }

    @Test
    void zeroVectorThrowsForCosine() {
        HnswIndex index = new HnswIndex(HnswParams.builder(2, Metric.COSINE).build());
        assertThrows(IllegalArgumentException.class, () -> index.add(new float[] {0, 0}));
    }

    @Test
    void invalidKAndEfThrow() {
        HnswIndex index = new HnswIndex(HnswParams.builder(2, Metric.L2).build());
        index.add(new float[] {1, 1});
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[] {1, 1}, 0));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[] {1, 1}, 1, 0));
    }

    @Test
    void nullVectorThrows() {
        HnswIndex index = new HnswIndex(HnswParams.builder(2, Metric.L2).build());
        assertThrows(NullPointerException.class, () -> index.add(null));
    }

    @Test
    void capacityIsEnforcedWithClearError() {
        HnswIndex index = new HnswIndex(HnswParams.builder(1, Metric.L2)
                .maxElements(1).build());
        index.add(new float[] {1});
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> index.add(new float[] {2}));
        assertTrue(error.getMessage().contains("capacity"));
    }

    @Test
    void dimensionMetricAndParamsAreExposed() {
        HnswParams params = HnswParams.builder(5, Metric.INNER_PRODUCT).m(4)
                .efConstruction(40).build();
        HnswIndex index = new HnswIndex(params);
        assertEquals(5, index.dimension());
        assertEquals(Metric.INNER_PRODUCT, index.metric());
        assertEquals(params, index.params());
    }

    @Test
    void saveAndLoadRoundTripPreservesSearchResults() throws Exception {
        float[][] base = SyntheticData.uniform(100, 4, 9L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(4, Metric.L2).m(8).efConstruction(100)
                        .maxElements(100).seed(9L).build());
        java.nio.file.Path path = java.nio.file.Files.createTempFile("hnsw", ".bin");
        try {
            index.save(path);
            HnswIndex loaded = HnswIndex.load(path);
            assertEquals(index.size(), loaded.size());
            assertEquals(index.metric(), loaded.metric());
            float[] query = base[0];
            assertEquals(index.search(query, 5), loaded.search(query, 5));
            loaded.close();
        } finally {
            java.nio.file.Files.deleteIfExists(path);
        }
    }
}
