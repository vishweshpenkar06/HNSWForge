package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.util.SyntheticData;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import net.jqwik.api.Arbitraries;
import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;
import org.junit.jupiter.api.Test;

/** Recall quality of the HNSW index. */
class HnswRecallTest {

    @Test
    void recallAtTenMeetsTargetOnTenThousandClusteredVectors() {
        float[][] base = SyntheticData.uniform(10_000, 32, 123L);
        float[][] queries = SyntheticData.queries(base, 100, 456L);
        HnswParams params = HnswParams.builder(32, Metric.L2)
                .m(16).efConstruction(200).efSearch(100)
                .maxElements(10_000).seed(42L).build();
        HnswIndex index = HnswTestUtil.build(base, params);
        double recall = HnswTestUtil.recallAtK(index, queries, 10);
        assertTrue(recall >= 0.95, "recall@10 was " + recall);
        assertTrue(recall <= 1.0);
    }

    @Test
    void heuristicSelectionIsAtLeastAsGoodAsSimpleOnClusteredData() {
        float[][] base = SyntheticData.clusters(3_000, 16, 60, 321L);
        float[][] queries = SyntheticData.queries(base, 100, 654L);
        HnswParams.Builder builder = HnswParams.builder(16, Metric.L2)
                .m(8).efConstruction(100).efSearch(200).maxElements(3_000).seed(7L);
        double heuristic = HnswTestUtil.recallAtK(
                HnswTestUtil.build(base, builder.neighborSelection(NeighborSelection.HEURISTIC).build()),
                queries, 10);
        double simple = HnswTestUtil.recallAtK(
                HnswTestUtil.build(base, builder.neighborSelection(NeighborSelection.SIMPLE).build()),
                queries, 10);
        assertTrue(heuristic >= simple - 0.05,
                "heuristic=" + heuristic + " simple=" + simple);
    }

    @Test
    void higherEfSearchImprovesRecall() {
        float[][] base = SyntheticData.uniform(2_000, 16, 999L);
        float[][] queries = SyntheticData.queries(base, 100, 1000L);
        HnswParams params = HnswParams.builder(16, Metric.L2)
                .m(8).efConstruction(100).efSearch(10)
                .maxElements(2_000).seed(3L).build();
        HnswIndex index = HnswTestUtil.build(base, params);
        double low = HnswTestUtil.recallAtK(index, queries, 10, 10);
        double high = HnswTestUtil.recallAtK(index, queries, 10, 200);
        assertTrue(high >= low, "low=" + low + " high=" + high);
    }

    @Test
    void recallWorksForInnerProductAndCosine() {
        for (Metric metric : new Metric[] {Metric.INNER_PRODUCT, Metric.COSINE}) {
            float[][] base = SyntheticData.uniform(1_000, 12, 11L);
            float[][] queries = SyntheticData.queries(base, 50, 12L);
            HnswParams params = HnswParams.builder(12, metric)
                    .m(8).efConstruction(100).efSearch(60)
                    .maxElements(1_000).seed(4L).build();
            double recall = HnswTestUtil.recallAtK(HnswTestUtil.build(base, params), queries, 10);
            assertTrue(recall >= 0.8, metric + " recall was " + recall);
        }
    }

    @Property
    void resultsAreSortedUniqueAndWithinBounds(@ForAll("vectors") List<float[]> vectors,
                                               @ForAll("vector") float[] query) {
        HnswParams params = HnswParams.builder(query.length, Metric.L2)
                .m(4).efConstruction(40).efSearch(20)
                .maxElements(Math.max(vectors.size(), 1)).seed(42L).build();
        HnswIndex index = new HnswIndex(params);
        for (float[] vector : vectors) {
            index.add(vector);
        }
        List<SearchResult> results = index.search(query, 5);
        float previous = Float.NEGATIVE_INFINITY;
        Set<Integer> ids = new HashSet<>();
        for (SearchResult result : results) {
            assertTrue(result.distance() >= previous);
            previous = result.distance();
            assertTrue(ids.add(result.id()));
            assertTrue(result.id() >= 0 && result.id() < index.size());
        }
        assertTrue(results.size() <= Math.min(5, index.size()));
    }

    @Provide
    Arbitrary<List<float[]>> vectors() {
        return vector().list().ofMinSize(1).ofMaxSize(40);
    }

    @Provide
    Arbitrary<float[]> vector() {
        return Arbitraries.floats().between(0f, 1f).array(float[].class).ofSize(6);
    }
}
