package dev.hnswforge.bench;

import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.util.SyntheticData;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Small CI smoke test that guards against a broken graph with a recall floor. */
class BenchmarkSmokeTest {

    @Test
    void recallAtTenOnTenThousandVectorsIsAtLeastPointNinetyFive() {
        float[][] base = SyntheticData.uniform(10_000, 32, 123L);
        float[][] queries = SyntheticData.queries(base, 50, 456L);
        HnswParams params = HnswParams.builder(32, Metric.L2)
                .m(16).efConstruction(200).efSearch(100)
                .maxElements(10_000).seed(42L).build();
        HnswIndex index = new HnswIndex(params);
        for (float[] vector : base) {
            index.add(vector);
        }
        BruteForceIndex exact = new BruteForceIndex(32, Metric.L2);
        for (float[] vector : base) {
            exact.add(vector);
        }
        double recall = 0.0;
        for (float[] query : queries) {
            Set<Integer> truth = new HashSet<>();
            for (SearchResult result : exact.search(query, 10)) {
                truth.add(result.id());
            }
            List<SearchResult> actual = index.search(query, 10, 100);
            int hits = 0;
            for (SearchResult result : actual) {
                if (truth.contains(result.id())) {
                    hits++;
                }
            }
            recall += hits / 10.0;
        }
        recall /= queries.length;
        assertTrue(recall >= 0.95, "recall@10 was " + recall);
    }
}
