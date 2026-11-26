package dev.hnswforge.index;

import dev.hnswforge.exact.BruteForceIndex;
import java.util.ArrayList;
import java.util.List;

/** Shared test helpers for HNSW tests. */
final class HnswTestUtil {

    private HnswTestUtil() {
    }

    static HnswIndex build(float[][] vectors, HnswParams params) {
        HnswIndex index = new HnswIndex(params);
        for (float[] vector : vectors) {
            index.add(vector);
        }
        return index;
    }

    static double recallAtK(HnswIndex index, float[][] queries, int k) {
        return recallAtK(index, queries, k, Math.max(k * 2, 100));
    }

    static double recallAtK(HnswIndex index, float[][] queries, int k, int efSearch) {
        BruteForceIndex exact = new BruteForceIndex(index.dimension(), index.metric());
        for (int id = 0; id < index.size(); id++) {
            exact.add(index.vector(id));
        }
        double total = 0.0;
        for (float[] query : queries) {
            List<Integer> truth = new ArrayList<>();
            for (SearchResult result : exact.search(query, k)) {
                truth.add(result.id());
            }
            List<SearchResult> actual = index.search(query, k, efSearch);
            int hits = 0;
            for (SearchResult result : actual) {
                if (truth.contains(result.id())) {
                    hits++;
                }
            }
            total += (double) hits / k;
        }
        return total / queries.length;
    }
}
