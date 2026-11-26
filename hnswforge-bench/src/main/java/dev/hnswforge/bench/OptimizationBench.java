package dev.hnswforge.bench;

import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.storage.StorageMode;
import dev.hnswforge.util.SyntheticData;
import java.util.HashSet;
import java.util.Set;

/**
 * Command-line harness that measures build time, query throughput and recall for
 * one combination of Phase 4 optimizations.
 */
public final class OptimizationBench {

    private OptimizationBench() {
    }

    /**
     * Runs one measurement.
     *
     * @param args {@code label n dim preferSimd offHeap prefetch efSearch}
     * @throws Exception if a number cannot be parsed
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 7) {
            throw new IllegalArgumentException(
                    "usage: OptimizationBench label n dim preferSimd offHeap prefetch efSearch");
        }
        String label = args[0];
        int n = Integer.parseInt(args[1]);
        int dim = Integer.parseInt(args[2]);
        boolean preferSimd = Boolean.parseBoolean(args[3]);
        boolean offHeap = Boolean.parseBoolean(args[4]);
        boolean prefetch = Boolean.parseBoolean(args[5]);
        int efSearch = Integer.parseInt(args[6]);

        float[][] base = SyntheticData.uniform(n, dim, 123L);
        float[][] queries = SyntheticData.queries(base, 200, 456L);
        HnswParams params = HnswParams.builder(dim, Metric.L2)
                .m(16).efConstruction(200).efSearch(efSearch)
                .maxElements(n).seed(42L).build();

        long start = System.nanoTime();
        HnswIndex index = new HnswIndex(params, preferSimd,
                offHeap ? StorageMode.OFF_HEAP : StorageMode.HEAP, prefetch);
        for (float[] vector : base) {
            index.add(vector);
        }
        double buildSeconds = (System.nanoTime() - start) / 1e9;

        for (int pass = 0; pass < 3; pass++) {
            for (float[] query : queries) {
                index.search(query, 10, efSearch);
            }
        }
        int passes = 5;
        start = System.nanoTime();
        for (int pass = 0; pass < passes; pass++) {
            for (float[] query : queries) {
                index.search(query, 10, efSearch);
            }
        }
        double qps = passes * queries.length / ((System.nanoTime() - start) / 1e9);

        BruteForceIndex exact = new BruteForceIndex(dim, Metric.L2);
        for (float[] vector : base) {
            exact.add(vector);
        }
        double recall = 0.0;
        for (float[] query : queries) {
            Set<Integer> truth = new HashSet<>();
            for (SearchResult result : exact.search(query, 10)) {
                truth.add(result.id());
            }
            int hits = 0;
            for (SearchResult result : index.search(query, 10, efSearch)) {
                if (truth.contains(result.id())) {
                    hits++;
                }
            }
            recall += hits / 10.0;
        }
        recall /= queries.length;

        System.out.printf(
                "%s,n=%d,dim=%d,build_s=%.2f,qps=%.0f,recall=%.4f%n",
                label, n, dim, buildSeconds, qps, recall);
        index.close();
    }
}
