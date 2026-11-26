package dev.hnswforge.bench;

import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.persist.HnswSerializer;
import dev.hnswforge.persist.IndexSnapshot;
import dev.hnswforge.util.SyntheticData;
import java.nio.file.Files;
import java.nio.file.Path;

/** Measures parallel build scaling and memory-mapped load time. */
public final class Phase5Bench {

    private Phase5Bench() {
    }

    /** Runs the build-scaling measurement at 1, 2, 4 and 8 threads. */
    public static void main(String[] args) throws Exception {
        int n = 100_000;
        float[][] base = SyntheticData.uniform(n, 32, 123L);
        for (int threads : new int[] {1, 2, 4, 8}) {
            HnswParams params = HnswParams.builder(32, Metric.L2)
                    .m(16).efConstruction(200).maxElements(n).seed(42L).build();
            HnswIndex index = new HnswIndex(params);
            long start = System.nanoTime();
            index.addAll(base, threads);
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.printf("threads=%d build_s=%.2f%n", threads, seconds);
        }

        int million = 1_000_000;
        int dimension = 32;
        HnswParams params = HnswParams.builder(dimension, Metric.L2)
                .m(16).efConstruction(200).maxElements(million).seed(42L).build();
        float[] vectors = new float[Math.multiplyExact(million, dimension)];
        int[] levels = new int[million];
        int[] layer0 = new int[Math.multiplyExact(million, 33)];
        IndexSnapshot snapshot = new IndexSnapshot(params, million, 0, 0,
                vectors, levels, layer0, new int[million][]);
        Path path = Files.createTempFile("hnsw-1m", ".bin");
        long start = System.nanoTime();
        HnswSerializer.save(snapshot, path);
        double saveSeconds = (System.nanoTime() - start) / 1e9;
        long bytes = Files.size(path);
        start = System.nanoTime();
        HnswIndex.loadMapped(path).close();
        double mapSeconds = (System.nanoTime() - start) / 1e9;
        Files.deleteIfExists(path);
        System.out.printf("mapped_1m: bytes=%d save_s=%.3f open_s=%.4f%n",
                bytes, saveSeconds, mapSeconds);
    }
}
