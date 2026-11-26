package dev.hnswforge.persist;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.util.BoundedMaxHeap;
import dev.hnswforge.util.SyntheticData;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Search, read-only contract and lifecycle tests for {@link MappedHnswIndex}. */
class MappedHnswIndexTest {

    private static final int DIMENSION = 8;

    @TempDir
    Path tempDir;

    @Test
    void searchRecallsTheExactNearestNeighbors() throws Exception {
        float[][] base = SyntheticData.uniform(200, DIMENSION, 123L);
        int[] levels = levelsFor(base.length);
        Path file = save(base, levels, Metric.L2);
        float[][] queries = SyntheticData.queries(base, 6, 777L);
        BruteForceIndex exact = new BruteForceIndex(DIMENSION, Metric.L2);
        for (float[] vector : base) {
            exact.add(vector);
        }

        try (MappedHnswIndex index = MappedHnswIndex.open(file)) {
            assertEquals(base.length, index.size());
            assertEquals(DIMENSION, index.dimension());
            assertEquals(Metric.L2, index.metric());
            for (float[] query : queries) {
                List<SearchResult> expected = exact.search(query, 5);
                List<SearchResult> actual = index.search(query, 5, 100);
                assertEquals(5, actual.size());
                assertEquals(1.0, recall(expected, actual), "query recall was below 1.0");
                float previous = Float.NEGATIVE_INFINITY;
                Set<Integer> ids = new HashSet<>();
                for (SearchResult result : actual) {
                    assertTrue(result.distance() >= previous);
                    assertTrue(ids.add(result.id()));
                    previous = result.distance();
                }
            }
        }
    }

    @Test
    void defaultSearchEffortFindsTheClosestVector() throws Exception {
        float[][] base = SyntheticData.uniform(120, DIMENSION, 55L);
        Path file = save(base, levelsFor(base.length), Metric.L2);

        try (MappedHnswIndex index = MappedHnswIndex.open(file)) {
            List<SearchResult> results = index.search(base[37], 1);
            assertEquals(1, results.size());
            assertEquals(37, results.get(0).id());
            assertEquals(0.0f, results.get(0).distance());
        }
    }

    @Test
    void deletedIdsNeverAppearInSearchResults() throws Exception {
        float[][] base = SyntheticData.uniform(120, DIMENSION, 55L);
        int[] deleted = {37, 38};
        Path file = save(base, levelsFor(base.length), Metric.L2, deleted);
        DistanceFunction distance = ScalarDistance.forMetric(Metric.L2);
        BoundedMaxHeap exact = new BoundedMaxHeap(5);
        for (int id = 0; id < base.length; id++) {
            if (id != 37 && id != 38) {
                exact.push(id, distance.distance(base[37], base[id]));
            }
        }
        Set<Integer> expected = new HashSet<>();
        while (exact.size() > 0) {
            expected.add(exact.pop());
        }

        try (MappedHnswIndex index = MappedHnswIndex.open(file)) {
            List<SearchResult> results = index.search(base[37], 5, 100);
            assertEquals(5, results.size());
            Set<Integer> actual = new HashSet<>();
            for (SearchResult result : results) {
                assertTrue(actual.add(result.id()));
            }
            assertEquals(expected, actual);
        }
    }

    @Test
    void addAndSaveAreUnsupported() throws Exception {
        float[][] base = SyntheticData.uniform(40, DIMENSION, 12L);
        Path file = save(base, levelsFor(base.length), Metric.L2);

        try (MappedHnswIndex index = MappedHnswIndex.open(file)) {
            UnsupportedOperationException addError = assertThrows(
                    UnsupportedOperationException.class,
                    () -> index.add(new float[DIMENSION]));
            assertEquals("mapped index is read-only", addError.getMessage());
            UnsupportedOperationException saveError = assertThrows(
                    UnsupportedOperationException.class,
                    () -> index.save(tempDir.resolve("other.bin")));
            assertEquals("mapped index is read-only", saveError.getMessage());
        }
    }

    @Test
    void closeMakesAllOperationsThrow() throws Exception {
        float[][] base = SyntheticData.uniform(40, DIMENSION, 13L);
        Path file = save(base, levelsFor(base.length), Metric.L2);
        MappedHnswIndex index = MappedHnswIndex.open(file);
        float[] query = base[0];

        index.close();
        index.close();
        assertThrows(IllegalStateException.class, () -> index.search(query, 3));
        assertThrows(IllegalStateException.class, () -> index.search(query, 3, 20));
        assertThrows(IllegalStateException.class, index::size);
        assertThrows(IllegalStateException.class, index::dimension);
        assertThrows(IllegalStateException.class, index::metric);
        assertThrows(IllegalStateException.class, () -> index.add(query));
    }

    @Test
    void concurrentSearchesAreSafe() throws Exception {
        float[][] base = SyntheticData.uniform(150, DIMENSION, 99L);
        Path file = save(base, levelsFor(base.length), Metric.L2);
        float[][] queries = SyntheticData.queries(base, 8, 100L);

        try (MappedHnswIndex index = MappedHnswIndex.open(file)) {
            ExecutorService pool = Executors.newFixedThreadPool(4);
            try {
                List<Future<?>> futures = new ArrayList<>();
                for (int worker = 0; worker < 4; worker++) {
                    futures.add(pool.submit(() -> {
                        for (int repeat = 0; repeat < 20; repeat++) {
                            for (float[] query : queries) {
                                index.search(query, 5, 30);
                            }
                        }
                        return null;
                    }));
                }
                for (Future<?> future : futures) {
                    future.get(30, TimeUnit.SECONDS);
                }
            } finally {
                pool.shutdownNow();
            }
        }
    }

    @Test
    void openRejectsCorruptedSections() throws Exception {
        float[][] base = SyntheticData.uniform(40, DIMENSION, 14L);
        Path file = save(base, levelsFor(base.length), Metric.L2);
        byte[] bytes = Files.readAllBytes(file);
        bytes[HnswSerializer.HEADER_SIZE + 7] ^= 0x01;
        Path corrupt = tempDir.resolve("corrupt.bin");
        Files.write(corrupt, bytes);

        assertThrows(IndexFormatException.class, () -> MappedHnswIndex.open(corrupt));
    }

    private Path save(float[][] base, int[] levels, Metric metric) throws IOException {
        return save(base, levels, metric, new int[0]);
    }

    private Path save(float[][] base, int[] levels, Metric metric, int[] deleted)
            throws IOException {
        IndexSnapshot snapshot = snapshot(base, levels, metric, deleted);
        Path file = tempDir.resolve("graph-" + metric + "-" + base.length
                + "-" + Arrays.hashCode(deleted) + ".bin");
        HnswSerializer.save(snapshot, file);
        return file;
    }

    private static int[] levelsFor(int size) {
        int[] levels = new int[size];
        for (int id = 0; id < size; id++) {
            levels[id] = id % 17 == 0 ? 2 : id % 5 == 0 ? 1 : 0;
        }
        return levels;
    }

    private static IndexSnapshot snapshot(float[][] base, int[] levels, Metric metric) {
        return snapshot(base, levels, metric, new int[0]);
    }

    private static IndexSnapshot snapshot(float[][] base, int[] levels, Metric metric,
                                          int[] deleted) {
        int size = base.length;
        int m = 6;
        int maxElements = size + 16;
        HnswParams params = HnswParams.builder(DIMENSION, metric)
                .m(m).efConstruction(64).efSearch(32)
                .maxElements(maxElements).seed(42L).build();
        float[] vectors = new float[size * DIMENSION];
        for (int id = 0; id < size; id++) {
            System.arraycopy(base[id], 0, vectors, id * DIMENSION, DIMENSION);
        }
        DistanceFunction distance = ScalarDistance.forMetric(metric);
        int layer0Stride = 1 + params.maxM0();
        int[] layer0 = new int[size * layer0Stride];
        for (int id = 0; id < size; id++) {
            BoundedMaxHeap nearest = new BoundedMaxHeap(params.maxM0());
            for (int other = 0; other < size; other++) {
                if (other != id) {
                    nearest.push(other, distance.distance(base[id], base[other]));
                }
            }
            int offset = id * layer0Stride;
            int count = nearest.size();
            layer0[offset] = count;
            for (int index = count - 1; index >= 0; index--) {
                layer0[offset + 1 + index] = nearest.pop();
            }
        }
        int upperStride = 1 + m;
        int[][] upper = new int[maxElements][];
        int entryPoint = 0;
        int maxLevel = 0;
        for (int id = 0; id < size; id++) {
            if (levels[id] > maxLevel) {
                maxLevel = levels[id];
                entryPoint = id;
            }
        }
        for (int id = 0; id < size; id++) {
            int level = levels[id];
            if (level == 0) {
                continue;
            }
            upper[id] = new int[level * upperStride];
            for (int layer = 1; layer <= level; layer++) {
                BoundedMaxHeap nearest = new BoundedMaxHeap(m);
                for (int other = 0; other < size; other++) {
                    if (other != id && levels[other] >= layer) {
                        nearest.push(other, distance.distance(base[id], base[other]));
                    }
                }
                int offset = (layer - 1) * upperStride;
                int count = nearest.size();
                upper[id][offset] = count;
                for (int index = count - 1; index >= 0; index--) {
                    upper[id][offset + 1 + index] = nearest.pop();
                }
            }
        }
        return new IndexSnapshot(params, size, entryPoint, maxLevel, vectors, levels,
                layer0, upper, deleted);
    }

    private static double recall(List<SearchResult> expected, List<SearchResult> actual) {
        Set<Integer> truth = new HashSet<>();
        for (SearchResult result : expected) {
            truth.add(result.id());
        }
        int hits = 0;
        for (SearchResult result : actual) {
            if (truth.contains(result.id())) {
                hits++;
            }
        }
        return (double) hits / expected.size();
    }
}
