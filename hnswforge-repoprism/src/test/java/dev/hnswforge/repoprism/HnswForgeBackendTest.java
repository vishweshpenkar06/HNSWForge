package dev.hnswforge.repoprism;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests the HNSWForge adapter's id mapping, normalization, persistence and locking.
 *
 * <p>Each test uses a fresh temporary directory supplied by JUnit.
 */
class HnswForgeBackendTest {

    private static final int DIMENSION = 16;

    @TempDir
    Path tempDir;

    /** A string-id round trip returns the nearest chunk with a higher score. */
    @Test
    void upsertAndSearchRoundTripWithStringIds() {
        try (HnswForgeBackend backend = new HnswForgeBackend(tempDir, DIMENSION)) {
            backend.upsert("alpha", oneHot(0));
            backend.upsert("beta", oneHot(1));
            List<ScoredChunk> results = backend.search(oneHot(0), 2);
            assertEquals(2, results.size());
            assertEquals("alpha", results.get(0).chunkId());
            assertTrue(results.get(0).score() > results.get(1).score());
            assertEquals(2, backend.liveCount());
        }
    }

    /** Tombstoned internal ids from a re-upsert are never returned or duplicated. */
    @Test
    void searchSkipsInternalIdsWithoutACurrentChunkMapping() {
        try (HnswForgeBackend backend = new HnswForgeBackend(tempDir, DIMENSION)) {
            backend.upsert("alpha", oneHot(0));
            backend.upsert("alpha", oneHot(3));
            List<ScoredChunk> results = backend.search(oneHot(3), 6);
            assertEquals(1, results.size());
            assertEquals("alpha", results.get(0).chunkId());
            assertEquals(1.0f, results.get(0).score(), 1.0e-5f);
        }
    }

    /** Re-upserting a chunk changes which chunk is nearest to the old and new vectors. */
    @Test
    void reUpsertReplacesTheVector() {
        try (HnswForgeBackend backend = new HnswForgeBackend(tempDir, DIMENSION)) {
            backend.upsert("alpha", oneHot(0));
            backend.upsert("gamma", combination(0, 1));
            backend.upsert("alpha", oneHot(5));
            assertEquals("gamma", backend.search(oneHot(0), 2).get(0).chunkId());
            assertEquals("alpha", backend.search(oneHot(5), 2).get(0).chunkId());
        }
    }

    /** Persist and load preserve both the id map and the search results. */
    @Test
    void persistAndLoadPreserveIdMapAndResults() throws Exception {
        Path directory = tempDir.resolve("saved-index");
        try (HnswForgeBackend backend = new HnswForgeBackend(directory, DIMENSION)) {
            backend.upsert("alpha", oneHot(0));
            backend.upsert("beta", oneHot(1));
            backend.upsert("alpha", oneHot(2));
            backend.persist();
        }
        try (HnswForgeBackend loaded = HnswForgeBackend.load(directory, DIMENSION)) {
            assertEquals(2, loaded.liveCount());
            List<ScoredChunk> results = loaded.search(oneHot(2), 6);
            assertEquals(2, results.size());
            assertEquals("alpha", results.get(0).chunkId());
            assertEquals(1.0f, results.get(0).score(), 1.0e-5f);
            loaded.upsert("beta", oneHot(4));
            assertEquals("beta", loaded.search(oneHot(4), 2).get(0).chunkId());
        }
    }

    /** Concurrent writers and readers share the index without corrupting the id maps. */
    @Test
    void concurrentUpsertsAndSearchesAreSafe() throws Exception {
        int writers = 4;
        int perWriter = 25;
        ExecutorService pool = Executors.newFixedThreadPool(writers + 2);
        ConcurrentLinkedQueue<Throwable> failures = new ConcurrentLinkedQueue<>();
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        try (HnswForgeBackend backend = new HnswForgeBackend(tempDir, DIMENSION)) {
            for (int writer = 0; writer < writers; writer++) {
                int worker = writer;
                futures.add(pool.submit(() -> {
                    try {
                        start.await();
                        for (int i = 0; i < perWriter; i++) {
                            backend.upsert("key-" + worker + "-" + i, oneHot(i % DIMENSION));
                        }
                    } catch (Throwable failure) {
                        failures.add(failure);
                    }
                }));
            }
            for (int reader = 0; reader < 2; reader++) {
                futures.add(pool.submit(() -> {
                    try {
                        start.await();
                        for (int i = 0; i < 100; i++) {
                            List<ScoredChunk> results = backend.search(oneHot(i % DIMENSION), 6);
                            if (results.size() > 6) {
                                throw new AssertionError("too many results: " + results.size());
                            }
                        }
                    } catch (Throwable failure) {
                        failures.add(failure);
                    }
                }));
            }
            start.countDown();
            for (Future<?> future : futures) {
                future.get(30, TimeUnit.SECONDS);
            }
            assertTrue(failures.isEmpty(), () -> "concurrent failures: " + failures);
            assertEquals(writers * perWriter, backend.liveCount());
        } finally {
            pool.shutdownNow();
        }
    }

    /** Invalid embeddings and non-positive result counts are rejected. */
    @Test
    void invalidInputsAreRejected() {
        try (HnswForgeBackend backend = new HnswForgeBackend(tempDir, DIMENSION)) {
            assertThrows(IllegalArgumentException.class, () -> backend.upsert("bad", new float[2]));
            assertThrows(IllegalArgumentException.class, () -> backend.upsert("bad", new float[DIMENSION]));
            assertThrows(IllegalArgumentException.class, () -> backend.search(oneHot(0), 0));
            assertThrows(IllegalArgumentException.class, () -> backend.upsert("\t", oneHot(0)));
        }
    }

    private static float[] oneHot(int index) {
        float[] vector = new float[DIMENSION];
        vector[index] = 1.0f;
        return vector;
    }

    private static float[] combination(int... indices) {
        float[] vector = new float[DIMENSION];
        for (int index : indices) {
            vector[index] = 1.0f;
        }
        return vector;
    }
}
