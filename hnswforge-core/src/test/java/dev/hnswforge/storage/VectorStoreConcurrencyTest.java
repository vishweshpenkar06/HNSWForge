package dev.hnswforge.storage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

import dev.hnswforge.distance.ScalarDistance;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

/** Regression tests for concurrent vector publication. */
class VectorStoreConcurrencyTest {

    @Test
    void heapStoreSizeIsMonotonicUnderOutOfOrderInserts() throws Exception {
        assertConcurrentInserts(new HeapVectorStore(4, 2_000, ScalarDistance.L2));
    }

    @Test
    void offHeapStoreSizeIsMonotonicUnderOutOfOrderInserts() throws Exception {
        try (OffHeapVectorStore store =
                     new OffHeapVectorStore(4, 2_000, ScalarDistance.L2)) {
            assertConcurrentInserts(store);
        }
    }

    private static void assertConcurrentInserts(VectorStore store) throws Exception {
        int count = 2_000;
        List<Integer> ids = new ArrayList<>();
        for (int id = 0; id < count; id++) {
            ids.add(id);
        }
        Collections.shuffle(ids, new java.util.Random(42));
        ExecutorService pool = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int id : ids) {
                final int vectorId = id;
                futures.add(pool.submit(() -> {
                    start.await();
                    store.set(vectorId, new float[] {vectorId, 1, 2, 3});
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> future : futures) {
                future.get();
            }
        } finally {
            pool.shutdownNow();
        }
        assertEquals(count, store.size());
        for (int id = 0; id < count; id++) {
            final int expectedId = id;
            assertEquals(expectedId, Math.round(store.copyOf(expectedId)[0]));
            assertDoesNotThrow(
                    () -> store.dist(new float[] {expectedId, 1, 2, 3}, expectedId));
        }
    }
}
