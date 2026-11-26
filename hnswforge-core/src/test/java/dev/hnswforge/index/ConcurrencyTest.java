package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.util.SyntheticData;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Stress tests for concurrent insertion and mixed read/write workloads. */
class ConcurrencyTest {

    @Test
    void eightThreadsInsertFiftyThousandVectorsSafely() {
        float[][] base = SyntheticData.uniform(50_000, 16, 123L);
        HnswParams params = HnswParams.builder(16, Metric.L2)
                .m(8).efConstruction(100).maxElements(50_000).seed(42L).build();
        HnswIndex index = new HnswIndex(params);
        assertTimeoutPreemptively(Duration.ofSeconds(120), () -> index.addAll(base, 8));

        assertEquals(50_000, index.size());
        assertEquals(50_000, index.liveCount());
        int bound = params.maxM0();
        for (int id = 0; id < index.size(); id++) {
            Set<Integer> seen = new HashSet<>();
            int count = index.neighborCount(id, 0);
            assertTrue(count <= bound, "degree exceeds bound at " + id);
            for (int i = 0; i < count; i++) {
                int neighbor = index.neighborAt(id, 0, i);
                assertTrue(neighbor >= 0 && neighbor < index.size());
                assertFalse(neighbor == id, "self loop");
                assertTrue(seen.add(neighbor), "duplicate neighbor");
            }
        }
    }

    @Test
    void concurrentBuildRecallStaysCloseToSequentialBuild() {
        float[][] base = SyntheticData.uniform(20_000, 16, 123L);
        float[][] queries = SyntheticData.queries(base, 100, 456L);
        HnswParams params = HnswParams.builder(16, Metric.L2)
                .m(8).efConstruction(100).efSearch(100)
                .maxElements(20_000).seed(42L).build();
        HnswIndex parallel = new HnswIndex(params);
        parallel.addAll(base, 8);
        HnswIndex sequential = new HnswIndex(params);
        for (float[] vector : base) {
            sequential.add(vector);
        }
        BruteForceIndex parallelTruth = new BruteForceIndex(16, Metric.L2);
        for (int id = 0; id < parallel.size(); id++) {
            parallelTruth.add(parallel.vector(id));
        }
        BruteForceIndex sequentialTruth = new BruteForceIndex(16, Metric.L2);
        for (int id = 0; id < sequential.size(); id++) {
            sequentialTruth.add(sequential.vector(id));
        }
        double parallelRecall = recall(parallel, parallelTruth, queries);
        double sequentialRecall = recall(sequential, sequentialTruth, queries);
        assertTrue(Math.abs(parallelRecall - sequentialRecall) <= 0.01,
                "parallel=" + parallelRecall + " sequential=" + sequentialRecall);
    }

    @Test
    void mixedInsertAndSearchWorkloadRunsWithoutInvalidResults() throws Exception {
        float[][] base = SyntheticData.uniform(20_000, 16, 7L);
        HnswParams params = HnswParams.builder(16, Metric.L2)
                .m(8).efConstruction(100).efSearch(60)
                .maxElements(100_000).seed(7L).build();
        HnswIndex index = new HnswIndex(params);
        AtomicBoolean stop = new AtomicBoolean();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        CountDownLatch ready = new CountDownLatch(8);
        List<Thread> threads = new ArrayList<>();
        for (int worker = 0; worker < 4; worker++) {
            final int offset = worker;
            threads.add(new Thread(() -> {
                ready.countDown();
                try {
                    int next = offset;
                    while (!stop.get()) {
                        index.add(base[next % base.length]);
                        next += 4;
                    }
                } catch (Throwable throwable) {
                    failure.compareAndSet(null, throwable);
                }
            }));
        }
        for (int worker = 0; worker < 4; worker++) {
            threads.add(new Thread(() -> {
                ready.countDown();
                try {
                    while (!stop.get()) {
                        float[] query = base[(int) (Math.random() * base.length)];
                        for (SearchResult result : index.search(query, 10, 60)) {
                            if (result.id() < 0 || result.id() >= index.size()) {
                                throw new IllegalStateException("invalid id: " + result.id());
                            }
                        }
                    }
                } catch (Throwable throwable) {
                    failure.compareAndSet(null, throwable);
                }
            }));
        }
        for (Thread thread : threads) {
            thread.start();
        }
        assertTrue(ready.await(10, TimeUnit.SECONDS));
        Thread.sleep(10_000);
        stop.set(true);
        for (Thread thread : threads) {
            thread.join(30_000);
            assertFalse(thread.isAlive(), "worker did not complete");
        }
        if (failure.get() != null) {
            throw new AssertionError("worker failed", failure.get());
        }
        assertTrue(index.size() > 0);
    }

    private static double recall(HnswIndex index, BruteForceIndex exact, float[][] queries) {
        double total = 0.0;
        for (float[] query : queries) {
            Set<Integer> truth = new HashSet<>();
            for (SearchResult result : exact.search(query, 10)) {
                truth.add(result.id());
            }
            int hits = 0;
            for (SearchResult result : index.search(query, 10, 100)) {
                if (truth.contains(result.id())) {
                    hits++;
                }
            }
            total += hits / 10.0;
        }
        return total / queries.length;
    }
}
