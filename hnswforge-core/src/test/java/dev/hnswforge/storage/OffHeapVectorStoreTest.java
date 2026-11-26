package dev.hnswforge.storage;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.index.Metric;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Tests for the off-heap vector store. */
class OffHeapVectorStoreTest {

    @Test
    void setAndCopyOfRoundTrip() {
        float[] vector = {1f, -2f, 3f};
        try (OffHeapVectorStore store = new OffHeapVectorStore(3, 4, ScalarDistance.L2)) {
            store.set(2, vector);
            assertEquals(3, store.dimension());
            assertEquals(4, store.capacity());
            assertEquals(3, store.size());
            assertArrayEquals(vector, store.copyOf(2));
            assertEquals(0f, store.dist(vector, 2), 1e-6f);
            assertEquals(12L * Float.BYTES, store.segment().byteSize());
        }
    }

    @Test
    void distancesMatchHeapVectorStore() {
        Random random = new Random(19L);
        for (Metric metric : Metric.values()) {
            DistanceFunction kernel = ScalarDistance.forMetric(metric);
            int dim = 7;
            int count = 6;
            float[][] vectors = new float[count][dim];
            for (float[] vector : vectors) {
                for (int i = 0; i < dim; i++) {
                    vector[i] = random.nextFloat() * 4f - 2f;
                }
            }
            try (OffHeapVectorStore offHeap = new OffHeapVectorStore(dim, count, kernel);
                    HeapVectorStore heap = new HeapVectorStore(dim, count, kernel)) {
                for (int id = 0; id < count; id++) {
                    offHeap.set(id, vectors[id]);
                    heap.set(id, vectors[id]);
                }
                float[] query = vectors[0];
                for (int id = 0; id < count; id++) {
                    assertEquals(heap.dist(query, id), offHeap.dist(query, id), 1e-6f);
                    assertEquals(heap.dist(id, (id + 1) % count),
                            offHeap.dist(id, (id + 1) % count), 1e-6f);
                }
            }
        }
    }

    @Test
    void closeMakesAllOtherMethodsThrow() {
        OffHeapVectorStore store = new OffHeapVectorStore(2, 2, ScalarDistance.L2);
        store.set(0, new float[] {1f, 2f});
        store.close();
        store.close();
        assertEquals("store is closed",
                assertThrows(IllegalStateException.class, store::dimension).getMessage());
        assertThrows(IllegalStateException.class, store::capacity);
        assertThrows(IllegalStateException.class, store::size);
        assertThrows(IllegalStateException.class, store::segment);
        assertThrows(IllegalStateException.class, () -> store.set(0, new float[] {1f, 2f}));
        assertThrows(IllegalStateException.class, () -> store.copyOf(0));
        assertThrows(IllegalStateException.class, () -> store.dist(new float[] {1f, 2f}, 0));
        assertThrows(IllegalStateException.class, () -> store.dist(0, 0));
    }

    @Test
    void invalidIdsAndDimensionsThrow() {
        try (OffHeapVectorStore store = new OffHeapVectorStore(2, 2, ScalarDistance.L2)) {
            assertThrows(IllegalArgumentException.class, () -> store.set(-1, new float[] {1f, 2f}));
            assertThrows(IllegalArgumentException.class, () -> store.set(2, new float[] {1f, 2f}));
            assertThrows(IllegalArgumentException.class, () -> store.set(0, new float[] {1f}));
            assertThrows(IllegalArgumentException.class, () -> store.copyOf(0));
            store.set(0, new float[] {1f, 2f});
            assertThrows(IllegalArgumentException.class, () -> store.copyOf(2));
            assertThrows(IllegalArgumentException.class, () -> store.dist(new float[] {1f, 2f}, 2));
            assertThrows(IllegalArgumentException.class, () -> store.dist(0, 2));
        }
    }
}
