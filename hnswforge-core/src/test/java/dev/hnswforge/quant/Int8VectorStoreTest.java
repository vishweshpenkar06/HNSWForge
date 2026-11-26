package dev.hnswforge.quant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.util.SyntheticData;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;
import java.util.function.IntToDoubleFunction;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests for {@link Int8VectorStore}. */
class Int8VectorStoreTest {

    @ParameterizedTest
    @ValueSource(ints = {4, 16, 32})
    void codeBytesAreOneQuarterOfFloat32(int dimension) {
        int n = 200;
        float[][] vectors = uniformSigned(n, dimension, 11L);
        Int8VectorStore store = new Int8VectorStore(dimension, n);
        for (int id = 0; id < n; id++) {
            store.set(id, vectors[id]);
        }
        assertEquals(dimension, store.dimension());
        assertEquals(n, store.capacity());
        assertEquals(n, store.size());
        assertEquals((long) n * dimension, store.codeBytes());
        assertEquals(4L * store.codeBytes(), (long) n * dimension * Float.BYTES);
    }

    @ParameterizedTest
    @ValueSource(ints = {4, 16, 32})
    void dequantizationErrorIsSmall(int dimension) {
        int n = 200;
        float[][] vectors = uniformSigned(n, dimension, 23L);
        Int8VectorStore store = new Int8VectorStore(dimension, n);
        for (int id = 0; id < n; id++) {
            store.set(id, vectors[id]);
        }
        float maxError = 0f;
        for (int id = 0; id < n; id++) {
            float[] copy = store.copyOf(id);
            for (int d = 0; d < dimension; d++) {
                maxError = Math.max(maxError, Math.abs(copy[d] - vectors[id][d]));
            }
        }
        assertTrue(maxError < 0.05f, "max error was " + maxError + " for dimension " + dimension);
    }

    @ParameterizedTest
    @ValueSource(ints = {4, 16, 32})
    void recallAtTenIsAtLeastNinetyPercent(int dimension) {
        int n = 200;
        float[][] vectors = uniformSigned(n, dimension, 31L);
        Int8VectorStore store = new Int8VectorStore(dimension, n);
        for (int id = 0; id < n; id++) {
            store.set(id, vectors[id]);
        }
        float[][] queries = SyntheticData.queries(vectors, 20, 43L);
        double totalRecall = 0.0;
        for (float[] query : queries) {
            int[] exact = topK(n, 10, id -> squaredL2(query, vectors[id]));
            int[] approximate = topK(n, 10, id -> store.dist(query, id));
            Set<Integer> exactSet = new HashSet<>();
            for (int id : exact) {
                exactSet.add(id);
            }
            int hits = 0;
            for (int id : approximate) {
                if (exactSet.contains(id)) {
                    hits++;
                }
            }
            totalRecall += hits / 10.0;
        }
        double recall = totalRecall / queries.length;
        assertTrue(recall >= 0.9, "recall@10 was " + recall + " for dimension " + dimension);
    }

    @Test
    void fullPrecisionDistanceMatchesSquaredL2() {
        Int8VectorStore store = new Int8VectorStore(3, 4, true);
        store.set(0, new float[] {0.25f, -0.5f, 1.5f});
        store.set(1, new float[] {2f, 2f, 2f});
        float[] query = {1f, 0f, 0.5f};
        assertEquals(1.8125f, store.fullPrecisionDist(query, 0), 1e-6f);
    }

    @Test
    void fullPrecisionDistanceThrowsWhenDisabled() {
        Int8VectorStore store = new Int8VectorStore(2, 2);
        store.set(0, new float[] {1f, 2f});
        assertThrows(IllegalStateException.class,
                () -> store.fullPrecisionDist(new float[] {0f, 0f}, 0));
    }

    @Test
    void distanceBetweenIdsMatchesDequantizedVectors() {
        Int8VectorStore store = new Int8VectorStore(2, 3);
        store.set(0, new float[] {0f, 0f});
        store.set(1, new float[] {1f, 1f});
        float[] a = store.copyOf(0);
        float[] b = store.copyOf(1);
        assertEquals((float) squaredL2(a, b), store.dist(0, 1), 1e-5f);
        float[] query = {0.5f, -0.25f};
        assertEquals((float) squaredL2(query, a), store.dist(query, 0), 1e-5f);
    }

    @Test
    void calibrateResetsRanges() {
        Int8VectorStore store = new Int8VectorStore(2, 4, true);
        store.set(0, new float[] {0.5f, 0.5f});
        Int8VectorStore.calibrate(store, new float[][] {{-2f, -1f}, {2f, 1f}});
        int code = Math.round((0.5f + 2f) / 4f * 255f);
        assertEquals(-2f + (code / 255f) * 4f, store.copyOf(0)[0], 1e-5f);
        store.set(1, new float[] {2f, 1f});
        float[] copy = store.copyOf(1);
        assertEquals(2f, copy[0], 1e-5f);
        assertEquals(1f, copy[1], 1e-5f);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Int8VectorStore(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Int8VectorStore(1, 0));
        Int8VectorStore store = new Int8VectorStore(2, 2);
        float[] vector = {1f, 2f};
        assertThrows(IllegalArgumentException.class, () -> store.set(-1, vector));
        assertThrows(IllegalArgumentException.class, () -> store.set(2, vector));
        assertThrows(IllegalArgumentException.class, () -> store.set(0, new float[] {1f}));
        assertThrows(IllegalArgumentException.class,
                () -> store.set(0, new float[] {Float.NaN, 2f}));
        store.set(0, vector);
        assertThrows(IllegalArgumentException.class, () -> store.copyOf(-1));
        assertThrows(IllegalArgumentException.class, () -> store.copyOf(1));
        assertThrows(IllegalArgumentException.class, () -> store.dist(new float[] {1f}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> store.dist(new float[] {Float.NaN, 0f}, 0));
        assertThrows(IllegalArgumentException.class, () -> store.dist(new float[] {0f, 0f}, -1));
        assertThrows(IllegalArgumentException.class, () -> store.dist(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> store.dist(0, 1));
    }

    private static float[][] uniformSigned(int count, int dimension, long seed) {
        float[][] vectors = SyntheticData.uniform(count, dimension, seed);
        for (float[] vector : vectors) {
            for (int d = 0; d < vector.length; d++) {
                vector[d] = 2f * vector[d] - 1f;
            }
        }
        return vectors;
    }

    private static double squaredL2(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double delta = (double) a[i] - b[i];
            sum += delta * delta;
        }
        return sum;
    }

    private static int[] topK(int count, int k, IntToDoubleFunction distance) {
        Integer[] ids = new Integer[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i;
        }
        Arrays.sort(ids, Comparator.<Integer>comparingDouble(distance::applyAsDouble)
                .thenComparingInt(id -> id));
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = ids[i];
        }
        return result;
    }
}
