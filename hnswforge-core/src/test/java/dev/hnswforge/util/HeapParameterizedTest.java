package dev.hnswforge.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class HeapParameterizedTest {

    @ParameterizedTest
    @CsvSource({"1, 10", "2, 10", "4, 20", "8, 20", "16, 40"})
    void boundedHeapRetainsBestElements(int capacity, int count) {
        BoundedMaxHeap heap = new BoundedMaxHeap(capacity);
        for (int id = count - 1; id >= 0; id--) {
            assertTrue(heap.push(id, id));
        }
        assertEquals(capacity, heap.size());
        assertEquals(capacity - 1, heap.peek());
        assertEquals((float) (capacity - 1), heap.peekDistance(), 1e-6f);
        for (int expected = capacity - 1; expected >= 0; expected--) {
            assertEquals(expected, heap.pop());
            assertEquals((float) expected, heap.popDistance(), 1e-6f);
        }
        assertEquals(0, heap.size());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 4, 8, 16})
    void boundedHeapRejectsWorseReplacement(int capacity) {
        BoundedMaxHeap heap = new BoundedMaxHeap(capacity);
        for (int id = 0; id < capacity; id++) {
            assertTrue(heap.push(id, id));
        }
        assertFalse(heap.push(1000, Float.MAX_VALUE));
        assertEquals(capacity, heap.size());
        assertTrue(heap.push(1000, -1f));
        assertEquals(capacity, heap.size());
        Set<Integer> retained = new HashSet<>();
        while (heap.size() > 0) {
            retained.add(heap.pop());
        }
        assertTrue(retained.contains(1000));
        assertFalse(retained.contains(capacity - 1));
        assertEquals(capacity, retained.size());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 4, 8})
    void minHeapPopsBestFirst(int count) {
        MinHeap heap = new MinHeap();
        for (int id = 0; id < count; id++) {
            heap.push(id, count - 1 - id);
        }
        for (int expected = count - 1; expected >= 0; expected--) {
            assertEquals(expected, heap.pop());
            assertEquals((float) (count - 1 - expected), heap.popDistance(), 1e-6f);
        }
        assertEquals(0, heap.size());

        heap.clear();
        heap.push(5, 1f);
        heap.push(2, 1f);
        heap.push(9, 1f);
        assertEquals(2, heap.pop());
        assertEquals(5, heap.pop());
        assertEquals(9, heap.pop());
    }

    @ParameterizedTest
    @CsvSource({"4, 0", "4, 3", "16, 7", "64, 63"})
    void visitedListEpochsAreIsolated(int capacity, int id) {
        VisitedList visited = new VisitedList(capacity);
        assertFalse(visited.isMarked(id));
        assertTrue(visited.mark(id));
        assertTrue(visited.isMarked(id));
        assertFalse(visited.mark(id));
        visited.begin();
        assertFalse(visited.isMarked(id));
        assertTrue(visited.mark(id));
        assertThrows(IllegalArgumentException.class, () -> visited.mark(capacity));
        assertThrows(IllegalArgumentException.class, () -> visited.isMarked(-1));
    }

    @ParameterizedTest
    @CsvSource({"3, 10, 20, 4", "8, 64, 100, 16"})
    void searchScratchSizesBuffers(int dimension, int capacity, int maxEf, int maxDegree) {
        SearchScratch scratch = new SearchScratch(dimension, capacity, maxEf, maxDegree);
        assertEquals(dimension, scratch.query.length);
        assertEquals(maxEf, scratch.results.capacity());
        assertEquals(maxDegree, scratch.chosen.length);
        assertEquals(maxDegree + 1, scratch.combinedIds.length);
        assertEquals(maxDegree + 1, scratch.combinedDistances.length);
        assertEquals(maxDegree + 1, scratch.neighborBatch.length);
        assertEquals(1, scratch.entryBuffer.length);
        assertTrue(scratch.visited.mark(capacity - 1));
        assertThrows(IllegalArgumentException.class, () -> scratch.visited.mark(capacity));
        scratch.ensureFound(maxEf + 5);
        assertTrue(scratch.foundIds.length >= maxEf + 5);
        assertTrue(scratch.foundDistances.length >= maxEf + 5);
    }
}
