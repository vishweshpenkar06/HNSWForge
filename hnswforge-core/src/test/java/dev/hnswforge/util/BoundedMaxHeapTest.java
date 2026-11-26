package dev.hnswforge.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import net.jqwik.api.Arbitraries;
import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import net.jqwik.api.Provide;
import org.junit.jupiter.api.Test;

/** Tests for the primitive bounded max-heap. */
class BoundedMaxHeapTest {

    @Test
    void popsWorstFirst() {
        BoundedMaxHeap heap = new BoundedMaxHeap(3);
        heap.push(1, 1.0f);
        heap.push(2, 3.0f);
        heap.push(3, 2.0f);
        assertEquals(2, heap.pop());
        assertEquals(3.0f, heap.popDistance(), 1e-6f);
        assertEquals(3, heap.pop());
        assertEquals(2.0f, heap.popDistance(), 1e-6f);
        assertEquals(1, heap.pop());
    }

    @Test
    void keepsOnlyBestElementsWhenFull() {
        BoundedMaxHeap heap = new BoundedMaxHeap(2);
        heap.push(1, 5.0f);
        heap.push(2, 3.0f);
        assertFalse(heap.push(3, 9.0f));
        assertTrue(heap.push(4, 1.0f));
        assertEquals(2, heap.size());
        assertEquals(2, heap.pop());
        assertEquals(4, heap.pop());
    }

    @Test
    void tieBreakKeepsLowerId() {
        BoundedMaxHeap heap = new BoundedMaxHeap(1);
        assertTrue(heap.push(7, 1.0f));
        assertTrue(heap.push(3, 1.0f));
        assertEquals(3, heap.peek());
    }

    @Test
    void peekExposesWorstElement() {
        BoundedMaxHeap heap = new BoundedMaxHeap(2);
        heap.push(1, 2.0f);
        heap.push(2, 5.0f);
        assertEquals(2, heap.peek());
        assertEquals(5.0f, heap.peekDistance(), 1e-6f);
        assertEquals(2, heap.size());
    }

    @Test
    void clearResetsHeap() {
        BoundedMaxHeap heap = new BoundedMaxHeap(2);
        heap.push(1, 1.0f);
        heap.clear();
        assertEquals(0, heap.size());
        assertThrows(IllegalStateException.class, heap::pop);
    }

    @Test
    void zeroCapacityThrows() {
        assertThrows(IllegalArgumentException.class, () -> new BoundedMaxHeap(0));
    }

    @Test
    void nanDistanceThrows() {
        BoundedMaxHeap heap = new BoundedMaxHeap(1);
        assertThrows(IllegalArgumentException.class, () -> heap.push(1, Float.NaN));
    }

    @Property
    void matchesPriorityQueueOracle(@ForAll("samples") List<Sample> samples,
                                    @ForAll @IntRange(min = 1, max = 12) int capacity) {
        Comparator<Sample> worstFirst = Comparator
                .comparingDouble(Sample::distance)
                .reversed()
                .thenComparing(Comparator.comparingInt(Sample::id).reversed());
        PriorityQueue<Sample> oracle = new PriorityQueue<>(worstFirst);
        for (Sample sample : samples) {
            oracle.add(sample);
        }
        while (oracle.size() > capacity) {
            oracle.poll();
        }

        BoundedMaxHeap heap = new BoundedMaxHeap(capacity);
        for (Sample sample : samples) {
            heap.push(sample.id(), sample.distance());
        }
        assertEquals(oracle.size(), heap.size());
        while (!oracle.isEmpty()) {
            Sample expected = oracle.poll();
            assertEquals(expected.id(), heap.pop());
            assertEquals(expected.distance(), heap.popDistance(), 1e-6f);
        }
    }

    @Provide
    Arbitrary<List<Sample>> samples() {
        return Arbitraries.integers().between(0, 300)
                .flatMap(id -> Arbitraries.floats().between(0f, 100f)
                        .map(distance -> new Sample(id, distance)))
                .list().ofSize(40);
    }

    /** A test pair of an id and a distance. */
    record Sample(int id, float distance) {}
}
