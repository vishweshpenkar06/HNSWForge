package dev.hnswforge.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import net.jqwik.api.Arbitraries;
import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;
import org.junit.jupiter.api.Test;

/** Tests for the primitive unbounded min-heap. */
class MinHeapTest {

    @Test
    void popsBestFirst() {
        MinHeap heap = new MinHeap();
        heap.push(1, 3.0f);
        heap.push(2, 1.0f);
        heap.push(3, 2.0f);
        assertEquals(2, heap.pop());
        assertEquals(1.0f, heap.popDistance(), 1e-6f);
        assertEquals(3, heap.pop());
        assertEquals(1, heap.pop());
    }

    @Test
    void tieBreakUsesLowerIdFirst() {
        MinHeap heap = new MinHeap();
        heap.push(9, 1.0f);
        heap.push(2, 1.0f);
        assertEquals(2, heap.pop());
        assertEquals(9, heap.pop());
    }

    @Test
    void growsBeyondInitialCapacity() {
        MinHeap heap = new MinHeap(1);
        for (int i = 0; i < 100; i++) {
            heap.push(i, 100 - i);
        }
        assertEquals(100, heap.size());
        assertEquals(99, heap.pop());
    }

    @Test
    void peekExposesBestElement() {
        MinHeap heap = new MinHeap();
        heap.push(1, 2.0f);
        heap.push(2, 5.0f);
        assertEquals(1, heap.peek());
        assertEquals(2.0f, heap.peekDistance(), 1e-6f);
    }

    @Test
    void clearResetsHeap() {
        MinHeap heap = new MinHeap();
        heap.push(1, 1.0f);
        heap.clear();
        assertEquals(0, heap.size());
        assertThrows(IllegalStateException.class, heap::pop);
    }

    @Test
    void invalidArgumentsThrow() {
        assertThrows(IllegalArgumentException.class, () -> new MinHeap(0));
        MinHeap heap = new MinHeap();
        assertThrows(IllegalArgumentException.class, () -> heap.push(1, Float.NaN));
    }

    @Property
    void matchesPriorityQueueOracle(@ForAll("samples") List<Sample> samples) {
        Comparator<Sample> bestFirst = Comparator
                .comparingDouble(Sample::distance)
                .thenComparingInt(Sample::id);
        PriorityQueue<Sample> oracle = new PriorityQueue<>(bestFirst);
        for (Sample sample : samples) {
            oracle.add(sample);
        }
        MinHeap heap = new MinHeap();
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
