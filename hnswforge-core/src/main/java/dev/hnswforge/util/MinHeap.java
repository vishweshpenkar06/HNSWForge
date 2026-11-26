package dev.hnswforge.util;

/**
 * An unbounded min-heap of {@code (id, distance)} pairs backed by primitive arrays.
 *
 * <p>The best element (smallest distance, and for ties the smallest id) is at the
 * root. It grows geometrically and is used as the candidate frontier during graph
 * search. Iteration via {@link #pop()} yields elements from best to worst.
 *
 * <p>Thread-safety: not thread-safe; callers must serialize access.
 */
public final class MinHeap {
    private int[] ids;
    private float[] distances;
    private int size;
    private float lastPoppedDistance;

    /** Creates an empty heap with an initial capacity of 16. */
    public MinHeap() {
        this(16);
    }

    /** Creates an empty heap with the given initial capacity, which must be positive. */
    public MinHeap(int initialCapacity) {
        if (initialCapacity <= 0) {
            throw new IllegalArgumentException("initialCapacity must be > 0, got " + initialCapacity);
        }
        this.ids = new int[initialCapacity];
        this.distances = new float[initialCapacity];
        this.lastPoppedDistance = Float.NaN;
    }

    /** Inserts an element, growing the backing arrays when necessary. */
    public void push(int id, float distance) {
        if (Float.isNaN(distance)) {
            throw new IllegalArgumentException("distance must not be NaN");
        }
        if (size == ids.length) {
            grow();
        }
        int index = size++;
        ids[index] = id;
        distances[index] = distance;
        siftUp(index);
    }

    /** Removes and returns the id of the best element. */
    public int pop() {
        ensureNotEmpty();
        int id = ids[0];
        lastPoppedDistance = distances[0];
        size--;
        if (size > 0) {
            ids[0] = ids[size];
            distances[0] = distances[size];
            siftDown(0);
        }
        return id;
    }

    /** @return distance of the element returned by the most recent {@link #pop()} */
    public float popDistance() {
        return lastPoppedDistance;
    }

    /** @return id of the best element without removing it */
    public int peek() {
        ensureNotEmpty();
        return ids[0];
    }

    /** @return distance of the best element without removing it */
    public float peekDistance() {
        ensureNotEmpty();
        return distances[0];
    }

    /** @return number of elements currently held */
    public int size() {
        return size;
    }

    /** Removes all elements without releasing the backing arrays. */
    public void clear() {
        size = 0;
        lastPoppedDistance = Float.NaN;
    }

    private static boolean isBetter(float distanceA, int idA, float distanceB, int idB) {
        return distanceA < distanceB || (distanceA == distanceB && idA < idB);
    }

    private void grow() {
        int newCapacity = ids.length * 2;
        int[] newIds = new int[newCapacity];
        float[] newDistances = new float[newCapacity];
        System.arraycopy(ids, 0, newIds, 0, size);
        System.arraycopy(distances, 0, newDistances, 0, size);
        ids = newIds;
        distances = newDistances;
    }

    private void siftUp(int index) {
        int child = index;
        while (child > 0) {
            int parent = (child - 1) >>> 1;
            if (!isBetter(distances[child], ids[child], distances[parent], ids[parent])) {
                break;
            }
            swap(child, parent);
            child = parent;
        }
    }

    private void siftDown(int index) {
        int parent = index;
        while (true) {
            int left = parent * 2 + 1;
            if (left >= size) {
                break;
            }
            int right = left + 1;
            int best = left;
            if (right < size && isBetter(distances[right], ids[right], distances[left], ids[left])) {
                best = right;
            }
            if (!isBetter(distances[best], ids[best], distances[parent], ids[parent])) {
                break;
            }
            swap(parent, best);
            parent = best;
        }
    }

    private void swap(int a, int b) {
        int id = ids[a];
        ids[a] = ids[b];
        ids[b] = id;
        float distance = distances[a];
        distances[a] = distances[b];
        distances[b] = distance;
    }

    private void ensureNotEmpty() {
        if (size == 0) {
            throw new IllegalStateException("heap is empty");
        }
    }
}
