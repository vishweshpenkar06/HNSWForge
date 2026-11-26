package dev.hnswforge.util;

/**
 * A bounded max-heap of {@code (id, distance)} pairs backed by primitive arrays.
 *
 * <p>The worst element (largest distance, and for ties the largest id) is kept at
 * the root. {@link #push(int, float)} replaces the root when the new element is
 * better, so the heap always holds the best {@code capacity} elements. Iteration
 * via {@link #pop()} yields elements from worst to best.
 *
 * <p>Thread-safety: not thread-safe; callers must serialize access.
 */
public final class BoundedMaxHeap {
    private final int[] ids;
    private final float[] distances;
    private int size;
    private float lastPoppedDistance;

    /** Creates an empty heap with the given capacity, which must be positive. */
    public BoundedMaxHeap(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0, got " + capacity);
        }
        this.ids = new int[capacity];
        this.distances = new float[capacity];
        this.lastPoppedDistance = Float.NaN;
    }

    /**
     * Inserts an element, evicting the current worst one when the heap is full
     * and the new element is better.
     *
     * @return true if the element is in the heap after the call
     */
    public boolean push(int id, float distance) {
        if (Float.isNaN(distance)) {
            throw new IllegalArgumentException("distance must not be NaN");
        }
        if (size < ids.length) {
            int index = size++;
            ids[index] = id;
            distances[index] = distance;
            siftUp(index);
            return true;
        }
        if (isWorse(distance, id, distances[0], ids[0])) {
            return false;
        }
        ids[0] = id;
        distances[0] = distance;
        siftDown(0);
        return true;
    }

    /** Removes and returns the id of the worst element. */
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

    /** @return id of the worst element without removing it */
    public int peek() {
        ensureNotEmpty();
        return ids[0];
    }

    /** @return distance of the worst element without removing it */
    public float peekDistance() {
        ensureNotEmpty();
        return distances[0];
    }

    /** @return number of elements currently held */
    public int size() {
        return size;
    }

    /** @return maximum number of elements held */
    public int capacity() {
        return ids.length;
    }

    /** Removes all elements. */
    public void clear() {
        size = 0;
        lastPoppedDistance = Float.NaN;
    }

    private static boolean isWorse(float distanceA, int idA, float distanceB, int idB) {
        return distanceA > distanceB || (distanceA == distanceB && idA > idB);
    }

    private void siftUp(int index) {
        int child = index;
        while (child > 0) {
            int parent = (child - 1) >>> 1;
            if (!isWorse(distances[child], ids[child], distances[parent], ids[parent])) {
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
            int worst = left;
            if (right < size && isWorse(distances[right], ids[right], distances[left], ids[left])) {
                worst = right;
            }
            if (!isWorse(distances[worst], ids[worst], distances[parent], ids[parent])) {
                break;
            }
            swap(parent, worst);
            parent = worst;
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
