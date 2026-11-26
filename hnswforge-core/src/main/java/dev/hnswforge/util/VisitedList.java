package dev.hnswforge.util;

import java.util.Arrays;

/**
 * An epoch-stamped visited set that avoids clearing an array on every use.
 *
 * <p>Calling {@link #begin()} increments the epoch; an id is considered visited
 * when its stamp equals the current epoch. The array is cleared only on integer
 * overflow. Intended to be pooled per thread.
 *
 * <p>Thread-safety: not thread-safe; each thread must own its own instance.
 */
public final class VisitedList {
    private final int[] stamps;
    private int epoch = 1;

    /** Creates an empty visited list for ids in {@code [0, capacity)}. */
    public VisitedList(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0, got " + capacity);
        }
        this.stamps = new int[capacity];
    }

    /** Starts a new search epoch, clearing the stamps only on overflow. */
    public void begin() {
        epoch++;
        if (epoch == Integer.MAX_VALUE) {
            Arrays.fill(stamps, 0);
            epoch = 1;
        }
    }

    /**
     * Marks an id as visited.
     *
     * @return true if it was not already visited in this epoch
     * @throws IllegalArgumentException if the id is outside the valid range
     */
    public boolean mark(int id) {
        checkId(id);
        if (stamps[id] == epoch) {
            return false;
        }
        stamps[id] = epoch;
        return true;
    }

    /** @return whether the id was marked in the current epoch */
    public boolean isMarked(int id) {
        checkId(id);
        return stamps[id] == epoch;
    }

    private void checkId(int id) {
        if (id < 0 || id >= stamps.length) {
            throw new IllegalArgumentException("id outside [0, capacity): " + id);
        }
    }
}
