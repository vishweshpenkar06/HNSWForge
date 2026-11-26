package dev.hnswforge.index;

import java.util.Objects;

/**
 * Immutable construction and search parameters for {@code HnswIndex}.
 *
 * <p>Validation is performed in the canonical constructor: {@code dimension > 0},
 * {@code m} in {@code [2, 128]}, {@code efConstruction >= m}, {@code efSearch >= 1},
 * {@code maxElements > 0}, and a non-null metric.
 *
 * <p>Thread-safety: immutable record; inherently thread-safe.
 *
 * @param dimension         vector dimension, must be positive
 * @param metric            distance metric, must not be null
 * @param m                 maximum neighbors per node above layer 0, in [2, 128]
 * @param efConstruction    candidate list size during insertion, at least {@code m}
 * @param efSearch          default candidate list size during search, at least 1
 * @param maxElements       capacity reserved by the index, must be positive
 * @param seed              seed for the deterministic level generator
 * @param neighborSelection heuristic or simple neighbor choice, must not be null
 */
public record HnswParams(
        int dimension,
        Metric metric,
        int m,
        int efConstruction,
        int efSearch,
        int maxElements,
        long seed,
        NeighborSelection neighborSelection) {

    /** Creates parameters with the default {@link NeighborSelection#HEURISTIC} strategy. */
    public HnswParams(int dimension, Metric metric, int m, int efConstruction,
                      int efSearch, int maxElements, long seed) {
        this(dimension, metric, m, efConstruction, efSearch, maxElements, seed,
                NeighborSelection.HEURISTIC);
    }

    /** Validates all parameters. */
    public HnswParams {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, got " + dimension);
        }
        Objects.requireNonNull(metric, "metric must not be null");
        if (m < 2 || m > 128) {
            throw new IllegalArgumentException("m must be in [2, 128], got " + m);
        }
        if (efConstruction < m) {
            throw new IllegalArgumentException(
                    "efConstruction must be >= m (" + m + "), got " + efConstruction);
        }
        if (efSearch < 1) {
            throw new IllegalArgumentException("efSearch must be >= 1, got " + efSearch);
        }
        if (maxElements <= 0) {
            throw new IllegalArgumentException("maxElements must be > 0, got " + maxElements);
        }
        Objects.requireNonNull(neighborSelection, "neighborSelection must not be null");
    }

    /** @return maximum degree allowed at layer 0 ({@code 2 * m}) */
    public int maxM0() {
        return 2 * m;
    }

    /** @return multiplier of the exponential level distribution ({@code 1 / ln(m)}) */
    public double levelMult() {
        return 1.0 / Math.log(m);
    }

    /**
     * Starts a builder with defaults {@code m=16}, {@code efConstruction=200},
     * {@code efSearch=64}, {@code maxElements=100000}, {@code seed=42},
     * {@code neighborSelection=HEURISTIC}.
     */
    public static Builder builder(int dimension, Metric metric) {
        return new Builder(dimension, metric);
    }

    /** Mutable builder for {@link HnswParams}. Not thread-safe. */
    public static final class Builder {
        private final int dimension;
        private final Metric metric;
        private int m = 16;
        private int efConstruction = 200;
        private int efSearch = 64;
        private int maxElements = 100_000;
        private long seed = 42L;
        private NeighborSelection neighborSelection = NeighborSelection.HEURISTIC;

        private Builder(int dimension, Metric metric) {
            this.dimension = dimension;
            this.metric = metric;
        }

        /** Sets {@code m}. */
        public Builder m(int value) {
            this.m = value;
            return this;
        }

        /** Sets {@code efConstruction}. */
        public Builder efConstruction(int value) {
            this.efConstruction = value;
            return this;
        }

        /** Sets {@code efSearch}. */
        public Builder efSearch(int value) {
            this.efSearch = value;
            return this;
        }

        /** Sets the reserved capacity. */
        public Builder maxElements(int value) {
            this.maxElements = value;
            return this;
        }

        /** Sets the random seed. */
        public Builder seed(long value) {
            this.seed = value;
            return this;
        }

        /** Sets the neighbor selection strategy. */
        public Builder neighborSelection(NeighborSelection value) {
            this.neighborSelection = value;
            return this;
        }

        /** Builds and validates the parameters. */
        public HnswParams build() {
            return new HnswParams(dimension, metric, m, efConstruction, efSearch,
                    maxElements, seed, neighborSelection);
        }
    }
}
