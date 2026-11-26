package dev.hnswforge.quant;

import dev.hnswforge.storage.VectorStore;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Objects;

/**
 * Scalar int8 quantized {@link VectorStore} with per-dimension min/max calibration.
 *
 * <p>Every dimension of every vector is stored as one byte in a single contiguous
 * {@code codes} array, using one quarter of the space of an equivalent float32
 * array. A code {@code c} in {@code [0, 255]} dequantizes to
 * {@code min[d] + (c / 255f) * (max[d] - min[d])}. Calibration is learned
 * incrementally from the first {@link #set} calls and expands as more extreme
 * values arrive; previously inserted codes are re-encoded when a range grows.
 * {@link #calibrate} can reset the ranges from a representative sample corpus.
 * Squared-L2 distances are computed directly against the byte codes without
 * allocating an intermediate vector.
 *
 * <p>Thread-safety: not thread-safe for writes. Once all writes are published,
 * concurrent reads through {@link #copyOf}, {@link #dist} and
 * {@link #fullPrecisionDist} are safe.
 */
public final class Int8VectorStore implements VectorStore {
    private static final int MAX_CODE = 255;

    private final int dimension;
    private final int capacity;
    private final byte[] codes;
    private final boolean keepFullPrecision;
    private final float[] fullPrecision;
    private float[] min;
    private float[] max;
    private boolean calibrated;
    private final AtomicInteger size = new AtomicInteger();

    /**
     * Creates a quantized store without full-precision retention.
     *
     * <p>Thread-safety: construction is safe; subsequent writes are not
     * thread-safe and must be published before readers use the store.
     *
     * @param dimension vector dimension, must be positive
     * @param capacity  maximum number of vectors, must be positive
     * @throws IllegalArgumentException if {@code dimension} or {@code capacity} is not positive
     */
    public Int8VectorStore(int dimension, int capacity) {
        this(dimension, capacity, false);
    }

    /**
     * Creates a quantized store, optionally retaining the original float vectors
     * for exact reranking through {@link #fullPrecisionDist}.
     *
     * <p>Thread-safety: construction is safe; subsequent writes are not
     * thread-safe and must be published before readers use the store.
     *
     * @param dimension         vector dimension, must be positive
     * @param capacity          maximum number of vectors, must be positive
     * @param keepFullPrecision whether to retain original float32 vectors
     * @throws IllegalArgumentException if {@code dimension} or {@code capacity} is not positive
     */
    public Int8VectorStore(int dimension, int capacity, boolean keepFullPrecision) {
        if (dimension <= 0 || capacity <= 0) {
            throw new IllegalArgumentException("dimension and capacity must be positive");
        }
        this.dimension = dimension;
        this.capacity = capacity;
        this.codes = new byte[Math.multiplyExact(dimension, capacity)];
        this.keepFullPrecision = keepFullPrecision;
        this.fullPrecision = keepFullPrecision
                ? new float[Math.multiplyExact(dimension, capacity)] : null;
    }

    /**
     * Returns the dimension of every stored vector.
     *
     * <p>Thread-safety: safe for concurrent reads.
     *
     * @return vector dimension
     */
    @Override
    public int dimension() {
        return dimension;
    }

    /**
     * Returns the maximum number of vectors this store can hold.
     *
     * <p>Thread-safety: safe for concurrent reads.
     *
     * @return capacity
     */
    @Override
    public int capacity() {
        return capacity;
    }

    /**
     * Returns the number of stored vectors, equal to the highest inserted id plus one.
     *
     * <p>Thread-safety: safe for concurrent reads after writes are published.
     *
     * @return number of stored vectors
     */
    @Override
    public int size() {
        return size.get();
    }

    /**
     * Stores a vector under {@code id}, quantizing it with the current calibration
     * and expanding the per-dimension ranges when needed. The first inserted
     * vector initializes the ranges. Vectors inserted before an expansion are
     * re-encoded so their dequantized values stay within one quantization step.
     *
     * <p>Thread-safety: must not be called concurrently with any other method on
     * this store.
     *
     * @param id     target id in {@code [0, capacity())}
     * @param vector vector of length {@link #dimension()} with finite values
     * @throws IllegalArgumentException if the id or vector is invalid
     */
    @Override
    public void set(int id, float[] vector) {
        if (id < 0 || id >= capacity) {
            throw new IllegalArgumentException("id outside [0, capacity()): " + id);
        }
        Objects.requireNonNull(vector, "vector must not be null");
        if (vector.length != dimension) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + dimension + ", got " + vector.length);
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("vector values must be finite");
            }
        }
        if (!calibrated) {
            min = vector.clone();
            max = vector.clone();
            calibrated = true;
        } else {
            float[] oldMin = min;
            float[] oldMax = max;
            float[] newMin = min.clone();
            float[] newMax = max.clone();
            boolean expanded = false;
            for (int d = 0; d < dimension; d++) {
                float value = vector[d];
                if (value < newMin[d]) {
                    newMin[d] = value;
                    expanded = true;
                }
                if (value > newMax[d]) {
                    newMax[d] = value;
                    expanded = true;
                }
            }
            if (expanded) {
                for (int previous = 0; previous < size.get(); previous++) {
                    if (previous != id) {
                        reencode(previous, oldMin, oldMax, newMin, newMax);
                    }
                }
                min = newMin;
                max = newMax;
            }
        }
        encode(id, vector);
        if (keepFullPrecision) {
            System.arraycopy(vector, 0, fullPrecision, id * dimension, dimension);
        }
        size.accumulateAndGet(id + 1, Math::max);
    }

    /**
     * Returns the dequantized vector stored at {@code id}.
     *
     * <p>Thread-safety: safe for concurrent reads after writes are published.
     *
     * @param id id in {@code [0, size())}
     * @return a fresh float array containing the dequantized values
     * @throws IllegalArgumentException if {@code id} is outside {@code [0, size())}
     */
    @Override
    public float[] copyOf(int id) {
        checkId(id);
        float[] result = new float[dimension];
        dequantize(id, result);
        return result;
    }

    /**
     * Computes squared L2 distance between a query and the dequantized vector at
     * {@code id}; smaller values mean closer vectors. No temporary vector is
     * allocated.
     *
     * <p>Thread-safety: safe for concurrent reads after writes are published.
     *
     * @param query query vector of length {@link #dimension()} with finite values
     * @param id    id in {@code [0, size())}
     * @return squared L2 distance
     * @throws IllegalArgumentException if the query or id is invalid
     */
    @Override
    public float dist(float[] query, int id) {
        checkQuery(query);
        checkId(id);
        int base = id * dimension;
        float sum = 0f;
        for (int d = 0; d < dimension; d++) {
            float value = min[d] + ((codes[base + d] & 0xFF) / (float) MAX_CODE)
                    * (max[d] - min[d]);
            float delta = query[d] - value;
            sum += delta * delta;
        }
        return sum;
    }

    /**
     * Computes squared L2 distance between the dequantized vectors at {@code a}
     * and {@code b}; smaller values mean closer vectors. No temporary vector is
     * allocated.
     *
     * <p>Thread-safety: safe for concurrent reads after writes are published.
     *
     * @param a first id in {@code [0, size())}
     * @param b second id in {@code [0, size())}
     * @return squared L2 distance
     * @throws IllegalArgumentException if either id is invalid
     */
    @Override
    public float dist(int a, int b) {
        checkId(a);
        checkId(b);
        int baseA = a * dimension;
        int baseB = b * dimension;
        float sum = 0f;
        for (int d = 0; d < dimension; d++) {
            float valueA = min[d] + ((codes[baseA + d] & 0xFF) / (float) MAX_CODE)
                    * (max[d] - min[d]);
            float valueB = min[d] + ((codes[baseB + d] & 0xFF) / (float) MAX_CODE)
                    * (max[d] - min[d]);
            float delta = valueA - valueB;
            sum += delta * delta;
        }
        return sum;
    }

    /**
     * Computes squared L2 distance between a query and the original float32
     * vector at {@code id}, for reranking. Available only when this store was
     * created with {@code keepFullPrecision = true}.
     *
     * <p>Thread-safety: safe for concurrent reads after writes are published.
     *
     * @param query query vector of length {@link #dimension()} with finite values
     * @param id    id in {@code [0, size())}
     * @return exact squared L2 distance against the unquantized vector
     * @throws IllegalStateException if full-precision retention is disabled
     * @throws IllegalArgumentException if the query or id is invalid
     */
    public float fullPrecisionDist(float[] query, int id) {
        if (!keepFullPrecision) {
            throw new IllegalStateException("full-precision vectors were not retained");
        }
        checkQuery(query);
        checkId(id);
        int base = id * dimension;
        float sum = 0f;
        for (int d = 0; d < dimension; d++) {
            float delta = query[d] - fullPrecision[base + d];
            sum += delta * delta;
        }
        return sum;
    }

    /**
     * Resets per-dimension calibration from a sample corpus. Existing codes, if
     * any, are re-encoded under the new ranges; this method is intended to be
     * called before vectors are inserted.
     *
     * <p>Thread-safety: call before publishing the store to readers; must not be
     * called concurrently with writes or reads.
     *
     * @param store   store whose calibration is replaced
     * @param samples non-empty corpus of vectors of length {@link VectorStore#dimension()}
     * @throws IllegalArgumentException if the corpus is empty, a sample has the
     *                                  wrong dimension or contains non-finite values
     */
    public static void calibrate(Int8VectorStore store, float[][] samples) {
        Objects.requireNonNull(store, "store must not be null");
        Objects.requireNonNull(samples, "samples must not be null");
        if (samples.length == 0) {
            throw new IllegalArgumentException("samples must be non-empty");
        }
        float[] oldMin = store.min;
        float[] oldMax = store.max;
        boolean wasCalibrated = store.calibrated;
        float[] newMin = new float[store.dimension];
        float[] newMax = new float[store.dimension];
        Arrays.fill(newMin, Float.POSITIVE_INFINITY);
        Arrays.fill(newMax, Float.NEGATIVE_INFINITY);
        for (float[] sample : samples) {
            Objects.requireNonNull(sample, "sample must not be null");
            if (sample.length != store.dimension) {
                throw new IllegalArgumentException("dimension mismatch: expected "
                        + store.dimension + ", got " + sample.length);
            }
            for (int d = 0; d < store.dimension; d++) {
                float value = sample[d];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("sample values must be finite");
                }
                if (value < newMin[d]) {
                    newMin[d] = value;
                }
                if (value > newMax[d]) {
                    newMax[d] = value;
                }
            }
        }
        store.min = newMin;
        store.max = newMax;
        store.calibrated = true;
        if (wasCalibrated) {
            for (int id = 0; id < store.size.get(); id++) {
                store.reencode(id, oldMin, oldMax, newMin, newMax);
            }
        }
    }

    /**
     * Returns the number of bytes used by the quantized code array.
     *
     * <p>Thread-safety: safe for concurrent reads.
     *
     * @return {@code capacity() * dimension()} bytes
     */
    public long codeBytes() {
        return codes.length;
    }

    /**
     * Releases resources. This implementation is a no-op and may be called more
     * than once.
     *
     * <p>Thread-safety: safe to call at any time.
     */
    @Override
    public void close() {
    }

    private void encode(int id, float[] vector) {
        int base = id * dimension;
        for (int d = 0; d < dimension; d++) {
            codes[base + d] = (byte) encode(vector[d], min[d], max[d]);
        }
    }

    private static int encode(float value, float lo, float hi) {
        float span = hi - lo;
        if (span <= 0f) {
            return 0;
        }
        float scaled = (value - lo) / span * MAX_CODE;
        if (scaled <= 0f) {
            return 0;
        }
        if (scaled >= MAX_CODE) {
            return MAX_CODE;
        }
        return Math.round(scaled);
    }

    private void reencode(int id, float[] oldMin, float[] oldMax, float[] newMin, float[] newMax) {
        int base = id * dimension;
        for (int d = 0; d < dimension; d++) {
            float value = keepFullPrecision ? fullPrecision[base + d]
                    : oldMin[d] + ((codes[base + d] & 0xFF) / (float) MAX_CODE)
                            * (oldMax[d] - oldMin[d]);
            codes[base + d] = (byte) encode(value, newMin[d], newMax[d]);
        }
    }

    private void dequantize(int id, float[] out) {
        int base = id * dimension;
        for (int d = 0; d < dimension; d++) {
            out[d] = min[d] + ((codes[base + d] & 0xFF) / (float) MAX_CODE)
                    * (max[d] - min[d]);
        }
    }

    private void checkId(int id) {
        if (id < 0 || id >= size.get()) {
            throw new IllegalArgumentException("id outside [0, size()): " + id);
        }
    }

    private void checkQuery(float[] query) {
        Objects.requireNonNull(query, "query must not be null");
        if (query.length != dimension) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + dimension + ", got " + query.length);
        }
        for (float value : query) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("query values must be finite");
            }
        }
    }
}
