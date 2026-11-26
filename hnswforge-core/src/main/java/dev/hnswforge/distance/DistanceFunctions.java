package dev.hnswforge.distance;

import dev.hnswforge.index.Metric;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Chooses between SIMD and scalar distance kernels at runtime.
 *
 * <p>This class never links {@code SimdDistance} directly: availability is probed
 * with {@code Class.forName} and the SIMD kernel is reached through a lazily loaded
 * holder, so environments without {@code jdk.incubator.vector} can still load this
 * class and use scalar kernels.
 *
 * <p>Thread-safety: stateless; safe for concurrent use. The fallback warning is
 * logged at most once.
 */
public final class DistanceFunctions {

    /** Whether {@code jdk.incubator.vector} is available at runtime. */
    public static final boolean SIMD_AVAILABLE;

    private static final Logger LOGGER = Logger.getLogger(DistanceFunctions.class.getName());
    private static final AtomicBoolean WARNED = new AtomicBoolean();

    static {
        boolean available;
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            available = true;
        } catch (ClassNotFoundException | LinkageError e) {
            available = false;
        }
        SIMD_AVAILABLE = available;
    }

    private DistanceFunctions() {
    }

    /**
     * Returns the kernel for a metric, preferring SIMD when requested and available.
     *
     * <p>Thread-safe: immutable lookup; a requested-but-unavailable SIMD choice logs
     * one warning and falls back to the scalar kernel.
     *
     * @param metric     metric to compute
     * @param preferSimd whether to use the SIMD kernel when possible
     * @return the selected distance kernel
     * @throws NullPointerException if {@code metric} is null
     */
    public static DistanceFunction forMetric(Metric metric, boolean preferSimd) {
        Objects.requireNonNull(metric, "metric must not be null");
        if (preferSimd) {
            if (SIMD_AVAILABLE) {
                return SimdHolder.forMetric(metric);
            }
            if (WARNED.compareAndSet(false, true)) {
                LOGGER.warning(
                        "SIMD requested but jdk.incubator.vector is unavailable; using scalar kernels");
            }
        }
        return ScalarDistance.forMetric(metric);
    }

    /**
     * Reports whether SIMD kernels can be used.
     *
     * <p>Thread-safe: reads an immutable flag.
     *
     * @return {@code true} when the incubator vector module is present
     */
    public static boolean simdAvailable() {
        return SIMD_AVAILABLE;
    }

    private static final class SimdHolder {
        private static DistanceFunction forMetric(Metric metric) {
            return SimdDistance.forMetric(metric);
        }
    }
}
