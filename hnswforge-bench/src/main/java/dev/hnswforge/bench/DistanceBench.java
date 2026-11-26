package dev.hnswforge.bench;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.distance.SimdDistance;
import dev.hnswforge.index.Metric;
import dev.hnswforge.util.SyntheticData;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

/** JMH comparison of scalar and SIMD distance kernels by dimension and metric. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
public class DistanceBench {

    @Param({"1", "3", "7", "16", "32", "100", "128", "960"})
    public int dim;

    @Param({"L2", "COSINE", "INNER_PRODUCT"})
    public String metricName;

    private float[] a;
    private float[] b;
    private DistanceFunction scalar;
    private DistanceFunction simd;

    /** Generates vectors and kernels for one parameter set. */
    @Setup
    public void setup() {
        float[][] vectors = SyntheticData.uniform(2, dim, 42L);
        a = vectors[0];
        b = vectors[1];
        Metric metric = Metric.valueOf(metricName);
        scalar = ScalarDistance.forMetric(metric);
        simd = SimdDistance.forMetric(metric);
    }

    /** Measures the scalar kernel. */
    @Benchmark
    public float scalar() {
        return scalar.distance(a, b);
    }

    /** Measures the Vector API kernel. */
    @Benchmark
    public float simd() {
        return simd.distance(a, b);
    }
}
