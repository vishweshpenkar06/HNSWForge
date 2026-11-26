package dev.hnswforge.bench;

import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.util.SyntheticData;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

/** JMH insertion benchmark on an index that grows during the trial. */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
public class InsertBench {

    @Param({"1000"})
    public int elementCount;

    private HnswIndex index;
    private float[][] vectors;
    private int cursor;

    /** Prepares an empty index and the vectors to insert. */
    @Setup(Level.Trial)
    public void setup() {
        vectors = SyntheticData.uniform(elementCount, 32, 123L);
        HnswParams params = HnswParams.builder(32, Metric.L2)
                .m(16).efConstruction(200)
                .maxElements(elementCount).seed(42L).build();
        index = new HnswIndex(params);
        cursor = 0;
    }

    /** Measures one insertion. */
    @Benchmark
    public int add() {
        return index.add(vectors[cursor++]);
    }
}
