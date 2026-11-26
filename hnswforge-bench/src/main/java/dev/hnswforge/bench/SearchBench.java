package dev.hnswforge.bench;

import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.util.SyntheticData;
import java.util.List;
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

/** JMH end-to-end search benchmark at a fixed recall setting. */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
public class SearchBench {

    @Param({"10000"})
    public int elementCount;

    private HnswIndex index;
    private float[][] queries;
    private int cursor;

    /** Builds the index once per trial. */
    @Setup(Level.Trial)
    public void setup() {
        float[][] base = SyntheticData.uniform(elementCount, 32, 123L);
        queries = SyntheticData.queries(base, 100, 456L);
        HnswParams params = HnswParams.builder(32, Metric.L2)
                .m(16).efConstruction(200).efSearch(100)
                .maxElements(elementCount).seed(42L).build();
        index = new HnswIndex(params);
        for (float[] vector : base) {
            index.add(vector);
        }
        cursor = 0;
    }

    /** Measures one k-NN query. */
    @Benchmark
    public List<SearchResult> search() {
        return index.search(queries[cursor++ % queries.length], 10, 100);
    }
}
