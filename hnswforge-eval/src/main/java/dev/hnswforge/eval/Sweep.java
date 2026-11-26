package dev.hnswforge.eval;

import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.util.SyntheticData;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Runs a parameter sweep over HNSW build and search settings and writes a CSV report.
 *
 * <p>One exact ground-truth computation is reused for every configuration of a
 * dataset. Each index is warmed up with three query passes and then timed over
 * five passes. Per-query nanosecond latencies from the timed passes produce p50,
 * p95 and p99 values in microseconds, while the median of the five per-pass
 * throughput values is reported as QPS. Missing dataset files are not an error:
 * deterministic synthetic data is used instead and a warning is printed to
 * standard error.
 *
 * <p>Thread-safety: all mutable state is local to {@link #run(String[])} and
 * {@code HnswIndex} supports concurrent searches, so separate invocations do not
 * interfere. The shared standard error stream is used only for warnings.
 */
public final class Sweep {

    private static final String CSV_HEADER = "dataset,n,dim,metric,m,efc,efs,k,recall,qps,p50_us,p95_us,p99_us,build_s,index_bytes_est";
    private static final int WARMUP_PASSES = 3;
    private static final int TIMED_PASSES = 5;
    private static final int SYNTHETIC_DIMENSION = 32;
    private static final int SYNTHETIC_QUERIES = 200;
    private static final long SYNTHETIC_BASE_SEED = 123L;
    private static final long SYNTHETIC_QUERY_SEED = 456L;
    private static final long INDEX_SEED = 42L;

    private Sweep() {
    }

    /**
     * Runs the sweep described by the options after the {@code sweep} subcommand.
     *
     * <p>Accepted flags are {@code --dataset}, {@code --n}, {@code --m},
     * {@code --efc}, {@code --efs}, {@code --k}, {@code --threads} and
     * {@code --out}. Unknown, duplicate or valueless flags and out-of-range
     * values are rejected before any data is loaded. The output file is
     * overwritten and receives the exact header declared by this class.
     *
     * <p>Not thread-safe with respect to the shared standard streams; otherwise
     * stateless and safe for concurrent use.
     *
     * @param args the full argument array, including the subcommand at index zero
     * @throws IOException if the dataset cannot be read or the CSV cannot be written
     * @throws IllegalArgumentException if any option is missing, unknown, duplicated or invalid
     * @throws IllegalStateException if a query worker fails or is interrupted
     */
    public static void run(String[] args) throws IOException {
        Map<String, String> options = parseOptions(args);
        String dataset = option(options, "--dataset", "synthetic");
        if (!Set.of("synthetic", "sift1m", "glove100").contains(dataset)) {
            throw new IllegalArgumentException("--dataset must be synthetic, sift1m, or glove100 but was '" + dataset + "'");
        }
        int n = positiveInt(option(options, "--n", "10000"), "--n");
        int[] mValues = parseIntList(option(options, "--m", "8,16,32"), "--m");
        int[] efcValues = parseIntList(option(options, "--efc", "100,200,400"), "--efc");
        int[] efsValues = parseIntList(option(options, "--efs", "10,20,40,80,160,320"), "--efs");
        int k = positiveInt(option(options, "--k", "10"), "--k");
        int threads = positiveInt(option(options, "--threads", "1"), "--threads");
        String outValue = option(options, "--out", "results.csv");
        if (outValue.isBlank()) {
            throw new IllegalArgumentException("--out must not be blank");
        }
        for (int m : mValues) {
            if (m < 2 || m > 128) {
                throw new IllegalArgumentException("--m values must be in [2, 128] but was " + m);
            }
            for (int efc : efcValues) {
                if (efc < m) {
                    throw new IllegalArgumentException("--efc values must be >= each --m value; got efc=" + efc + " m=" + m);
                }
            }
        }
        Path out = Path.of(outValue);

        Dataset data = loadDataset(dataset, n);
        int count = data.base().length;
        int dim = data.base()[0].length;
        int[][] truth = GroundTruth.compute(data.base(), data.queries(), Metric.L2, k);
        int combinations = mValues.length * efcValues.length * efsValues.length;
        double bestRecall = -1.0;
        int bestM = 0;
        int bestEfc = 0;
        int bestEfs = 0;

        try (ExecutorService pool = threads > 1 ? Executors.newFixedThreadPool(threads) : null;
             BufferedWriter writer = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writer.write(CSV_HEADER);
            writer.newLine();
            for (int m : mValues) {
                for (int efc : efcValues) {
                    for (int efs : efsValues) {
                        HnswParams params = HnswParams.builder(dim, Metric.L2)
                                .m(m)
                                .efConstruction(efc)
                                .efSearch(efs)
                                .maxElements(count)
                                .seed(INDEX_SEED)
                                .build();
                        int queryCount = data.queries().length;
                        long[] allLatencies = new long[queryCount * TIMED_PASSES];
                        double[] passQps = new double[TIMED_PASSES];
                        AtomicReferenceArray<List<SearchResult>> lastResults =
                                new AtomicReferenceArray<>(queryCount);
                        double buildSeconds;
                        try (HnswIndex index = new HnswIndex(params)) {
                            long buildStart = System.nanoTime();
                            for (float[] vector : data.base()) {
                                index.add(vector);
                            }
                            buildSeconds = (System.nanoTime() - buildStart) / 1_000_000_000.0;
                            for (int pass = 0; pass < WARMUP_PASSES; pass++) {
                                runPass(index, data.queries(), k, threads, pool, null);
                            }
                            for (int pass = 0; pass < TIMED_PASSES; pass++) {
                                long passStart = System.nanoTime();
                                long[] latencies = runPass(index, data.queries(), k, threads, pool, lastResults);
                                long passElapsed = System.nanoTime() - passStart;
                                System.arraycopy(latencies, 0, allLatencies,
                                        pass * queryCount, queryCount);
                                passQps[pass] = queryCount / (passElapsed / 1_000_000_000.0);
                            }
                        }
                        long[] sortedLatencies = allLatencies.clone();
                        Arrays.sort(sortedLatencies);
                        Arrays.sort(passQps);
                        double recall = Recall.recallAtK(truth, snapshot(lastResults), k);
                        String row = String.format(Locale.ROOT,
                                "%s,%d,%d,%s,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%.6f,%d",
                                data.name(), count, dim, "L2", m, efc, efs, k, recall,
                                passQps[passQps.length / 2],
                                percentileUs(sortedLatencies, 0.50),
                                percentileUs(sortedLatencies, 0.95),
                                percentileUs(sortedLatencies, 0.99),
                                buildSeconds, estimateIndexBytes(count, dim, m));
                        writer.write(row);
                        writer.newLine();
                        if (recall > bestRecall) {
                            bestRecall = recall;
                            bestM = m;
                            bestEfc = efc;
                            bestEfs = efs;
                        }
                    }
                }
            }
        }
        System.out.printf(Locale.ROOT,
                "Sweep complete: %d combinations, dataset=%s n=%d dim=%d k=%d, best recall %.6f at m=%d efc=%d efs=%d, wrote %s%n",
                combinations, data.name(), count, dim, k, bestRecall, bestM, bestEfc, bestEfs, out);
    }

    /**
     * Parses the flags accepted by the sweep subcommand.
     *
     * @param args the full argument array
     * @return the parsed flag-to-value map, empty when no flags were supplied
     * @throws IllegalArgumentException on unknown, duplicate, or valueless flags
     */
    private static Map<String, String> parseOptions(String[] args) {
        Set<String> allowed = Set.of("--dataset", "--n", "--m", "--efc", "--efs", "--k", "--threads", "--out");
        Map<String, String> options = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            String flag = args[i];
            if (!allowed.contains(flag)) {
                throw new IllegalArgumentException("Unknown option: " + flag);
            }
            if (options.containsKey(flag)) {
                throw new IllegalArgumentException("Duplicate option: " + flag);
            }
            if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
                throw new IllegalArgumentException("Missing value for " + flag);
            }
            options.put(flag, args[++i]);
        }
        return options;
    }

    /**
     * Looks up an option or its default.
     *
     * @param options the parsed options
     * @param flag the flag to look up
     * @param fallback the value used when the flag is absent
     * @return the option value or the fallback
     */
    private static String option(Map<String, String> options, String flag, String fallback) {
        return options.getOrDefault(flag, fallback);
    }

    /**
     * Parses a positive integer option value.
     *
     * @param value the raw value
     * @param flag the flag name used in error messages
     * @return the parsed positive integer
     * @throws IllegalArgumentException if the value is not a positive integer
     */
    private static int positiveInt(String value, String flag) {
        try {
            int parsed = Integer.parseInt(value);
            if (parsed <= 0) {
                throw new NumberFormatException("not positive");
            }
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " must be a positive integer but was '" + value + "'");
        }
    }

    /**
     * Parses a comma-separated list of positive integers.
     *
     * @param value the raw value
     * @param flag the flag name used in error messages
     * @return the parsed values in declaration order, never empty
     * @throws IllegalArgumentException if the value is empty or contains a non-positive or malformed integer
     */
    private static int[] parseIntList(String value, String flag) {
        if (value.isBlank()) {
            throw new IllegalArgumentException(flag + " must not be blank");
        }
        String[] tokens = value.split(",", -1);
        int[] parsed = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            try {
                parsed[i] = Integer.parseInt(tokens[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(flag + " must be a comma-separated list of integers but was '" + value + "'");
            }
            if (parsed[i] <= 0) {
                throw new IllegalArgumentException(flag + " values must be positive but was " + parsed[i]);
            }
        }
        return parsed;
    }

    /**
     * Resolves the requested dataset, falling back to synthetic data with a warning.
     *
     * <p>Base vectors are limited to {@code n} rows. The dataset column of the
     * CSV identifies the data actually used, so a fallback is reported as
     * {@code synthetic}.
     *
     * @param requested the requested dataset name
     * @param n the maximum number of base vectors
     * @return a non-empty dataset snapshot
     * @throws IOException if a present dataset file is unreadable or inconsistent
     * @throws IllegalArgumentException if {@code n} is not positive
     */
    private static Dataset loadDataset(String requested, int n) throws IOException {
        if ("synthetic".equals(requested)) {
            return syntheticDataset(n);
        }
        boolean sift = "sift1m".equals(requested);
        List<Path> basePaths = sift
                ? List.of(Path.of("data/sift_base.fvecs"), Path.of("data/sift/sift_base.fvecs"))
                : List.of(Path.of("data/glove100.fvecs"), Path.of("data/glove100.txt"));
        List<Path> queryPaths = sift
                ? List.of(Path.of("data/sift_query.fvecs"), Path.of("data/sift/sift_query.fvecs"))
                : List.of(Path.of("data/glove100_query.fvecs"), Path.of("data/glove100_query.txt"),
                        Path.of("data/glove100_queries.txt"));
        Path basePath = firstExisting(basePaths);
        Path queryPath = firstExisting(queryPaths);
        if (basePath == null || queryPath == null) {
            System.err.println("warning: " + requested + " data files not found; using synthetic data instead");
            return syntheticDataset(n);
        }
        float[][] loadedBase = readVectors(basePath, n);
        float[][] queries = readVectors(queryPath, 0);
        if (loadedBase.length == 0 || queries.length == 0) {
            System.err.println("warning: " + requested + " data files are empty; using synthetic data instead");
            return syntheticDataset(n);
        }
        int dim = loadedBase[0].length;
        for (float[] query : queries) {
            if (query.length != dim) {
                throw new IOException("Query dimension " + query.length + " in " + queryPath
                        + " does not match base dimension " + dim + " in " + basePath);
            }
        }
        float[][] base = Arrays.copyOf(loadedBase, Math.min(n, loadedBase.length));
        if (loadedBase.length < n) {
            System.err.println("warning: " + basePath + " has only " + loadedBase.length
                    + " vectors; using " + base.length + " instead of " + n);
        }
        return new Dataset(requested, base, queries);
    }

    /**
     * Builds deterministic synthetic base and query vectors.
     *
     * @param n the number of base vectors, must be positive
     * @return a synthetic dataset snapshot
     */
    private static Dataset syntheticDataset(int n) {
        float[][] base = SyntheticData.uniform(n, SYNTHETIC_DIMENSION, SYNTHETIC_BASE_SEED);
        float[][] queries = SyntheticData.queries(base, SYNTHETIC_QUERIES, SYNTHETIC_QUERY_SEED);
        return new Dataset("synthetic", base, queries);
    }

    /**
     * Reads a vector file by extension.
     *
     * @param path the file to read
     * @param limit the maximum number of vectors for text files, non-positive for unlimited
     * @return the vectors in file order
     * @throws IOException if the file is unreadable or malformed
     */
    private static float[][] readVectors(Path path, int limit) throws IOException {
        return path.getFileName().toString().endsWith(".fvecs")
                ? FvecsReader.read(path)
                : GloveReader.read(path, limit);
    }

    /**
     * Returns the first candidate that is a regular file.
     *
     * @param candidates the paths to test, in priority order
     * @return the first existing path, or {@code null} when none exists
     */
    private static Path firstExisting(List<Path> candidates) {
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
        }
        return null;
    }

    /**
     * Executes one query pass, recording per-query latencies and optionally results.
     *
     * @param index the index to query
     * @param queries the queries in file order
     * @param k the number of neighbours requested
     * @param threads the worker count; a value of one uses the calling thread
     * @param pool the fixed thread pool used when {@code threads > 1}
     * @param results destination for the results, or {@code null} to skip storing them
     * @return nanosecond latency for each query
     * @throws IllegalStateException if a worker fails or is interrupted
     */
    private static long[] runPass(HnswIndex index, float[][] queries, int k, int threads,
                                  ExecutorService pool, AtomicReferenceArray<List<SearchResult>> results) {
        long[] latencies = new long[queries.length];
        if (threads == 1) {
            for (int i = 0; i < queries.length; i++) {
                long start = System.nanoTime();
                List<SearchResult> found = index.search(queries[i], k);
                latencies[i] = System.nanoTime() - start;
                if (results != null) {
                    results.set(i, found);
                }
            }
            return latencies;
        }
        List<Future<?>> futures = new ArrayList<>(queries.length);
        for (int i = 0; i < queries.length; i++) {
            int queryIndex = i;
            futures.add(pool.submit(() -> {
                long start = System.nanoTime();
                List<SearchResult> found = index.search(queries[queryIndex], k);
                latencies[queryIndex] = System.nanoTime() - start;
                if (results != null) {
                    results.set(queryIndex, found);
                }
            }));
        }
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while waiting for query tasks", e);
            } catch (ExecutionException e) {
                throw new IllegalStateException("Query task failed", e.getCause());
            }
        }
        return latencies;
    }

    /**
     * Copies the last pass results into an ordered list for recall evaluation.
     *
     * @param results the per-query result slots
     * @return the results in query order
     */
    private static List<List<SearchResult>> snapshot(AtomicReferenceArray<List<SearchResult>> results) {
        List<List<SearchResult>> snapshot = new ArrayList<>(results.length());
        for (int i = 0; i < results.length(); i++) {
            snapshot.add(results.get(i));
        }
        return snapshot;
    }

    /**
     * Converts a sorted latency sample to a percentile in microseconds.
     *
     * @param sortedNanos latencies in nanoseconds, sorted ascending and non-empty
     * @param fraction the percentile as a fraction in {@code (0, 1]}
     * @return the selected latency in microseconds
     */
    private static double percentileUs(long[] sortedNanos, double fraction) {
        int index = (int) Math.ceil(fraction * sortedNanos.length) - 1;
        index = Math.max(0, Math.min(index, sortedNanos.length - 1));
        return sortedNanos[index] / 1000.0;
    }

    /**
     * Estimates index memory using vectors, layer-zero links and upper-layer links.
     *
     * @param n the number of base vectors
     * @param dim the vector dimension
     * @param m the maximum degree above layer zero, at least two
     * @return the estimated size in bytes
     */
    private static long estimateIndexBytes(int n, int dim, int m) {
        long vectors = (long) n * dim * Integer.BYTES;
        long links = (long) n * (1L + 2L * m) * Integer.BYTES;
        long upper = ((long) n / (m - 1)) * (1L + m) * Integer.BYTES;
        return vectors + links + upper;
    }

    /**
     * Immutable dataset snapshot used by a sweep.
     *
     * @param name the dataset name written to the CSV
     * @param base the base vectors, non-empty
     * @param queries the query vectors, non-empty
     */
    private record Dataset(String name, float[][] base, float[][] queries) {
    }
}
