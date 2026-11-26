package dev.hnswforge.bench;

import dev.hnswforge.eval.FvecsReader;
import dev.hnswforge.eval.GroundTruth;
import dev.hnswforge.eval.Recall;
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
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * End-to-end comparison harness for HNSWForge, Lucene, pgvector and precomputed hnswlib results.
 *
 * <p>Contract: each in-process system is built once and searched with three warmup
 * passes followed by five timed passes per {@code efSearch} value. Recall@K uses one
 * exact ground-truth computation shared by all systems; QPS and latency percentiles
 * come from the timed passes; {@code hnswlib} rows are copied from a CSV after
 * filtering by n, dim and k, and are never remeasured. Output is one row per
 * (system, efSearch) with the header
 * {@code system,dataset,n,dim,metric,m,efc,efs,k,recall,qps,p50_us,p95_us,p99_us,build_s,index_bytes_est}.
 * Bad arguments and missing dataset or hnswlib files produce exit code 2.
 *
 * <p>Thread-safety: this class is stateless and all work runs on the calling thread;
 * instances of the underlying systems are created, searched and closed within
 * {@link #run(String[])}. It is safe to call {@code run} concurrently in separate
 * invocations that write to different output paths.
 */
public final class CompareMain {
    private static final String[] HNSWLIB_COLUMNS = {
        "dataset", "n", "dim", "metric", "m", "efc", "efs", "k", "recall",
        "qps", "p50_us", "p95_us", "p99_us", "build_s", "index_bytes_est"
    };
    private static final String HEADER = "system," + String.join(",", HNSWLIB_COLUMNS);
    private static final Set<String> SUPPORTED_SYSTEMS =
            Set.of("hnswforge", "lucene", "pgvector", "hnswlib");
    private static final int QUERY_COUNT = 200;
    private static final int WARMUP_PASSES = 3;
    private static final int TIMED_PASSES = 5;
    private static final int DEFAULT_N = 10_000;
    private static final int DEFAULT_M = 16;
    private static final int DEFAULT_EFC = 200;
    private static final int DEFAULT_K = 10;
    private static final String DEFAULT_EFS = "10,20,40,80,160,320";

    private CompareMain() {
    }

    /**
     * Runs the harness and exits with the returned status when it is non-zero.
     *
     * <p>Thread-safety: safe to call once from the main thread.
     *
     * @param args command-line arguments described by {@code --help}
     * @throws Exception if the JVM cannot be asked to exit with a status
     */
    public static void main(String[] args) throws Exception {
        int status = run(args);
        if (status != 0) {
            System.exit(status);
        }
    }

    /**
     * Parses arguments, builds and measures the requested systems, and writes the CSV.
     *
     * <p>Thread-safety: all state is local to the invocation.
     *
     * @param args command-line arguments; must not be null
     * @return {@code 0} on success, {@code 2} for bad arguments or missing input files
     * @throws IllegalArgumentException if generated data, ground truth or a system rejects a value
     * @throws IllegalStateException if a baseline cannot be built, searched or closed
     */
    public static int run(String[] args) {
        if (args != null && Arrays.asList(args).contains("--help")) {
            usage();
            return 0;
        }
        Args parsed;
        try {
            parsed = Args.parse(args);
        } catch (IllegalArgumentException exception) {
            System.err.println("error: " + exception.getMessage());
            usage();
            return 2;
        }
        try {
            Corpus corpus = loadCorpus(parsed);
            int[][] truth = GroundTruth.compute(corpus.base(), corpus.queries(), Metric.L2,
                    parsed.k());
            List<String> rows = new ArrayList<>();
            for (String name : parsed.systems()) {
                if (name.equals("hnswlib")) {
                    rows.addAll(mergeHnswlib(parsed, corpus));
                    continue;
                }
                BuiltSystem built = build(name, parsed, corpus);
                try {
                    for (int ef : parsed.efs()) {
                        rows.add(measureRow(name, built, parsed, corpus, truth, ef));
                    }
                } finally {
                    built.system().close();
                }
            }
            writeCsv(parsed.out(), rows);
        } catch (IOException exception) {
            System.err.println("error: " + exception.getMessage());
            return 2;
        }
        return 0;
    }

    private static void usage() {
        System.err.println("Usage: CompareMain --dataset synthetic|sift1m [--n N]"
                + " --systems hnswforge,lucene,pgvector,hnswlib [--hnswlib-csv PATH]"
                + " [--efs 10,20,40,80,160,320] [--k 10] [--out results/compare.csv]"
                + " [--m 16] [--efc 200]");
    }

    private static Corpus loadCorpus(Args args) throws IOException {
        if (args.dataset().equals("synthetic")) {
            float[][] base = SyntheticData.uniform(args.n(), 32, 123L);
            return new Corpus(base, SyntheticData.queries(base, QUERY_COUNT, 456L), 32);
        }
        Path basePath = Path.of("data/sift_base.fvecs");
        Path queryPath = Path.of("data/sift_query.fvecs");
        if (!Files.isRegularFile(basePath) || !Files.isRegularFile(queryPath)) {
            throw new IOException("sift1m dataset files are missing: expected data/sift_base.fvecs"
                    + " and data/sift_query.fvecs relative to the working directory");
        }
        float[][] loaded = FvecsReader.read(basePath);
        if (loaded.length < args.n()) {
            throw new IOException("sift1m base has " + loaded.length + " vectors, but n="
                    + args.n() + " was requested");
        }
        float[][] base = Arrays.copyOf(loaded, args.n());
        float[][] queries = FvecsReader.read(queryPath);
        if (queries.length == 0) {
            throw new IOException("sift1m query file is empty: " + queryPath);
        }
        int dimension = base[0].length;
        for (int i = 0; i < queries.length; i++) {
            if (queries[i].length != dimension) {
                throw new IOException("sift1m query " + i + " has dimension " + queries[i].length
                        + " but expected " + dimension);
            }
        }
        return new Corpus(base, queries, dimension);
    }

    private static BuiltSystem build(String name, Args args, Corpus corpus) {
        long start = System.nanoTime();
        ComparisonSystem system = switch (name) {
            case "hnswforge" -> buildForge(args, corpus);
            case "lucene" -> {
                LuceneHnswBaseline baseline = new LuceneHnswBaseline(corpus.dimension(),
                        args.m(), args.efc(), corpus.base().length);
                baseline.addAll(corpus.base());
                yield baseline;
            }
            case "pgvector" -> {
                PgvectorBaseline baseline = new PgvectorBaseline(corpus.dimension(),
                        args.m(), args.efc());
                baseline.createSchema();
                baseline.addAll(corpus.base());
                yield baseline;
            }
            default -> throw new IllegalArgumentException("unsupported system: " + name);
        };
        return new BuiltSystem(system, (System.nanoTime() - start) / 1e9);
    }

    private static ComparisonSystem buildForge(Args args, Corpus corpus) {
        int maxEf = 0;
        for (int ef : args.efs()) {
            maxEf = Math.max(maxEf, ef);
        }
        HnswParams params = HnswParams.builder(corpus.dimension(), Metric.L2)
                .m(args.m()).efConstruction(args.efc()).efSearch(maxEf)
                .maxElements(corpus.base().length).seed(42L).build();
        HnswIndex index = new HnswIndex(params);
        index.addAll(corpus.base(), 1);
        return new ForgeSystem(index, corpus.base().length, corpus.dimension(), args.m(), maxEf);
    }

    private static String measureRow(String name, BuiltSystem built, Args args, Corpus corpus,
                                     int[][] truth, int ef) {
        built.system().setEfSearch(ef);
        for (int pass = 0; pass < WARMUP_PASSES; pass++) {
            for (float[] query : corpus.queries()) {
                built.system().search(query, args.k());
            }
        }
        long[] latenciesNanos = new long[corpus.queries().length * TIMED_PASSES];
        long totalNanos = 0L;
        int at = 0;
        for (int pass = 0; pass < TIMED_PASSES; pass++) {
            for (float[] query : corpus.queries()) {
                long start = System.nanoTime();
                built.system().search(query, args.k());
                long elapsed = System.nanoTime() - start;
                latenciesNanos[at++] = elapsed;
                totalNanos += elapsed;
            }
        }
        List<List<SearchResult>> results = new ArrayList<>(corpus.queries().length);
        for (float[] query : corpus.queries()) {
            results.add(built.system().search(query, args.k()));
        }
        double recall = Recall.recallAtK(truth, results, args.k());
        double qps = latenciesNanos.length * 1e9 / totalNanos;
        long[] sorted = latenciesNanos.clone();
        Arrays.sort(sorted);
        return String.format(Locale.ROOT,
                "%s,%s,%d,%d,%s,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%.6f,%d",
                name, args.dataset(), corpus.base().length, corpus.dimension(), "L2",
                args.m(), args.efc(), ef, args.k(), recall, qps,
                percentileUs(sorted, 50.0), percentileUs(sorted, 95.0),
                percentileUs(sorted, 99.0), built.buildSeconds(),
                built.system().indexBytesEstimate());
    }

    private static double percentileUs(long[] sortedNanos, double percentile) {
        if (sortedNanos.length == 0) {
            return 0.0;
        }
        double rank = percentile / 100.0 * (sortedNanos.length - 1);
        int lower = (int) Math.floor(rank);
        int upper = (int) Math.ceil(rank);
        double fraction = rank - lower;
        double value = sortedNanos[lower]
                + (sortedNanos[upper] - sortedNanos[lower]) * fraction;
        return value / 1000.0;
    }

    private static List<String> mergeHnswlib(Args args, Corpus corpus) throws IOException {
        if (!Files.isRegularFile(args.hnswlibCsv())) {
            throw new IOException("hnswlib CSV not found: " + args.hnswlibCsv());
        }
        List<String> lines = Files.readAllLines(args.hnswlibCsv(), StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("hnswlib CSV is empty: " + args.hnswlibCsv());
        }
        String[] header = lines.get(0).split(",", -1);
        Map<String, Integer> columns = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            columns.put(header[i].trim(), i);
        }
        for (String column : HNSWLIB_COLUMNS) {
            if (!columns.containsKey(column)) {
                throw new IOException("hnswlib CSV is missing column '" + column + "': "
                        + args.hnswlibCsv());
            }
        }
        List<String> rows = new ArrayList<>();
        for (int index = 1; index < lines.size(); index++) {
            String line = lines.get(index);
            if (line.isBlank()) {
                continue;
            }
            String[] fields = line.split(",", -1);
            int number = columnInt(fields, columns, "n", index + 1, args.hnswlibCsv());
            int dimension = columnInt(fields, columns, "dim", index + 1, args.hnswlibCsv());
            int k = columnInt(fields, columns, "k", index + 1, args.hnswlibCsv());
            if (number != args.n() || dimension != corpus.dimension() || k != args.k()) {
                continue;
            }
            StringBuilder row = new StringBuilder("hnswlib");
            for (String column : HNSWLIB_COLUMNS) {
                row.append(',').append(columnText(fields, columns, column, index + 1,
                        args.hnswlibCsv()));
            }
            rows.add(row.toString());
        }
        return rows;
    }

    private static String columnText(String[] fields, Map<String, Integer> columns, String name,
                                     int lineNumber, Path path) throws IOException {
        int index = columns.get(name);
        if (index >= fields.length) {
            throw new IOException("malformed hnswlib CSV line " + lineNumber + " in " + path
                    + ": missing value for " + name);
        }
        return fields[index].trim();
    }

    private static int columnInt(String[] fields, Map<String, Integer> columns, String name,
                                 int lineNumber, Path path) throws IOException {
        String text = columnText(fields, columns, name, lineNumber, path);
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException exception) {
            throw new IOException("malformed hnswlib CSV line " + lineNumber + " in " + path
                    + ": invalid integer for " + name, exception);
        }
    }

    private static void writeCsv(Path out, List<String> rows) throws IOException {
        Path parent = out.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writer.write(HEADER);
            writer.newLine();
            for (String row : rows) {
                writer.write(row);
                writer.newLine();
            }
        }
    }

    private record Args(String dataset, int n, List<String> systems, Path hnswlibCsv,
                        int[] efs, int k, Path out, int m, int efc) {

        private static Args parse(String[] args) {
            if (args == null) {
                throw new IllegalArgumentException("args must not be null");
            }
            String dataset = null;
            String systemsText = null;
            String hnswlibCsv = "results/hnswlib.csv";
            String efsText = DEFAULT_EFS;
            String out = "results/compare.csv";
            int n = DEFAULT_N;
            int k = DEFAULT_K;
            int m = DEFAULT_M;
            int efc = DEFAULT_EFC;
            for (int i = 0; i < args.length; i++) {
                String flag = args[i];
                switch (flag) {
                    case "--dataset" -> dataset = next(args, ++i, flag);
                    case "--n" -> n = integer(next(args, ++i, flag), flag);
                    case "--systems" -> systemsText = next(args, ++i, flag);
                    case "--hnswlib-csv" -> hnswlibCsv = next(args, ++i, flag);
                    case "--efs" -> efsText = next(args, ++i, flag);
                    case "--k" -> k = integer(next(args, ++i, flag), flag);
                    case "--out" -> out = next(args, ++i, flag);
                    case "--m" -> m = integer(next(args, ++i, flag), flag);
                    case "--efc" -> efc = integer(next(args, ++i, flag), flag);
                    default -> throw new IllegalArgumentException("unknown option: " + flag);
                }
            }
            if (!"synthetic".equals(dataset) && !"sift1m".equals(dataset)) {
                throw new IllegalArgumentException("--dataset must be synthetic or sift1m");
            }
            if (systemsText == null) {
                throw new IllegalArgumentException("--systems is required");
            }
            if (n <= 0) {
                throw new IllegalArgumentException("--n must be positive");
            }
            if (k <= 0 || k > n) {
                throw new IllegalArgumentException("--k must be in [1, n]");
            }
            if (m < 2 || m > 100) {
                throw new IllegalArgumentException("--m must be in [2, 100]");
            }
            if (efc < m) {
                throw new IllegalArgumentException("--efc must be >= --m");
            }
            if (out.isBlank() || hnswlibCsv.isBlank()) {
                throw new IllegalArgumentException("--out and --hnswlib-csv must not be blank");
            }
            List<String> systems = parseSystems(systemsText);
            int[] efs = parseEfs(efsText);
            return new Args(dataset, n, systems, Path.of(hnswlibCsv), efs, k, Path.of(out), m, efc);
        }

        private static String next(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException("missing value for " + flag);
            }
            return args[index];
        }

        private static int integer(String text, String flag) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException("invalid integer for " + flag + ": " + text);
            }
        }

        private static List<String> parseSystems(String text) {
            Set<String> systems = new LinkedHashSet<>();
            for (String token : text.split(",", -1)) {
                String name = token.trim();
                if (!SUPPORTED_SYSTEMS.contains(name)) {
                    throw new IllegalArgumentException("unknown system '" + name
                            + "'; expected hnswforge, lucene, pgvector or hnswlib");
                }
                systems.add(name);
            }
            if (systems.isEmpty()) {
                throw new IllegalArgumentException("--systems must not be empty");
            }
            return List.copyOf(systems);
        }

        private static int[] parseEfs(String text) {
            String[] tokens = text.split(",", -1);
            int[] efs = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                int ef = integer(tokens[i].trim(), "--efs");
                if (ef <= 0) {
                    throw new IllegalArgumentException("--efs values must be positive");
                }
                efs[i] = ef;
            }
            return efs;
        }
    }

    private record Corpus(float[][] base, float[][] queries, int dimension) {
    }

    private record BuiltSystem(ComparisonSystem system, double buildSeconds) {
    }

    private static final class ForgeSystem implements ComparisonSystem {
        private final HnswIndex index;
        private final long indexBytes;
        private int ef;

        private ForgeSystem(HnswIndex index, int count, int dimension, int m, int ef) {
            this.index = index;
            this.ef = ef;
            this.indexBytes = (long) count * ((long) dimension * Float.BYTES
                    + (long) (1 + 2 * m) * Integer.BYTES + 12L);
        }

        @Override
        public List<SearchResult> search(float[] query, int k) {
            return index.search(query, k, ef);
        }

        @Override
        public void setEfSearch(int value) {
            this.ef = value;
        }

        @Override
        public long indexBytesEstimate() {
            return indexBytes;
        }

        @Override
        public void close() {
            index.close();
        }
    }
}
