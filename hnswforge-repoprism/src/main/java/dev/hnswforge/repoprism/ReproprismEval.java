package dev.hnswforge.repoprism;

import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Command-line evaluation of exact, HNSWForge-only and hybrid RRF retrieval.
 *
 * <p>Usage:
 * <pre>
 * repoprism-eval --embeddings chunks.jsonl --queries eval72.jsonl --gold gold.jsonl [--out report.csv]
 * </pre>
 * Embedding lines are {@code {"chunk_id":"...","embedding":[f,...]}}, query lines are
 * {@code {"query_id":"...","embedding":[f,...]}} and gold lines are
 * {@code {"query_id":"...","chunk_ids":["...",...]}}. The parser handles these flat
 * objects directly and reports the offending line number for malformed input.
 *
 * <p>Hit@6 is the fraction of queries whose gold chunk ids intersect the top six
 * results. The hybrid leg fuses the exact and HNSWForge ranked lists with reciprocal
 * rank fusion using {@code 1/(60+rank)} and one-based ranks. pgvector-only numbers
 * cannot be derived from this fixture and are explicitly not reported.
 *
 * <p>Thread-safety: all evaluation state is local to a call; the methods that write
 * to the shared standard streams are not safe for concurrent invocations.
 */
public final class ReproprismEval {

    private static final int K = 6;
    private static final int RRF_CONSTANT = 60;
    private static final Set<String> ALLOWED_OPTIONS =
            Set.of("--embeddings", "--queries", "--gold", "--out");

    private ReproprismEval() {
    }

    /**
     * Runs the CLI and terminates the JVM with the returned exit code.
     *
     * <p>Not thread-safe for concurrent invocations because it writes the shared
     * standard streams.
     *
     * @param args the command-line options; must not be null
     */
    public static void main(String[] args) {
        System.exit(run(args));
    }

    /**
     * Runs the CLI without terminating the JVM, for tests and embedders.
     *
     * <p>Not thread-safe for concurrent invocations because it writes the shared
     * standard streams.
     *
     * @param args the command-line options; may be null or empty to request usage
     * @return {@code 0} on success, {@code 2} on bad arguments, {@code 1} on I/O or data failure
     */
    public static int run(String[] args) {
        try {
            Map<String, String> options = parseOptions(args);
            Path embeddings = Path.of(required(options, "--embeddings"));
            Path queries = Path.of(required(options, "--queries"));
            Path gold = Path.of(required(options, "--gold"));
            Path out = options.containsKey("--out") ? Path.of(options.get("--out")) : null;
            EvalResult result = evaluate(embeddings, queries, gold);
            printTable(result);
            if (out != null) {
                writeCsv(out, result);
            }
            System.out.println("Note: pgvector-only results require a real RepoPrism export"
                    + " and are not computed from this fixture.");
            return 0;
        } catch (IllegalArgumentException failure) {
            System.err.println("error: " + failure.getMessage());
            printUsage();
            return 2;
        } catch (IOException failure) {
            System.err.println("error: " + failure.getMessage());
            return 1;
        }
    }

    /**
     * Evaluates a fixture and returns hit@6 for each configuration.
     *
     * <p>The exact leg uses a {@link BruteForceIndex} with cosine similarity, the
     * approximate leg uses a temporary {@link HnswForgeBackend} instance and the
     * hybrid leg applies RRF to both ranked lists. The temporary index directory is
     * removed and all indexes are closed before this method returns.
     *
     * <p>Thread-safety: no shared mutable state; safe for concurrent invocations.
     *
     * @param embeddingsPath JSONL chunk embeddings
     * @param queriesPath    JSONL query embeddings
     * @param goldPath       JSONL relevance judgments
     * @return the measured hit@6 values
     * @throws IOException              if a fixture cannot be read or the temporary directory fails
     * @throws IllegalArgumentException if a fixture is malformed or inconsistent
     * @throws NullPointerException     if any path is null
     */
    public static EvalResult evaluate(Path embeddingsPath, Path queriesPath,
                                      Path goldPath) throws IOException {
        Objects.requireNonNull(embeddingsPath, "embeddingsPath must not be null");
        Objects.requireNonNull(queriesPath, "queriesPath must not be null");
        Objects.requireNonNull(goldPath, "goldPath must not be null");
        Corpus corpus = readEmbeddings(embeddingsPath);
        List<Query> queries = readQueries(queriesPath, corpus.dimension());
        Map<String, List<String>> gold = readGold(goldPath, queries);

        BruteForceIndex exact = new BruteForceIndex(corpus.dimension(), Metric.COSINE);
        Path tempDirectory = Files.createTempDirectory("repoprism-eval-");
        try {
            for (float[] vector : corpus.vectors()) {
                exact.add(vector);
            }
            try (HnswForgeBackend backend = new HnswForgeBackend(tempDirectory, corpus.dimension())) {
                for (int i = 0; i < corpus.chunkIds().size(); i++) {
                    backend.upsert(corpus.chunkIds().get(i), corpus.vectors().get(i));
                }
                int exactHits = 0;
                int hnswHits = 0;
                int hybridHits = 0;
                for (Query query : queries) {
                    List<String> exactIds = new ArrayList<>();
                    for (SearchResult result : exact.search(query.embedding(), K)) {
                        exactIds.add(corpus.chunkIds().get(result.id()));
                    }
                    List<String> hnswIds = backend.search(query.embedding(), K).stream()
                            .map(ScoredChunk::chunkId)
                            .toList();
                    List<String> relevant = gold.getOrDefault(query.queryId(), List.of());
                    if (containsAny(relevant, exactIds)) {
                        exactHits++;
                    }
                    if (containsAny(relevant, hnswIds)) {
                        hnswHits++;
                    }
                    if (containsAny(relevant, rrf(exactIds, hnswIds))) {
                        hybridHits++;
                    }
                }
                int count = queries.size();
                return new EvalResult(count, (double) exactHits / count,
                        (double) hnswHits / count, (double) hybridHits / count);
            }
        } finally {
            exact.close();
            deleteRecursively(tempDirectory);
        }
    }

    /**
     * Measured hit@6 values for one fixture.
     *
     * <p>Thread-safety: immutable record; inherently thread-safe.
     *
     * @param queryCount   number of evaluated queries, positive
     * @param exactHitAt6  hit@6 of the exact leg, in {@code [0, 1]}
     * @param hnswHitAt6   hit@6 of the HNSWForge leg, in {@code [0, 1]}
     * @param hybridHitAt6 hit@6 of the RRF hybrid leg, in {@code [0, 1]}
     */
    public record EvalResult(int queryCount, double exactHitAt6, double hnswHitAt6,
                             double hybridHitAt6) {

        /** Validates that all rates are finite fractions and the query count is positive. */
        public EvalResult {
            if (queryCount <= 0) {
                throw new IllegalArgumentException("queryCount must be > 0, got " + queryCount);
            }
            validateRate(exactHitAt6, "exactHitAt6");
            validateRate(hnswHitAt6, "hnswHitAt6");
            validateRate(hybridHitAt6, "hybridHitAt6");
        }

        private static void validateRate(double rate, String name) {
            if (!Double.isFinite(rate) || rate < 0.0 || rate > 1.0) {
                throw new IllegalArgumentException(name + " must be in [0, 1], got " + rate);
            }
        }
    }

    private record Corpus(List<String> chunkIds, List<float[]> vectors, int dimension) {
    }

    private record Query(String queryId, float[] embedding) {
    }

    private static Map<String, String> parseOptions(String[] args) {
        if (args == null) {
            throw new IllegalArgumentException("args must not be null");
        }
        Map<String, String> options = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String flag = args[i];
            if (!ALLOWED_OPTIONS.contains(flag)) {
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

    private static String required(Map<String, String> options, String flag) {
        String value = options.get(flag);
        if (value == null) {
            throw new IllegalArgumentException("Missing required option " + flag);
        }
        return value;
    }

    private static void printTable(EvalResult result) {
        System.out.printf(Locale.ROOT, "%-16s %9s %10s%n", "configuration", "hit@6", "queries");
        System.out.printf(Locale.ROOT, "%-16s %8.2f%% %10d%n", "exact",
                100.0 * result.exactHitAt6(), result.queryCount());
        System.out.printf(Locale.ROOT, "%-16s %8.2f%% %10d%n", "hnswforge",
                100.0 * result.hnswHitAt6(), result.queryCount());
        System.out.printf(Locale.ROOT, "%-16s %8.2f%% %10d%n", "hybrid_rrf",
                100.0 * result.hybridHitAt6(), result.queryCount());
    }

    private static void writeCsv(Path out, EvalResult result) throws IOException {
        Path parent = out.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        List<String> lines = List.of(
                "configuration,hit_at_6,query_count",
                String.format(Locale.ROOT, "exact,%.6f,%d",
                        result.exactHitAt6(), result.queryCount()),
                String.format(Locale.ROOT, "hnswforge,%.6f,%d",
                        result.hnswHitAt6(), result.queryCount()),
                String.format(Locale.ROOT, "hybrid_rrf,%.6f,%d",
                        result.hybridHitAt6(), result.queryCount()));
        Files.write(out, lines, StandardCharsets.UTF_8);
    }

    private static void printUsage() {
        System.err.println("Usage:");
        System.err.println("  repoprism-eval --embeddings chunks.jsonl --queries eval72.jsonl"
                + " --gold gold.jsonl [--out report.csv]");
    }

    private static Corpus readEmbeddings(Path path) throws IOException {
        List<String> chunkIds = new ArrayList<>();
        List<float[]> vectors = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int dimension = -1;
        int lineNumber = 0;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            lineNumber++;
            if (line.isBlank()) {
                continue;
            }
            try {
                Map<String, Object> object = new JsonParser(line).parseObject();
                String chunkId = requireString(object, "chunk_id");
                if (!seen.add(chunkId)) {
                    throw new IllegalArgumentException("duplicate chunk_id '" + chunkId + "'");
                }
                float[] vector = requireEmbedding(object, "embedding");
                if (isZero(vector)) {
                    throw new IllegalArgumentException("embedding must not be a zero vector");
                }
                if (dimension < 0) {
                    dimension = vector.length;
                } else if (vector.length != dimension) {
                    throw new IllegalArgumentException("dimension mismatch: expected " + dimension
                            + ", got " + vector.length);
                }
                chunkIds.add(chunkId);
                vectors.add(vector);
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException(
                        path + " line " + lineNumber + ": " + failure.getMessage(), failure);
            }
        }
        if (chunkIds.isEmpty()) {
            throw new IllegalArgumentException(path + ": no chunk embeddings found");
        }
        return new Corpus(List.copyOf(chunkIds), List.copyOf(vectors), dimension);
    }

    private static List<Query> readQueries(Path path, int dimension) throws IOException {
        List<Query> queries = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int lineNumber = 0;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            lineNumber++;
            if (line.isBlank()) {
                continue;
            }
            try {
                Map<String, Object> object = new JsonParser(line).parseObject();
                String queryId = requireString(object, "query_id");
                if (!seen.add(queryId)) {
                    throw new IllegalArgumentException("duplicate query_id '" + queryId + "'");
                }
                float[] vector = requireEmbedding(object, "embedding");
                if (vector.length != dimension) {
                    throw new IllegalArgumentException("dimension mismatch: expected " + dimension
                            + ", got " + vector.length);
                }
                if (isZero(vector)) {
                    throw new IllegalArgumentException("embedding must not be a zero vector");
                }
                queries.add(new Query(queryId, vector));
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException(
                        path + " line " + lineNumber + ": " + failure.getMessage(), failure);
            }
        }
        if (queries.isEmpty()) {
            throw new IllegalArgumentException(path + ": no queries found");
        }
        return List.copyOf(queries);
    }

    private static Map<String, List<String>> readGold(Path path, List<Query> queries)
            throws IOException {
        Set<String> knownQueries = new HashSet<>();
        for (Query query : queries) {
            knownQueries.add(query.queryId());
        }
        Map<String, List<String>> gold = new HashMap<>();
        int lineNumber = 0;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            lineNumber++;
            if (line.isBlank()) {
                continue;
            }
            try {
                Map<String, Object> object = new JsonParser(line).parseObject();
                String queryId = requireString(object, "query_id");
                if (!knownQueries.contains(queryId)) {
                    throw new IllegalArgumentException("unknown query_id '" + queryId + "'");
                }
                List<String> chunkIds = requireStrings(object, "chunk_ids");
                if (gold.putIfAbsent(queryId, chunkIds) != null) {
                    throw new IllegalArgumentException("duplicate query_id '" + queryId + "'");
                }
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException(
                        path + " line " + lineNumber + ": " + failure.getMessage(), failure);
            }
        }
        return Map.copyOf(gold);
    }

    private static String requireString(Map<String, Object> object, String key) {
        Object value = object.get(key);
        if (!(value instanceof String text) || text.isBlank()) {
            throw new IllegalArgumentException("missing or invalid '" + key + "'");
        }
        return text;
    }

    private static List<String> requireStrings(Map<String, Object> object, String key) {
        Object value = object.get(key);
        if (!(value instanceof List<?> items)) {
            throw new IllegalArgumentException("missing or invalid '" + key + "'");
        }
        List<String> result = new ArrayList<>();
        for (Object item : items) {
            if (!(item instanceof String text) || text.isBlank()) {
                throw new IllegalArgumentException("'" + key + "' must contain only non-blank strings");
            }
            result.add(text);
        }
        if (result.isEmpty()) {
            throw new IllegalArgumentException("'" + key + "' must not be empty");
        }
        return List.copyOf(result);
    }

    private static float[] requireEmbedding(Map<String, Object> object, String key) {
        Object value = object.get(key);
        if (!(value instanceof List<?> items)) {
            throw new IllegalArgumentException("missing or invalid '" + key + "'");
        }
        float[] vector = new float[items.size()];
        for (int i = 0; i < items.size(); i++) {
            Object item = items.get(i);
            if (!(item instanceof Double number)) {
                throw new IllegalArgumentException("'" + key + "' must contain only numbers");
            }
            float component = number.floatValue();
            if (!Float.isFinite(component)) {
                throw new IllegalArgumentException("'" + key + "' must contain only finite numbers");
            }
            vector[i] = component;
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException("'" + key + "' must not be empty");
        }
        return vector;
    }

    private static boolean isZero(float[] vector) {
        for (float component : vector) {
            if (component != 0.0f) {
                return false;
            }
        }
        return true;
    }

    private static boolean containsAny(List<String> relevant, List<String> ranked) {
        for (String id : relevant) {
            if (ranked.contains(id)) {
                return true;
            }
        }
        return false;
    }

    private static List<String> rrf(List<String> exactIds, List<String> hnswIds) {
        Map<String, Double> scores = new HashMap<>();
        accumulateRrf(scores, exactIds);
        accumulateRrf(scores, hnswIds);
        List<String> fused = new ArrayList<>(scores.keySet());
        fused.sort(Comparator.<String>comparingDouble(scores::get).reversed()
                .thenComparing(Comparator.naturalOrder()));
        if (fused.size() > K) {
            return List.copyOf(fused.subList(0, K));
        }
        return List.copyOf(fused);
    }

    private static void accumulateRrf(Map<String, Double> scores, List<String> ranked) {
        for (int rank = 1; rank <= ranked.size(); rank++) {
            scores.merge(ranked.get(rank - 1), 1.0 / (RRF_CONSTANT + rank), Double::sum);
        }
    }

    private static void deleteRecursively(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    private static final class JsonParser {

        private final String text;
        private int pos;

        private JsonParser(String text) {
            this.text = text;
        }

        private Map<String, Object> parseObject() {
            skipWhitespace();
            expect('{');
            Map<String, Object> result = new LinkedHashMap<>();
            skipWhitespace();
            if (peek() == '}') {
                pos++;
            } else {
                while (true) {
                    skipWhitespace();
                    String key = parseString();
                    skipWhitespace();
                    expect(':');
                    skipWhitespace();
                    Object value = parseValue();
                    if (result.containsKey(key)) {
                        throw error("duplicate key '" + key + "'");
                    }
                    result.put(key, value);
                    skipWhitespace();
                    char separator = next();
                    if (separator == '}') {
                        break;
                    }
                    if (separator != ',') {
                        throw error("expected ',' or '}'");
                    }
                }
            }
            skipWhitespace();
            if (pos != text.length()) {
                throw error("trailing content after JSON object");
            }
            return result;
        }

        private Object parseValue() {
            char next = peek();
            if (next == '"') {
                return parseString();
            }
            if (next == '[') {
                return parseArray();
            }
            if (next == '-' || (next >= '0' && next <= '9')) {
                return parseNumber();
            }
            throw error("expected a string, number or array");
        }

        private List<Object> parseArray() {
            expect('[');
            List<Object> result = new ArrayList<>();
            skipWhitespace();
            if (peek() == ']') {
                pos++;
                return result;
            }
            while (true) {
                skipWhitespace();
                result.add(parseValue());
                skipWhitespace();
                char separator = next();
                if (separator == ']') {
                    return result;
                }
                if (separator != ',') {
                    throw error("expected ',' or ']'");
                }
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder value = new StringBuilder();
            while (true) {
                char current = next();
                if (current == '"') {
                    return value.toString();
                }
                if (current == '\\') {
                    char escaped = next();
                    value.append(switch (escaped) {
                        case '"' -> '"';
                        case '\\' -> '\\';
                        case '/' -> '/';
                        case 'b' -> '\b';
                        case 'f' -> '\f';
                        case 'n' -> '\n';
                        case 'r' -> '\r';
                        case 't' -> '\t';
                        default -> throw error("invalid escape '\\" + escaped + "'");
                    });
                } else {
                    if (current < 0x20) {
                        throw error("unescaped control character in string");
                    }
                    value.append(current);
                }
            }
        }

        private Double parseNumber() {
            int start = pos;
            if (peek() == '-') {
                pos++;
            }
            while (pos < text.length() && isDigit(text.charAt(pos))) {
                pos++;
            }
            if (pos < text.length() && text.charAt(pos) == '.') {
                pos++;
                while (pos < text.length() && isDigit(text.charAt(pos))) {
                    pos++;
                }
            }
            if (pos < text.length() && (text.charAt(pos) == 'e' || text.charAt(pos) == 'E')) {
                pos++;
                if (pos < text.length() && (text.charAt(pos) == '+' || text.charAt(pos) == '-')) {
                    pos++;
                }
                while (pos < text.length() && isDigit(text.charAt(pos))) {
                    pos++;
                }
            }
            String literal = text.substring(start, pos);
            try {
                return Double.valueOf(literal);
            } catch (NumberFormatException failure) {
                throw error("invalid number '" + literal + "'");
            }
        }

        private static boolean isDigit(char value) {
            return value >= '0' && value <= '9';
        }

        private void skipWhitespace() {
            while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) {
                pos++;
            }
        }

        private char peek() {
            if (pos >= text.length()) {
                throw error("unexpected end of input");
            }
            return text.charAt(pos);
        }

        private char next() {
            char current = peek();
            pos++;
            return current;
        }

        private void expect(char expected) {
            char actual = next();
            if (actual != expected) {
                throw error("expected '" + expected + "' but found '" + actual + "'");
            }
        }

        private IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message);
        }
    }
}
