package dev.hnswforge.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal dependency-free HTTP JSON facade over {@link HnswIndex}, built on the JDK
 * {@code com.sun.net.httpserver} API.
 *
 * <p>Endpoints: {@code POST /vectors} accepts {@code {"vector":[f1,...]}} and returns
 * {@code 201 {"id":N}}; {@code POST /search} accepts {@code {"vector":[...],"k":N,"ef":N}}
 * with optional {@code ef} and returns {@code 200 {"results":[{"id":N,"distance":D},...]}};
 * {@code POST /snapshot} accepts {@code {"path":"/path/to/index.bin"}} and returns
 * {@code 200 {"saved":true}}; {@code GET /health} returns {@code 200 {"status":"ok","size":N}}.
 * Malformed or invalid requests return {@code 400 {"error":"..."}}, unexpected failures
 * return {@code 500 {"error":"..."}}, and every response has an {@code application/json}
 * content type. The JSON parser accepts only flat objects with string, integral and
 * finite numeric-array fields; unknown or duplicate fields are rejected.
 *
 * <p>Thread-safety: {@code HnswIndex} is thread-safe, so endpoints may be called
 * concurrently on the virtual-thread executor. {@link #start()}, {@link #stop(int)} and
 * {@link #port()} are safe to call concurrently with requests and with each other;
 * {@code start()} and {@code stop(int)} are idempotent.
 *
 * <p>A curl demo is in the README.
 */
public final class RestServer {
    private final HttpServer server;
    private final ExecutorService executor;
    private final HnswIndex index;
    private boolean started;

    /**
     * Creates the index and binds an HTTP server to {@code port}, then starts accepting
     * requests. Passing port {@code 0} selects an ephemeral port; {@link #port()} returns
     * the port actually bound.
     *
     * @param port        listening port, in {@code [0, 65535]}
     * @param dimension   vector dimension, must be positive
     * @param metric      distance metric, must not be null
     * @param maxElements index capacity, must be positive
     * @throws IllegalArgumentException if any argument is outside its valid range
     * @throws UncheckedIOException if the server cannot be bound
     */
    public RestServer(int port, int dimension, Metric metric, int maxElements) {
        if (port < 0 || port > 65535) {
            throw new IllegalArgumentException("port must be in [0, 65535], got " + port);
        }
        HnswParams params = HnswParams.builder(dimension, metric)
                .maxElements(maxElements)
                .build();
        this.index = new HnswIndex(params);
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        try {
            this.server = HttpServer.create(new InetSocketAddress(port), 0);
        } catch (IOException exception) {
            executor.shutdown();
            index.close();
            throw new UncheckedIOException("failed to bind HTTP server on port " + port, exception);
        }
        server.setExecutor(executor);
        server.createContext("/vectors", exchange -> serve(exchange, request -> {
            requireMethod(exchange, "POST");
            return vectors(readBody(exchange));
        }));
        server.createContext("/search", exchange -> serve(exchange, request -> {
            requireMethod(exchange, "POST");
            return search(readBody(exchange));
        }));
        server.createContext("/snapshot", exchange -> serve(exchange, request -> {
            requireMethod(exchange, "POST");
            return snapshot(readBody(exchange));
        }));
        server.createContext("/health", exchange -> serve(exchange, request -> {
            requireMethod(exchange, "GET");
            return health();
        }));
        start();
    }

    /**
     * Starts accepting requests. Idempotent: calling this after construction or a
     * previous start has no effect.
     *
     * <p>Thread-safety: synchronized with {@link #stop(int)}.
     */
    public synchronized void start() {
        if (!started) {
            server.start();
            started = true;
        }
    }

    /**
     * Stops accepting requests, shuts down the request executor and closes the index.
     * Idempotent, including after construction-time failures to bind.
     *
     * @param delaySeconds seconds to wait for in-flight exchanges before stopping, must be non-negative
     * @throws IllegalArgumentException if {@code delaySeconds} is negative
     *
     * <p>Thread-safety: synchronized with {@link #start()}; in-flight requests may still
     * complete while the index is closing.
     */
    public synchronized void stop(int delaySeconds) {
        if (delaySeconds < 0) {
            throw new IllegalArgumentException("delaySeconds must be >= 0, got " + delaySeconds);
        }
        if (started) {
            server.stop(delaySeconds);
            started = false;
        }
        executor.shutdown();
        index.close();
    }

    /**
     * Returns the port the server is bound to, which differs from the constructor
     * argument when that argument is {@code 0}.
     *
     * @return the bound listening port
     *
     * <p>Thread-safety: safe at any time after construction.
     */
    public int port() {
        return server.getAddress().getPort();
    }

    /**
     * Runs the server until the process is stopped, installing a shutdown hook that
     * stops it on SIGTERM. Defaults are port {@code 8080}, dimension {@code 128},
     * metric {@code L2} and capacity {@code 100000}; options are
     * {@code --port N --dim N --metric L2|COSINE|INNER_PRODUCT --max-elements N}.
     *
     * @param args optional option pairs; unrecognized flags and malformed values fail fast
     *
     * <p>Thread-safety: not applicable; this is the process entry point.
     */
    public static void main(String[] args) {
        int port = 8080;
        int dimension = 128;
        Metric metric = Metric.L2;
        int maxElements = 100_000;
        for (int i = 0; i < args.length; i++) {
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("missing value for " + args[i]);
            }
            String flag = args[i];
            String value = args[++i];
            switch (flag) {
                case "--port" -> port = Integer.parseInt(value);
                case "--dim" -> dimension = Integer.parseInt(value);
                case "--metric" -> metric = Metric.valueOf(value.toUpperCase(Locale.ROOT));
                case "--max-elements" -> maxElements = Integer.parseInt(value);
                default -> throw new IllegalArgumentException("unknown option: " + flag);
            }
        }
        RestServer server = new RestServer(port, dimension, metric, maxElements);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> server.stop(0)));
        System.out.println("HNSWForge REST server listening on port " + server.port());
    }

    private Response vectors(String body) {
        Map<String, Object> fields = new Json(body).object();
        checkFields(fields, Set.of("vector"), Set.of());
        float[] vector = vectorField(fields, "vector");
        checkDimension(vector);
        return new Response(201, "{\"id\":" + index.add(vector) + "}");
    }

    private Response search(String body) {
        Map<String, Object> fields = new Json(body).object();
        checkFields(fields, Set.of("vector", "k"), Set.of("ef"));
        float[] vector = vectorField(fields, "vector");
        checkDimension(vector);
        int k = integerField(fields, "k");
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        List<SearchResult> results;
        if (fields.containsKey("ef")) {
            int ef = integerField(fields, "ef");
            if (ef <= 0) {
                throw new IllegalArgumentException("ef must be > 0, got " + ef);
            }
            results = index.search(vector, k, ef);
        } else {
            results = index.search(vector, k);
        }
        StringBuilder json = new StringBuilder("{\"results\":[");
        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            float distance = result.distance();
            if (!Float.isFinite(distance)) {
                throw new IllegalStateException("search returned a non-finite distance");
            }
            if (i > 0) {
                json.append(',');
            }
            json.append("{\"id\":").append(result.id())
                    .append(",\"distance\":").append(Float.toString(distance)).append('}');
        }
        return new Response(200, json.append("]}").toString());
    }

    private Response snapshot(String body) {
        Map<String, Object> fields = new Json(body).object();
        checkFields(fields, Set.of("path"), Set.of());
        String path = stringField(fields, "path");
        if (path.isEmpty()) {
            throw new IllegalArgumentException("path must not be empty");
        }
        try {
            index.save(Path.of(path));
        } catch (InvalidPathException exception) {
            throw new IllegalArgumentException("invalid path: " + path, exception);
        } catch (IOException exception) {
            throw new IllegalStateException("failed to save index: " + exception.getMessage(), exception);
        }
        return new Response(200, "{\"saved\":true}");
    }

    private Response health() {
        return new Response(200, "{\"status\":\"ok\",\"size\":" + index.size() + "}");
    }

    private void checkDimension(float[] vector) {
        if (vector.length != index.dimension()) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + index.dimension() + ", got " + vector.length);
        }
    }

    private void serve(HttpExchange exchange, Endpoint endpoint) throws IOException {
        Response response;
        try {
            response = endpoint.handle(exchange);
        } catch (IllegalArgumentException exception) {
            response = new Response(400, error(exception.getMessage()));
        } catch (IOException exception) {
            response = new Response(400, error("bad request: " + exception.getMessage()));
        } catch (RuntimeException exception) {
            response = new Response(500, error(String.valueOf(exception.getMessage())));
        }
        byte[] bytes = response.body().getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(response.status(), bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static String readBody(HttpExchange exchange) throws IOException {
        try (InputStream in = exchange.getRequestBody()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void requireMethod(HttpExchange exchange, String expected) {
        String actual = exchange.getRequestMethod();
        if (!actual.equals(expected)) {
            throw new IllegalArgumentException("expected " + expected + ", got " + actual);
        }
    }

    private static void checkFields(Map<String, Object> fields, Set<String> required,
                                    Set<String> optional) {
        for (String key : required) {
            if (!fields.containsKey(key)) {
                throw new IllegalArgumentException("missing field: " + key);
            }
        }
        for (String key : fields.keySet()) {
            if (!required.contains(key) && !optional.contains(key)) {
                throw new IllegalArgumentException("unknown field: " + key);
            }
        }
    }

    private static float[] vectorField(Map<String, Object> fields, String name) {
        if (!(fields.get(name) instanceof List<?> numbers)) {
            throw new IllegalArgumentException(name + " must be an array of numbers");
        }
        float[] vector = new float[numbers.size()];
        for (int i = 0; i < vector.length; i++) {
            if (!(numbers.get(i) instanceof Double number)) {
                throw new IllegalArgumentException(name + " must contain only numbers");
            }
            float value = number.floatValue();
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " contains an out-of-range number");
            }
            vector[i] = value;
        }
        return vector;
    }

    private static int integerField(Map<String, Object> fields, String name) {
        if (!(fields.get(name) instanceof Double number) || number != Math.rint(number)
                || number < Integer.MIN_VALUE || number > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        return number.intValue();
    }

    private static String stringField(Map<String, Object> fields, String name) {
        if (!(fields.get(name) instanceof String text)) {
            throw new IllegalArgumentException(name + " must be a string");
        }
        return text;
    }

    private static String error(String message) {
        return "{\"error\":" + quote(String.valueOf(message)) + "}";
    }

    private static String quote(String value) {
        StringBuilder out = new StringBuilder(value.length() + 2).append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        String hex = Integer.toHexString(c);
                        out.append("\\u").append("0".repeat(4 - hex.length())).append(hex);
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.append('"').toString();
    }

    @FunctionalInterface
    private interface Endpoint {
        Response handle(HttpExchange exchange) throws IOException;
    }

    private record Response(int status, String body) {
    }

    private static final class Json {
        private final String text;
        private int pos;

        private Json(String text) {
            this.text = text;
        }

        private Map<String, Object> object() {
            expect('{');
            Map<String, Object> fields = new HashMap<>();
            skipWhitespace();
            if (consume('}')) {
                return fields;
            }
            do {
                skipWhitespace();
                String key = string();
                skipWhitespace();
                expect(':');
                skipWhitespace();
                if (fields.putIfAbsent(key, value()) != null) {
                    throw new IllegalArgumentException("duplicate field: " + key);
                }
                skipWhitespace();
            } while (consume(','));
            expect('}');
            skipWhitespace();
            if (pos != text.length()) {
                throw new IllegalArgumentException("unexpected trailing content");
            }
            return fields;
        }

        private Object value() {
            if (pos >= text.length()) {
                throw new IllegalArgumentException("expected a value");
            }
            return switch (text.charAt(pos)) {
                case '"' -> string();
                case '[' -> array();
                default -> number();
            };
        }

        private List<Double> array() {
            expect('[');
            List<Double> values = new ArrayList<>();
            skipWhitespace();
            if (consume(']')) {
                return List.copyOf(values);
            }
            do {
                skipWhitespace();
                values.add(number());
                skipWhitespace();
            } while (consume(','));
            expect(']');
            return List.copyOf(values);
        }

        private double number() {
            int start = pos;
            while (pos < text.length() && "+-0123456789.eE".indexOf(text.charAt(pos)) >= 0) {
                pos++;
            }
            String token = text.substring(start, pos);
            try {
                double parsed = Double.parseDouble(token);
                if (!Double.isFinite(parsed)) {
                    throw new IllegalArgumentException("numbers must be finite");
                }
                return parsed;
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException("invalid number: " + token, exception);
            }
        }

        private String string() {
            expect('"');
            StringBuilder out = new StringBuilder();
            while (pos < text.length()) {
                char c = text.charAt(pos++);
                if (c == '"') {
                    return out.toString();
                }
                if (c == '\\') {
                    if (pos >= text.length()) {
                        break;
                    }
                    char escaped = text.charAt(pos++);
                    switch (escaped) {
                        case '"', '\\', '/' -> out.append(escaped);
                        case 'b' -> out.append('\b');
                        case 'f' -> out.append('\f');
                        case 'n' -> out.append('\n');
                        case 'r' -> out.append('\r');
                        case 't' -> out.append('\t');
                        case 'u' -> {
                            if (pos + 4 > text.length()) {
                                throw new IllegalArgumentException("incomplete unicode escape");
                            }
                            String hex = text.substring(pos, pos + 4);
                            try {
                                out.append((char) Integer.parseInt(hex, 16));
                            } catch (NumberFormatException exception) {
                                throw new IllegalArgumentException("invalid unicode escape: " + hex, exception);
                            }
                            pos += 4;
                        }
                        default -> throw new IllegalArgumentException("invalid escape: \\" + escaped);
                    }
                } else if (c < 0x20) {
                    throw new IllegalArgumentException("control character in string");
                } else {
                    out.append(c);
                }
            }
            throw new IllegalArgumentException("unterminated string");
        }

        private void skipWhitespace() {
            while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) {
                pos++;
            }
        }

        private boolean consume(char expected) {
            if (pos < text.length() && text.charAt(pos) == expected) {
                pos++;
                return true;
            }
            return false;
        }

        private void expect(char expected) {
            if (!consume(expected)) {
                throw new IllegalArgumentException("expected '" + expected + "'");
            }
        }
    }
}
