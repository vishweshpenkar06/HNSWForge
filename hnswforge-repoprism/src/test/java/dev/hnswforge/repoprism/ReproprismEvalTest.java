package dev.hnswforge.repoprism;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests the evaluation CLI's exit codes, hit@6 agreement and malformed-input reporting.
 *
 * <p>Fixtures are generated programmatically so no external data files are needed.
 */
class ReproprismEvalTest {

    private static final int DIMENSION = 16;
    private static final int CHUNK_COUNT = 10;

    @TempDir
    Path tempDir;

    /** The CLI succeeds and the approximate leg agrees with exact hit@6. */
    @Test
    void runReportsHnswHitAt6WithinToleranceOfExact() throws IOException {
        Path embeddings = writeEmbeddings();
        Path queries = writeQueries();
        Path gold = writeGold();
        Path csv = tempDir.resolve("report.csv");
        int exitCode = ReproprismEval.run(new String[] {
                "--embeddings", embeddings.toString(),
                "--queries", queries.toString(),
                "--gold", gold.toString(),
                "--out", csv.toString()});
        assertEquals(0, exitCode);
        assertTrue(Files.exists(csv));
        assertTrue(Files.readString(csv, StandardCharsets.UTF_8).contains("hnswforge"));

        ReproprismEval.EvalResult result = ReproprismEval.evaluate(embeddings, queries, gold);
        assertEquals(3, result.queryCount());
        assertEquals(1.0, result.exactHitAt6(), 1.0e-12);
        assertEquals(1.0, result.hnswHitAt6(), 1.0e-12);
        assertEquals(1.0, result.hybridHitAt6(), 1.0e-12);
        assertTrue(Math.abs(result.exactHitAt6() - result.hnswHitAt6()) <= 0.005,
                () -> "exact=" + result.exactHitAt6() + " hnsw=" + result.hnswHitAt6());
    }

    /** Bad arguments produce the usage exit code. */
    @Test
    void missingRequiredOptionIsAUsageError() {
        assertEquals(2, ReproprismEval.run(new String[] {"--embeddings", "chunks.jsonl"}));
    }

    /** Malformed JSONL is rejected with the offending line number. */
    @Test
    void malformedQueryLineReportsItsLineNumber() throws IOException {
        Path embeddings = writeEmbeddings();
        Path queries = tempDir.resolve("bad-queries.jsonl");
        Files.writeString(queries,
                "{\"query_id\":\"q0\",\"embedding\":[" + embedding(0) + "]}\n"
                        + "{\"query_id\":\"q1\",\"embedding\":[1,\"oops\"]}\n",
                StandardCharsets.UTF_8);
        Path gold = writeGold();
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> ReproprismEval.evaluate(embeddings, queries, gold));
        assertTrue(failure.getMessage().contains("line 2"), failure.getMessage());
    }

    private Path writeEmbeddings() throws IOException {
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < CHUNK_COUNT; i++) {
            lines.add("{\"chunk_id\":\"chunk-" + i + "\",\"embedding\":["
                    + embedding(i) + "]}");
        }
        Path path = tempDir.resolve("chunks.jsonl");
        Files.write(path, lines, StandardCharsets.UTF_8);
        return path;
    }

    private Path writeQueries() throws IOException {
        List<String> lines = List.of(
                "{\"query_id\":\"q0\",\"embedding\":[" + embedding(0) + "]}",
                "{\"query_id\":\"q1\",\"embedding\":[" + embedding(3) + "]}",
                "{\"query_id\":\"q2\",\"embedding\":[" + embedding(9) + "]}");
        Path path = tempDir.resolve("eval72.jsonl");
        Files.write(path, lines, StandardCharsets.UTF_8);
        return path;
    }

    private Path writeGold() throws IOException {
        List<String> lines = List.of(
                "{\"query_id\":\"q0\",\"chunk_ids\":[\"chunk-0\"]}",
                "{\"query_id\":\"q1\",\"chunk_ids\":[\"chunk-3\"]}",
                "{\"query_id\":\"q2\",\"chunk_ids\":[\"chunk-9\"]}");
        Path path = tempDir.resolve("gold.jsonl");
        Files.write(path, lines, StandardCharsets.UTF_8);
        return path;
    }

    private static String embedding(int index) {
        StringBuilder value = new StringBuilder();
        for (int dimension = 0; dimension < DIMENSION; dimension++) {
            if (dimension > 0) {
                value.append(',');
            }
            value.append(dimension == index ? "1.0" : "0.0");
        }
        return value.toString();
    }
}
