package dev.hnswforge.repoprism;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class BackendParameterizedTest {

    @TempDir
    Path tempDir;

    @ParameterizedTest
    @CsvSource({"4, 0", "4, 3", "8, 0", "8, 7", "16, 0", "16, 15"})
    void upsertSearchRoundTrip(int dimension, int index) {
        Path directory = tempDir.resolve("rt-" + dimension + "-" + index);
        try (HnswForgeBackend backend = new HnswForgeBackend(directory, dimension)) {
            backend.upsert("chunk", oneHot(dimension, index));
            List<ScoredChunk> results = backend.search(oneHot(dimension, index), 1);
            assertEquals(1, results.size());
            assertEquals("chunk", results.get(0).chunkId());
            assertEquals(1.0f, results.get(0).score(), 1e-4f);
            assertEquals(1, backend.liveCount());
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {4, 8, 16})
    void reupsertKeepsOneLiveChunk(int dimension) {
        Path directory = tempDir.resolve("reupsert-" + dimension);
        try (HnswForgeBackend backend = new HnswForgeBackend(directory, dimension)) {
            backend.upsert("c", oneHot(dimension, 0));
            backend.upsert("c", oneHot(dimension, dimension - 1));
            List<ScoredChunk> results = backend.search(oneHot(dimension, dimension - 1), 5);
            assertEquals(1, results.size());
            assertEquals("c", results.get(0).chunkId());
            assertEquals(1, backend.liveCount());
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {4, 8, 16})
    void rebuildReplacesAllChunks(int dimension) {
        Path directory = tempDir.resolve("rebuild-" + dimension);
        try (HnswForgeBackend backend = new HnswForgeBackend(directory, dimension)) {
            backend.upsert("old", oneHot(dimension, 0));
            backend.rebuild(List.of(
                    new Chunk("a", oneHot(dimension, 0)),
                    new Chunk("b", oneHot(dimension, 1))));
            List<ScoredChunk> results = backend.search(oneHot(dimension, 0), 5);
            assertEquals(2, results.size());
            assertEquals("a", results.get(0).chunkId());
            assertTrue(results.stream().noneMatch(result -> result.chunkId().equals("old")));
            assertEquals(2, backend.liveCount());
        }
    }

    private static float[] oneHot(int dimension, int index) {
        float[] vector = new float[dimension];
        vector[index] = 1.0f;
        return vector;
    }
}
