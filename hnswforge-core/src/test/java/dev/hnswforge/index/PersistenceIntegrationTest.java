package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.persist.IndexFormatException;
import dev.hnswforge.util.SyntheticData;
import java.util.List;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** End-to-end persistence checks through the public {@link HnswIndex} API. */
class PersistenceIntegrationTest {

    @Test
    void loadPreservesResultsForOneThousandQueries() throws Exception {
        float[][] base = SyntheticData.uniform(1_000, 8, 11L);
        float[][] queries = SyntheticData.uniform(1_000, 8, 12L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(8, Metric.L2).m(8).efConstruction(100)
                        .maxElements(1_000).seed(11L).build());
        Path path = Files.createTempFile("hnsw-roundtrip", ".bin");
        try {
            index.save(path);
            HnswIndex loaded = HnswIndex.load(path);
            assertEquals(index.size(), loaded.size());
            for (float[] query : queries) {
                assertEquals(index.search(query, 5), loaded.search(query, 5));
            }
            loaded.close();
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    void mappedLoadReturnsIdenticalResultsAndRejectsWrites() throws Exception {
        float[][] base = SyntheticData.uniform(500, 8, 21L);
        float[][] queries = SyntheticData.uniform(50, 8, 22L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(8, Metric.L2).m(8).efConstruction(100)
                        .maxElements(500).seed(21L).build());
        Path path = Files.createTempFile("hnsw-mapped", ".bin");
        try {
            index.save(path);
            VectorIndex mapped = HnswIndex.loadMapped(path);
            assertEquals(index.size(), mapped.size());
            for (float[] query : queries) {
                List<SearchResult> expected = index.search(query, 5);
                List<SearchResult> actual = mapped.search(query, 5);
                assertEquals(expected.size(), actual.size());
                for (int i = 0; i < expected.size(); i++) {
                    assertEquals(expected.get(i).id(), actual.get(i).id());
                    assertEquals(expected.get(i).distance(), actual.get(i).distance(), 1e-5f);
                }
            }
            assertThrows(UnsupportedOperationException.class,
                    () -> mapped.add(new float[8]));
            assertThrows(UnsupportedOperationException.class,
                    () -> mapped.save(path));
            mapped.close();
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    void corruptedHeaderIsRejected() throws Exception {
        float[][] base = SyntheticData.uniform(100, 4, 31L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(4, Metric.L2).m(8).efConstruction(80)
                        .maxElements(100).seed(31L).build());
        Path path = Files.createTempFile("hnsw-corrupt", ".bin");
        try {
            index.save(path);
            byte[] bytes = Files.readAllBytes(path);
            bytes[0] = 'X';
            Files.write(path, bytes);
            assertThrows(IndexFormatException.class, () -> HnswIndex.load(path));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    void truncatedFileIsRejected() throws Exception {
        float[][] base = SyntheticData.uniform(100, 4, 41L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(4, Metric.L2).m(8).efConstruction(80)
                        .maxElements(100).seed(41L).build());
        Path path = Files.createTempFile("hnsw-truncated", ".bin");
        try {
            index.save(path);
            byte[] bytes = Files.readAllBytes(path);
            byte[] truncated = java.util.Arrays.copyOf(bytes, bytes.length / 2);
            Files.write(path, truncated);
            assertThrows(IndexFormatException.class, () -> HnswIndex.load(path));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    void tombstonesSurviveSaveAndLoad() throws Exception {
        float[][] base = SyntheticData.uniform(300, 8, 51L);
        HnswIndex index = HnswTestUtil.build(base,
                HnswParams.builder(8, Metric.L2).m(8).efConstruction(100)
                        .maxElements(300).seed(51L).build());
        for (int id = 0; id < 30; id++) {
            index.delete(id);
        }
        Path path = Files.createTempFile("hnsw-tombstones", ".bin");
        try {
            index.save(path);
            HnswIndex loaded = HnswIndex.load(path);
            assertEquals(300, loaded.size());
            assertEquals(270, loaded.liveCount());
            float[] query = base[150];
            for (SearchResult result : loaded.search(query, 50, 100)) {
                assertTrue(result.id() >= 30, "deleted id reappeared: " + result.id());
            }
            loaded.close();
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    void embeddedV1FixtureLoads() throws Exception {
        Path fixture = Path.of("src/test/resources/v1_fixture.bin");
        assertTrue(Files.exists(fixture), "fixture missing");
        VectorIndex index = HnswIndex.load(fixture);
        assertTrue(index.size() > 0);
        index.close();
    }
}
