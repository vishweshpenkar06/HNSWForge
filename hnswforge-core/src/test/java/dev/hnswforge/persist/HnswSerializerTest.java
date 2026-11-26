package dev.hnswforge.persist;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Round-trip and corruption tests for the version 1 binary format. */
class HnswSerializerTest {

    @TempDir
    Path tempDir;

    @Test
    void roundTripPreservesEveryArray() throws IOException {
        IndexSnapshot original = snapshot(Metric.L2, 42L);
        Path file = tempDir.resolve("graph.bin");
        HnswSerializer.save(original, file);

        IndexSnapshot loaded = HnswSerializer.load(file);

        assertEquals(original.params(), loaded.params());
        assertEquals(original.size(), loaded.size());
        assertEquals(original.entryPoint(), loaded.entryPoint());
        assertEquals(original.maxLevel(), loaded.maxLevel());
        assertArrayEquals(original.vectors(), loaded.vectors());
        assertArrayEquals(original.levels(), loaded.levels());
        assertArrayEquals(original.layer0(), loaded.layer0());
        assertArrayEquals(original.upper(), loaded.upper());
        assertNotSame(original.vectors(), loaded.vectors());
        assertNotSame(original.levels(), loaded.levels());
        assertNotSame(original.layer0(), loaded.layer0());
        assertNotSame(original.upper(), loaded.upper());
    }

    @Test
    void roundTripHandlesAnEmptyIndex() throws IOException {
        HnswParams params = HnswParams.builder(3, Metric.INNER_PRODUCT)
                .m(2).efConstruction(10).efSearch(4)
                .maxElements(8).seed(9L).build();
        IndexSnapshot original = new IndexSnapshot(params, 0, -1, -1,
                new float[0], new int[0], new int[0], new int[8][]);
        Path file = tempDir.resolve("empty.bin");
        HnswSerializer.save(original, file);

        IndexSnapshot loaded = HnswSerializer.load(file);

        assertEquals(0, loaded.size());
        assertEquals(-1, loaded.entryPoint());
        assertEquals(-1, loaded.maxLevel());
        assertArrayEquals(new float[0], loaded.vectors());
        assertArrayEquals(new int[0], loaded.levels());
        assertArrayEquals(new int[0], loaded.layer0());
        assertEquals(8, loaded.upper().length);
    }

    @Test
    void badMagicIsRejected() throws IOException {
        byte[] bytes = validBytes();
        bytes[0] = 'X';
        Path corrupt = write("bad-magic.bin", bytes);

        IndexFormatException error = assertThrows(IndexFormatException.class,
                () -> HnswSerializer.load(corrupt));

        assertTrue(error.getMessage().contains("magic"), error.getMessage());
    }

    @Test
    void badVersionIsRejected() throws IOException {
        byte[] bytes = validBytes();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(8, 2);
        Path corrupt = write("bad-version.bin", bytes);

        IndexFormatException error = assertThrows(IndexFormatException.class,
                () -> HnswSerializer.load(corrupt));

        assertTrue(error.getMessage().contains("version"), error.getMessage());
    }

    @Test
    void truncatedHeaderIsRejected() throws IOException {
        Path corrupt = write("short.bin", Arrays.copyOf(validBytes(), 100));

        IndexFormatException error = assertThrows(IndexFormatException.class,
                () -> HnswSerializer.load(corrupt));

        assertTrue(error.getMessage().contains("truncated"), error.getMessage());
    }

    @Test
    void truncatedSectionsAreRejected() throws IOException {
        byte[] bytes = validBytes();
        Path corrupt = write("short-sections.bin", Arrays.copyOf(bytes, bytes.length - 4));

        IndexFormatException error = assertThrows(IndexFormatException.class,
                () -> HnswSerializer.load(corrupt));

        assertTrue(error.getMessage().contains("length"), error.getMessage());
    }

    @Test
    void flippedSectionByteIsCaughtByCrc() throws IOException {
        byte[] bytes = validBytes();
        bytes[HnswSerializer.HEADER_SIZE + 3] ^= 0x01;
        Path corrupt = write("flipped.bin", bytes);

        IndexFormatException error = assertThrows(IndexFormatException.class,
                () -> HnswSerializer.load(corrupt));

        assertTrue(error.getMessage().contains("CRC"), error.getMessage());
    }

    @Test
    void v1FixtureLoadsSuccessfully() throws IOException {
        try (InputStream input = getClass().getResourceAsStream("/v1_fixture.bin")) {
            assertNotNull(input, "v1_fixture.bin must be on the test classpath");
            Path file = write("fixture.bin", input.readAllBytes());

            IndexSnapshot loaded = HnswSerializer.load(file);

            assertEquals(Metric.L2, loaded.params().metric());
            assertEquals(4, loaded.params().dimension());
            assertEquals(12, loaded.size());
            assertEquals(2, loaded.maxLevel());
            assertEquals(48, loaded.vectors().length);
            assertEquals(12, loaded.levels().length);
            assertEquals(12 * (1 + loaded.params().maxM0()), loaded.layer0().length);
            assertEquals(loaded.params().maxElements(), loaded.upper().length);
        }
    }

    @Test
    void invalidSnapshotIsRejectedBeforeWriting() throws IOException {
        IndexSnapshot original = snapshot(Metric.L2, 1L);
        int[] badLevels = original.levels().clone();
        badLevels[0] = 9;
        IndexSnapshot invalid = new IndexSnapshot(original.params(), original.size(),
                original.entryPoint(), original.maxLevel(), original.vectors(),
                badLevels, original.layer0(), original.upper());
        Path file = tempDir.resolve("invalid.bin");

        assertThrows(IllegalArgumentException.class, () -> HnswSerializer.save(invalid, file));
        assertTrue(Files.notExists(file));
    }

    private byte[] validBytes() throws IOException {
        Path file = tempDir.resolve("valid-source.bin");
        HnswSerializer.save(snapshot(Metric.L2, 42L), file);
        return Files.readAllBytes(file);
    }

    private Path write(String name, byte[] bytes) throws IOException {
        Path file = tempDir.resolve(name);
        Files.write(file, bytes);
        return file;
    }

    private static IndexSnapshot snapshot(Metric metric, long seed) {
        int dimension = 4;
        int size = 20;
        int m = 4;
        int maxElements = 24;
        HnswParams params = HnswParams.builder(dimension, metric)
                .m(m).efConstruction(20).efSearch(8)
                .maxElements(maxElements).seed(seed).build();
        float[] vectors = new float[size * dimension];
        for (int id = 0; id < size; id++) {
            for (int axis = 0; axis < dimension; axis++) {
                vectors[id * dimension + axis] = (id + 1) * 0.01f * (axis + 1);
            }
        }
        int[] levels = new int[size];
        for (int id = 0; id < size; id++) {
            levels[id] = id % 7 == 0 ? 2 : id % 3 == 0 ? 1 : 0;
        }
        int layer0Stride = 1 + params.maxM0();
        int[] layer0 = new int[size * layer0Stride];
        for (int id = 0; id < size; id++) {
            int offset = id * layer0Stride;
            layer0[offset] = 3;
            for (int index = 0; index < 3; index++) {
                layer0[offset + 1 + index] = (id + 1 + index) % size;
            }
        }
        int upperStride = 1 + m;
        int[][] upper = new int[maxElements][];
        for (int id = 0; id < size; id++) {
            int level = levels[id];
            if (level == 0) {
                continue;
            }
            upper[id] = new int[level * upperStride];
            for (int layer = 1; layer <= level; layer++) {
                int offset = (layer - 1) * upperStride;
                upper[id][offset] = 2;
                upper[id][offset + 1] = (id + 2) % size;
                upper[id][offset + 2] = (id + 5) % size;
            }
        }
        return new IndexSnapshot(params, size, 0, 2, vectors, levels, layer0, upper);
    }
}
