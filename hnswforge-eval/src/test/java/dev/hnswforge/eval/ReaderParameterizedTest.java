package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ReaderParameterizedTest {

    @TempDir
    Path tempDir;

    @ParameterizedTest
    @ValueSource(ints = {1, 3, 8, 16})
    void fvecsRoundTrip(int dimension) throws IOException {
        float[][] expected = {
            floatRow(dimension, 1.5f),
            floatRow(dimension, -2.25f),
            floatRow(dimension, 10f)
        };
        Path file = tempDir.resolve("vectors-" + dimension + ".fvecs");
        Files.write(file, fvecs(expected));

        float[][] actual = FvecsReader.read(file);

        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], 1e-6f);
        }
        assertEquals(dimension, FvecsReader.dimension(file));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 3, 8, 16})
    void ivecsRoundTrip(int dimension) throws IOException {
        int[][] expected = {
            intRow(dimension, 1),
            intRow(dimension, -5),
            intRow(dimension, 100)
        };
        Path file = tempDir.resolve("vectors-" + dimension + ".ivecs");
        Files.write(file, ivecs(expected));

        int[][] actual = IvecsReader.read(file);

        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i]);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 3, 8, 16})
    void bvecsRoundTrip(int dimension) throws IOException {
        byte[][] expected = {
            byteRow(dimension, 1),
            byteRow(dimension, -5),
            byteRow(dimension, 100)
        };
        Path file = tempDir.resolve("vectors-" + dimension + ".bvecs");
        Files.write(file, bvecs(expected));

        byte[][] actual = BvecsReader.read(file);

        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i]);
        }
    }

    private static float[] floatRow(int dimension, float start) {
        float[] row = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            row[i] = start + i;
        }
        return row;
    }

    private static int[] intRow(int dimension, int start) {
        int[] row = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            row[i] = start + i;
        }
        return row;
    }

    private static byte[] byteRow(int dimension, int start) {
        byte[] row = new byte[dimension];
        for (int i = 0; i < dimension; i++) {
            row[i] = (byte) (start + i);
        }
        return row;
    }

    private static byte[] fvecs(float[][] vectors) {
        int bytes = 0;
        for (float[] vector : vectors) {
            bytes += Integer.BYTES + vector.length * Float.BYTES;
        }
        ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] vector : vectors) {
            buffer.putInt(vector.length);
            for (float value : vector) {
                buffer.putFloat(value);
            }
        }
        return buffer.array();
    }

    private static byte[] ivecs(int[][] vectors) {
        int bytes = 0;
        for (int[] vector : vectors) {
            bytes += Integer.BYTES + vector.length * Integer.BYTES;
        }
        ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int[] vector : vectors) {
            buffer.putInt(vector.length);
            for (int value : vector) {
                buffer.putInt(value);
            }
        }
        return buffer.array();
    }

    private static byte[] bvecs(byte[][] vectors) {
        int bytes = 0;
        for (byte[] vector : vectors) {
            bytes += Integer.BYTES + vector.length;
        }
        ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (byte[] vector : vectors) {
            buffer.putInt(vector.length);
            buffer.put(vector);
        }
        return buffer.array();
    }
}
