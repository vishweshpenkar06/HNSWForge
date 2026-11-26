package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FvecsReaderTest {

  @TempDir
  Path tempDir;

  @Test
  void roundTripsTinyFile() throws IOException {
    float[][] expected = {{1.0f, 2.0f}, {-3.5f, 4.25f}};
    Path file = tempDir.resolve("tiny.fvecs");
    Files.write(file, fvecs(expected));

    float[][] actual = FvecsReader.read(file);

    assertEquals(expected.length, actual.length);
    for (int i = 0; i < expected.length; i++) {
      assertArrayEquals(expected[i], actual[i]);
    }
  }

  @Test
  void dimensionHelperReadsFirstHeader() throws IOException {
    Path file = tempDir.resolve("dim.fvecs");
    Files.write(file, fvecs(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}}));

    assertEquals(3, FvecsReader.dimension(file));
  }

  @Test
  void truncatedFileThrows() throws IOException {
    Path file = tempDir.resolve("truncated.fvecs");
    ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES + 2 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(3).putFloat(1f).putFloat(2f);
    Files.write(file, buffer.array());

    IOException error = assertThrows(IOException.class, () -> FvecsReader.read(file));

    assertTrue(error.getMessage().contains("Truncated"), error.getMessage());
  }

  @Test
  void mismatchedDimensionsThrow() throws IOException {
    Path file = tempDir.resolve("mismatch.fvecs");
    ByteBuffer buffer = ByteBuffer.allocate(3 * Integer.BYTES + 5 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(2).putFloat(1f).putFloat(2f);
    buffer.putInt(3).putFloat(3f).putFloat(4f).putFloat(5f);
    Files.write(file, buffer.array());

    IOException error = assertThrows(IOException.class, () -> FvecsReader.read(file));

    assertTrue(error.getMessage().contains("Dimension mismatch"), error.getMessage());
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
}
