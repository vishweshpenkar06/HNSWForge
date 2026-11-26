package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BvecsReaderTest {

  @TempDir
  Path tempDir;

  @Test
  void roundTripsTinyFile() throws IOException {
    Path file = tempDir.resolve("tiny.bvecs");
    ByteBuffer buffer = ByteBuffer.allocate(2 * Integer.BYTES + 6).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(3).put((byte) 1).put((byte) -2).put((byte) 0);
    buffer.putInt(3).put((byte) 100).put((byte) -128).put((byte) 0);
    Files.write(file, buffer.array());

    byte[][] actual = BvecsReader.read(file);

    assertEquals(2, actual.length);
    assertArrayEquals(new byte[] {1, -2, 0}, actual[0]);
    assertArrayEquals(new byte[] {100, -128, 0}, actual[1]);
  }
}
