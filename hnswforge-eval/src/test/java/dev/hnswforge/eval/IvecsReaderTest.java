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

class IvecsReaderTest {

  @TempDir
  Path tempDir;

  @Test
  void roundTripsTinyFile() throws IOException {
    Path file = tempDir.resolve("tiny.ivecs");
    ByteBuffer buffer = ByteBuffer.allocate(2 * Integer.BYTES + 6 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(3).putInt(10).putInt(-20).putInt(0);
    buffer.putInt(3).putInt(30).putInt(40).putInt(50);
    Files.write(file, buffer.array());

    int[][] actual = IvecsReader.read(file);

    assertEquals(2, actual.length);
    assertArrayEquals(new int[] {10, -20, 0}, actual[0]);
    assertArrayEquals(new int[] {30, 40, 50}, actual[1]);
  }
}
