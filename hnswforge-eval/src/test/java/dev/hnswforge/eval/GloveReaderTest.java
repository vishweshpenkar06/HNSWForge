package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GloveReaderTest {

  @TempDir
  Path tempDir;

  @Test
  void parsesSampleWithLimitAndWords() throws IOException {
    Path file = tempDir.resolve("sample.txt");
    Files.writeString(file, "alpha 1 2 3\n\nbeta 4 5 6\ngamma 7 8 9\n", StandardCharsets.UTF_8);

    float[][] limited = GloveReader.read(file, 2);

    assertEquals(2, limited.length);
    assertArrayEquals(new float[] {1f, 2f, 3f}, limited[0]);
    assertArrayEquals(new float[] {4f, 5f, 6f}, limited[1]);

    GloveReader.GloveData data = GloveReader.readWithWords(file, 0);

    assertEquals(List.of("alpha", "beta", "gamma"), data.words());
    assertEquals(3, data.vectors().length);
    assertArrayEquals(new float[] {7f, 8f, 9f}, data.vectors()[2]);
    assertEquals(3, GloveReader.read(file).length);
  }

  @Test
  void malformedLineThrowsWithLineNumber() throws IOException {
    Path file = tempDir.resolve("bad.txt");
    Files.writeString(file, "alpha 1 2 3\nbeta 4 nope 6\n", StandardCharsets.UTF_8);

    IOException error = assertThrows(IOException.class, () -> GloveReader.read(file));

    assertTrue(error.getMessage().contains("Line 2"), error.getMessage());
  }

  @Test
  void inconsistentFieldCountThrows() throws IOException {
    Path file = tempDir.resolve("ragged.txt");
    Files.writeString(file, "alpha 1 2 3\nbeta 4 5\n", StandardCharsets.UTF_8);

    IOException error = assertThrows(IOException.class, () -> GloveReader.read(file));

    assertTrue(error.getMessage().contains("Line 2"), error.getMessage());
  }
}
