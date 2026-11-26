package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.index.SearchResult;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MainTest {

  @TempDir
  Path tempDir;

  @Test
  void groundTruthCommandWritesIvecs() throws IOException {
    Path base = tempDir.resolve("base.fvecs");
    Path queries = tempDir.resolve("queries.fvecs");
    Path truth = tempDir.resolve("truth.ivecs");
    Files.write(base, fvecs(new float[][] {{0f, 0f}, {0f, 1f}, {0f, 3f}}));
    Files.write(queries, fvecs(new float[][] {{0f, 0f}}));

    int code = Main.run(new String[] {"groundtruth", "--base", base.toString(), "--queries", queries.toString(),
        "--metric", "L2", "--k", "2", "--out", truth.toString()});

    assertEquals(0, code);
    assertArrayEquals(new int[] {0, 1}, GroundTruth.readIvecs(truth)[0]);
  }

  @Test
  void recallCommandPrintsRecall() throws IOException {
    Path truth = tempDir.resolve("truth.ivecs");
    Path results = tempDir.resolve("results.txt");
    GroundTruth.writeIvecs(truth, new int[][] {{0, 1, 2}});
    Recall.writeResultIds(results, List.of(List.of(
        new SearchResult(2, 0f), new SearchResult(1, 1f), new SearchResult(0, 2f))));
    AtomicInteger code = new AtomicInteger(-1);

    String output = captureStdout(() -> code.set(Main.run(new String[] {"recall",
        "--truth", truth.toString(), "--results", results.toString(), "--k", "3"})));

    assertEquals(0, code.get(), output);
    assertTrue(output.contains("1.000000"), output);
  }

  @Test
  void badArgumentsReturnTwo() {
    assertEquals(2, Main.run(new String[] {"bogus"}));
    assertEquals(2, Main.run(new String[0]));
    assertEquals(2, Main.run(new String[] {"recall", "--truth", "missing.ivecs"}));
  }

  private static String captureStdout(Runnable action) {
    PrintStream original = System.out;
    ByteArrayOutputStream captured = new ByteArrayOutputStream();
    try {
      System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));
      action.run();
    } finally {
      System.setOut(original);
    }
    return captured.toString(StandardCharsets.UTF_8);
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
