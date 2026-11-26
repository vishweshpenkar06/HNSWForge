package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.hnswforge.index.Metric;

import java.io.IOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GroundTruthTest {

  @TempDir
  Path tempDir;

  @Test
  void computeMatchesExpectedIds() {
    float[][] base = {{0f, 0f}, {0f, 1f}, {0f, 3f}, {0f, 10f}};
    float[][] queries = {{0f, 0f}, {0f, 10f}};

    int[][] truth = GroundTruth.compute(base, queries, Metric.L2, 2);

    assertEquals(2, truth.length);
    assertArrayEquals(new int[] {0, 1}, truth[0]);
    assertArrayEquals(new int[] {3, 2}, truth[1]);
  }

  @Test
  void ivecsRoundTrip() throws IOException {
    int[][] expected = {{1, 2, 3}, {4, 5, 6}};
    Path file = tempDir.resolve("truth.ivecs");

    GroundTruth.writeIvecs(file, expected);
    int[][] actual = GroundTruth.readIvecs(file);

    assertEquals(expected.length, actual.length);
    for (int i = 0; i < expected.length; i++) {
      assertArrayEquals(expected[i], actual[i]);
    }
  }

  @Test
  void invalidKAndNaNThrow() {
    float[][] base = {{0f, 0f}, {1f, 1f}};
    float[][] queries = {{0f, 0f}};

    assertThrows(IllegalArgumentException.class, () -> GroundTruth.compute(base, queries, Metric.L2, 0));
    assertThrows(IllegalArgumentException.class, () -> GroundTruth.compute(base, queries, Metric.L2, -1));
    assertThrows(IllegalArgumentException.class,
        () -> GroundTruth.compute(new float[][] {{Float.NaN, 0f}}, queries, Metric.L2, 1));
    assertThrows(IllegalArgumentException.class,
        () -> GroundTruth.compute(base, new float[][] {{0f}}, Metric.L2, 1));
  }
}
