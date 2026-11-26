package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.hnswforge.index.SearchResult;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RecallTest {

  @TempDir
  Path tempDir;

  @Test
  void perfectResultsGiveOne() {
    int[][] truth = {{0, 1, 2}};
    List<List<SearchResult>> results = List.of(List.of(
        new SearchResult(2, 2f), new SearchResult(0, 0f), new SearchResult(1, 1f)));

    assertEquals(1.0, Recall.recallAtK(truth, results, 3), 1e-12);
  }

  @Test
  void emptyOverlapGivesZero() {
    int[][] truth = {{0, 1}};
    List<List<SearchResult>> results = List.of(List.of(new SearchResult(2, 0f), new SearchResult(3, 1f)));

    assertEquals(0.0, Recall.recallAtK(truth, results, 2), 1e-12);
  }

  @Test
  void partialOverlapGivesExactFraction() {
    int[][] truth = {{0, 1, 2, 3}};
    List<List<SearchResult>> results = List.of(List.of(
        new SearchResult(0, 0f), new SearchResult(3, 1f), new SearchResult(4, 2f)));

    assertEquals(0.5, Recall.recallAtK(truth, results, 4), 1e-12);
    assertEquals(0.5, Recall.recallAtK(truth, results, 2), 1e-12);
  }

  @Test
  void resultIdsRoundTripThroughText() throws IOException {
    List<List<SearchResult>> results = List.of(
        List.of(new SearchResult(7, 0f), new SearchResult(3, 0f)),
        List.of(new SearchResult(1, 0f), new SearchResult(9, 1f)));
    Path file = tempDir.resolve("results.txt");

    Recall.writeResultIds(file, results);
    List<List<SearchResult>> readBack = Recall.readResultIds(file);

    assertEquals(2, readBack.size());
    assertEquals(1.0, Recall.recallAtK(new int[][] {{7, 3}, {1, 9}}, readBack, 2), 1e-12);
  }

  @Test
  void nullInputsGiveZeroAndInvalidKThrows() {
    assertEquals(0.0, Recall.recallAtK(null, null, 10), 1e-12);
    org.junit.jupiter.api.Assertions.assertThrows(
        IllegalArgumentException.class, () -> Recall.recallAtK(new int[][] {{1}}, List.of(), 0));
  }
}
