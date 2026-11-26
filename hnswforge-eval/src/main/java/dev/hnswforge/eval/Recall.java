package dev.hnswforge.eval;

import dev.hnswforge.index.SearchResult;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Computes recall of approximate search results against exact ground truth.
 *
 * <p>Recall for one query is the number of ground-truth ids present among the
 * first {@code k} retrieved ids divided by {@code k}. The denominator is always
 * {@code k}, even when the ground truth or the result list contains fewer than
 * {@code k} ids, which is the conventional ANN definition and penalizes
 * incomplete result lists. The returned value is the arithmetic mean across
 * queries and is always in {@code [0, 1]}. Duplicate retrieved ids are counted
 * once. A missing query row, or an empty ground-truth or result row, contributes
 * zero to the mean; {@code null} inputs yield {@code 0.0}.
 *
 * <p>All methods are stateless and thread-safe.
 */
public final class Recall {

  private Recall() {
  }

  /**
   * Computes mean recall at {@code k}.
   *
   * <p>Thread-safe and stateless.
   *
   * @param groundTruth one int array of exact ids per query, best first
   * @param results one list of search results per query, best first
   * @param k the cutoff and denominator; must be positive
   * @return mean recall in {@code [0, 1]}, or {@code 0.0} if either input is {@code null} or empty
   * @throws IllegalArgumentException if {@code k} is not positive
   */
  public static double recallAtK(int[][] groundTruth, List<List<SearchResult>> results, int k) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be > 0 but was " + k);
    }
    if (groundTruth == null || results == null) {
      return 0.0;
    }
    int queryCount = Math.min(groundTruth.length, results.size());
    if (queryCount == 0) {
      return 0.0;
    }
    double total = 0.0;
    for (int i = 0; i < queryCount; i++) {
      total += recallForQuery(groundTruth[i], results.get(i), k);
    }
    return total / queryCount;
  }

  /**
   * Reads result ids from a text file with one query per line.
   *
   * <p>Blank lines are skipped. Each remaining line is a whitespace-separated
   * list of integer ids; distances in the produced results are set to {@code 0f}.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to read; must not be {@code null}
   * @return one list of results per non-blank line, in file order
   * @throws IOException if the file is unreadable or contains a malformed id
   * @throws IllegalArgumentException if {@code path} is {@code null}
   */
  public static List<List<SearchResult>> readResultIds(Path path) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    List<List<SearchResult>> all = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      int lineNumber = 0;
      while ((line = reader.readLine()) != null) {
        lineNumber++;
        String trimmed = line.trim();
        if (trimmed.isEmpty()) {
          continue;
        }
        List<SearchResult> query = new ArrayList<>();
        for (String token : trimmed.split("\\s+")) {
          try {
            query.add(new SearchResult(Integer.parseInt(token), 0f));
          } catch (NumberFormatException e) {
            throw new IOException("Malformed result id '" + token + "' on line " + lineNumber + " of " + path, e);
          }
        }
        all.add(List.copyOf(query));
      }
    }
    return all;
  }

  /**
   * Writes result ids to a text file with one query per line.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to write; must not be {@code null}
   * @param results one list of results per query, in order
   * @throws IOException if the file cannot be written
   * @throws IllegalArgumentException if an argument or a query/result entry is {@code null}
   */
  public static void writeResultIds(Path path, List<List<SearchResult>> results) throws IOException {
    if (path == null || results == null) {
      throw new IllegalArgumentException("path and results must not be null");
    }
    try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
      for (List<SearchResult> query : results) {
        if (query == null) {
          throw new IllegalArgumentException("results must not contain null queries");
        }
        StringBuilder line = new StringBuilder();
        for (SearchResult result : query) {
          if (result == null) {
            throw new IllegalArgumentException("results must not contain null entries");
          }
          if (!line.isEmpty()) {
            line.append(' ');
          }
          line.append(result.id());
        }
        writer.write(line.toString());
        writer.newLine();
      }
    }
  }

  /**
   * Computes recall for a single query.
   *
   * @param truth exact ids for the query
   * @param results retrieved results for the query
   * @param k the cutoff and denominator
   * @return recall in {@code [0, 1]}
   */
  private static double recallForQuery(int[] truth, List<SearchResult> results, int k) {
    if (truth == null || results == null || truth.length == 0 || results.isEmpty()) {
      return 0.0;
    }
    Set<Integer> retrieved = new HashSet<>();
    int resultLimit = Math.min(k, results.size());
    for (int i = 0; i < resultLimit; i++) {
      SearchResult result = results.get(i);
      if (result != null) {
        retrieved.add(result.id());
      }
    }
    int hits = 0;
    int truthLimit = Math.min(k, truth.length);
    for (int i = 0; i < truthLimit; i++) {
      if (retrieved.contains(truth[i])) {
        hits++;
      }
    }
    return (double) hits / k;
  }
}
