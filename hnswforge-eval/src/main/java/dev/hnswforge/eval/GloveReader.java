package dev.hnswforge.eval;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads GloVe word-vector text files.
 *
 * <p>Each non-blank line has the form {@code word f1 f2 ... fn} with fields
 * separated by whitespace. Every line must have the same number of floating
 * point fields; the first non-blank line fixes the dimension. A malformed
 * number or inconsistent field count raises an {@link IOException} that
 * includes the one-based line number.
 *
 * <p>Instances of {@link GloveData} are immutable and thread-safe. All reader
 * methods are stateless and thread-safe.
 */
public final class GloveReader {

  /**
   * Words and their vectors, in file order.
   *
   * @param words the words, one per vector
   * @param vectors the vectors, parallel to {@code words}
   */
  public record GloveData(List<String> words, float[][] vectors) {

    /**
     * Creates an immutable snapshot of the parsed data.
     *
     * <p>Thread-safe; the snapshot is immutable.
     *
     * @param words the words, one per vector
     * @param vectors the vectors, parallel to {@code words}
     * @throws IllegalArgumentException if either argument is {@code null}
     */
    public GloveData {
      if (words == null || vectors == null) {
        throw new IllegalArgumentException("words and vectors must not be null");
      }
      words = List.copyOf(words);
      vectors = vectors.clone();
      for (float[] vector : vectors) {
        if (vector == null) {
          throw new IllegalArgumentException("vectors must not contain null entries");
        }
      }
    }
  }

  private GloveReader() {
  }

  /**
   * Reads all word vectors from a file.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to read; must not be {@code null}
   * @return the vectors in file order
   * @throws IOException if the file is unreadable or malformed
   * @throws IllegalArgumentException if {@code path} is {@code null}
   */
  public static float[][] read(Path path) throws IOException {
    return readWithWords(path, 0).vectors();
  }

  /**
   * Reads at most {@code limit} word vectors from a file.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to read; must not be {@code null}
   * @param limit maximum number of vectors to read; values less than or equal to zero mean unlimited
   * @return the vectors in file order, possibly empty
   * @throws IOException if the file is unreadable or malformed before the limit is reached
   * @throws IllegalArgumentException if {@code path} is {@code null}
   */
  public static float[][] read(Path path, int limit) throws IOException {
    return readData(path, limit).vectors();
  }

  /**
   * Reads word vectors together with their words.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to read; must not be {@code null}
   * @param limit maximum number of vectors to read; values less than or equal to zero mean unlimited
   * @return an immutable snapshot of the words and vectors in file order
   * @throws IOException if the file is unreadable or malformed before the limit is reached
   * @throws IllegalArgumentException if {@code path} is {@code null}
   */
  public static GloveData readWithWords(Path path, int limit) throws IOException {
    return readData(path, limit);
  }

  /**
   * Parses a file into words and vectors.
   *
   * @param path the file to read
   * @param limit maximum number of vectors; non-positive means unlimited
   * @return the parsed data
   * @throws IOException if the file is unreadable or malformed
   */
  private static GloveData readData(Path path, int limit) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    List<String> words = new ArrayList<>();
    List<float[]> vectors = new ArrayList<>();
    int dimension = -1;
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      int lineNumber = 0;
      while ((line = reader.readLine()) != null) {
        lineNumber++;
        String trimmed = line.trim();
        if (trimmed.isEmpty()) {
          continue;
        }
        if (limit > 0 && vectors.size() >= limit) {
          break;
        }
        String[] tokens = trimmed.split("\\s+");
        if (tokens.length < 2) {
          throw new IOException("Line " + lineNumber + " of " + path
              + ": expected a word followed by at least one float but found " + tokens.length + " field(s)");
        }
        int fields = tokens.length - 1;
        if (dimension < 0) {
          dimension = fields;
        } else if (fields != dimension) {
          throw new IOException("Line " + lineNumber + " of " + path
              + ": expected " + dimension + " floats but found " + fields);
        }
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
          try {
            vector[i] = Float.parseFloat(tokens[i + 1]);
          } catch (NumberFormatException e) {
            throw new IOException("Line " + lineNumber + " of " + path
                + ": malformed float '" + tokens[i + 1] + "'", e);
          }
        }
        words.add(tokens[0]);
        vectors.add(vector);
      }
    }
    return new GloveData(words, vectors.toArray(float[][]::new));
  }
}
