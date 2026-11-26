package dev.hnswforge.eval;

import dev.hnswforge.exact.BruteForceIndex;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Computes exact nearest-neighbour ids and caches them in TEXMEX ivecs format.
 *
 * <p>Computation uses a {@link BruteForceIndex} built from the base vectors and
 * queries it in a parallel stream. Results are written as one ivecs record per
 * query, with ids in ascending-distance order.
 *
 * <p>All methods are stateless and thread-safe.
 */
public final class GroundTruth {

  private GroundTruth() {
  }

  /**
   * Computes exact ids for each query.
   *
   * <p>Thread-safe and stateless.
   *
   * @param base non-empty array of equally sized, non-NaN base vectors
   * @param queries non-empty array of equally sized, non-NaN query vectors
   * @param metric the distance metric; must not be {@code null}
   * @param k number of neighbours to return; must be positive
   * @return one int array of ids per query, sorted best first; an array may be shorter than {@code k} if the base is smaller
   * @throws IllegalArgumentException if an argument is {@code null}, {@code k} is not positive, arrays are empty,
   *                                  dimensions disagree, or a vector contains {@code NaN}
   */
  public static int[][] compute(float[][] base, float[][] queries, Metric metric, int k) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be > 0 but was " + k);
    }
    if (metric == null) {
      throw new IllegalArgumentException("metric must not be null");
    }
    int baseDimension = validateVectors(base, "base");
    int queryDimension = validateVectors(queries, "queries");
    if (baseDimension != queryDimension) {
      throw new IllegalArgumentException("Base dimension " + baseDimension
          + " does not match query dimension " + queryDimension);
    }
    try (BruteForceIndex index = new BruteForceIndex(baseDimension, metric)) {
      for (float[] vector : base) {
        index.add(vector);
      }
      return IntStream.range(0, queries.length).parallel()
          .mapToObj(i -> ids(index.search(queries[i], k)))
          .toArray(int[][]::new);
    }
  }

  /**
   * Writes ids in TEXMEX ivecs format, using 32-bit little-endian values.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to write; must not be {@code null}
   * @param ids one non-empty int array per query
   * @throws IOException if the file cannot be written
   * @throws IllegalArgumentException if an argument or a row is {@code null} or empty
   */
  public static void writeIvecs(Path path, int[][] ids) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    if (ids == null) {
      throw new IllegalArgumentException("ids must not be null");
    }
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
      ByteBuffer buffer = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < ids.length; i++) {
        int[] row = ids[i];
        if (row == null || row.length == 0) {
          throw new IllegalArgumentException("ids[" + i + "] must be non-null and non-empty");
        }
        long byteCount = 4L + (long) row.length * Integer.BYTES;
        if (byteCount > Integer.MAX_VALUE) {
          throw new IllegalArgumentException("ids[" + i + "] is too large to write");
        }
        if (byteCount > buffer.capacity()) {
          flush(channel, buffer);
          ByteBuffer large = ByteBuffer.allocate((int) byteCount).order(ByteOrder.LITTLE_ENDIAN);
          writeRow(large, row);
          flush(channel, large);
        } else {
          if (buffer.remaining() < byteCount) {
            flush(channel, buffer);
          }
          writeRow(buffer, row);
        }
      }
      flush(channel, buffer);
    }
  }

  /**
   * Reads ids previously written in TEXMEX ivecs format.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to read; must not be {@code null}
   * @return one int array per query
   * @throws IOException if the file is unreadable, truncated, or malformed
   * @throws IllegalArgumentException if {@code path} is {@code null}
   */
  public static int[][] readIvecs(Path path) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    return IvecsReader.read(path);
  }

  /**
   * Validates a matrix of float vectors and returns its dimension.
   *
   * @param vectors the matrix to validate
   * @param name the name used in error messages
   * @return the common positive dimension
   * @throws IllegalArgumentException if the matrix is empty, ragged, or contains {@code NaN}
   */
  private static int validateVectors(float[][] vectors, String name) {
    if (vectors == null || vectors.length == 0) {
      throw new IllegalArgumentException(name + " must be non-null and non-empty");
    }
    int dimension = -1;
    for (int i = 0; i < vectors.length; i++) {
      float[] vector = vectors[i];
      if (vector == null || vector.length == 0) {
        throw new IllegalArgumentException(name + "[" + i + "] must be non-null and non-empty");
      }
      for (float value : vector) {
        if (Float.isNaN(value)) {
          throw new IllegalArgumentException(name + "[" + i + "] contains NaN");
        }
      }
      if (dimension < 0) {
        dimension = vector.length;
      } else if (vector.length != dimension) {
        throw new IllegalArgumentException(name + "[" + i + "] has dimension " + vector.length
            + " but expected " + dimension);
      }
    }
    return dimension;
  }

  /**
   * Extracts ids from search results.
   *
   * @param results the search results
   * @return the ids in the same order
   */
  private static int[] ids(List<SearchResult> results) {
    int[] ids = new int[results.size()];
    for (int i = 0; i < ids.length; i++) {
      ids[i] = results.get(i).id();
    }
    return ids;
  }

  /**
   * Writes one ivecs record.
   *
   * @param buffer the destination buffer
   * @param row the ids to write
   */
  private static void writeRow(ByteBuffer buffer, int[] row) {
    buffer.putInt(row.length);
    for (int id : row) {
      buffer.putInt(id);
    }
  }

  /**
   * Flushes a buffer to a channel and clears it for reuse.
   *
   * @param channel the destination channel
   * @param buffer the buffer to flush
   * @throws IOException if the channel cannot be written
   */
  private static void flush(FileChannel channel, ByteBuffer buffer) throws IOException {
    buffer.flip();
    while (buffer.hasRemaining()) {
      channel.write(buffer);
    }
    buffer.clear();
  }
}
