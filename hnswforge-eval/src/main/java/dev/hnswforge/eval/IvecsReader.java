package dev.hnswforge.eval;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads TEXMEX {@code .ivecs} files.
 *
 * <p>Each record consists of a 32-bit little-endian dimension followed by that
 * many 32-bit little-endian signed integers. All records must declare the same
 * positive dimension. An invalid dimension, a dimension change, or a truncated
 * record causes an {@link IOException} naming the offending record index. An
 * empty file is valid and yields an empty array.
 *
 * <p>All methods are stateless and thread-safe.
 */
public final class IvecsReader {

  private IvecsReader() {
  }

  /**
   * Reads every vector in the file.
   *
   * <p>Thread-safe and stateless.
   *
   * @param path the file to read; must not be {@code null}
   * @return the vectors in file order, possibly empty for an empty file
   * @throws IOException if the file is unreadable, truncated, or malformed
   * @throws IllegalArgumentException if {@code path} is {@code null}
   */
  public static int[][] read(Path path) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      long fileSize = channel.size();
      ByteBuffer header = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      List<int[]> vectors = new ArrayList<>();
      long position = 0;
      int expectedDimension = -1;
      int index = 0;
      while (position < fileSize) {
        header.clear();
        readFully(channel, header, position, "dimension header of vector " + index + " in " + path);
        int dimension = header.getInt(0);
        if (dimension <= 0) {
          throw new IOException("Invalid dimension " + dimension + " for vector " + index + " in " + path);
        }
        if (expectedDimension < 0) {
          expectedDimension = dimension;
        } else if (dimension != expectedDimension) {
          throw new IOException("Dimension mismatch at vector " + index + " in " + path
              + ": expected " + expectedDimension + " but found " + dimension);
        }
        position += Integer.BYTES;
        long byteCount = (long) dimension * Integer.BYTES;
        long available = Math.max(0, fileSize - position);
        if (byteCount > Integer.MAX_VALUE) {
          throw new IOException("Vector " + index + " in " + path + " has an unsupported dimension " + dimension);
        }
        if (byteCount > available) {
          throw new IOException("Truncated vector " + index + " in " + path
              + ": expected " + byteCount + " bytes but only " + available + " remain");
        }
        ByteBuffer payload = ByteBuffer.allocate((int) byteCount).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, payload, position, "vector " + index + " in " + path);
        payload.flip();
        int[] vector = new int[dimension];
        for (int i = 0; i < dimension; i++) {
          vector[i] = payload.getInt();
        }
        vectors.add(vector);
        position += byteCount;
        index++;
      }
      return vectors.toArray(int[][]::new);
    }
  }

  /**
   * Fills a buffer from an absolute channel position, failing on early EOF.
   *
   * @param channel the open channel
   * @param buffer the buffer to fill
   * @param position the absolute position of the first byte
   * @param description text identifying the bytes in error messages
   * @throws IOException if the channel ends before the buffer is full
   */
  private static void readFully(FileChannel channel, ByteBuffer buffer, long position, String description)
      throws IOException {
    long offset = position;
    while (buffer.hasRemaining()) {
      int read = channel.read(buffer, offset);
      if (read < 0) {
        throw new IOException("Truncated " + description);
      }
      offset += read;
    }
  }
}
