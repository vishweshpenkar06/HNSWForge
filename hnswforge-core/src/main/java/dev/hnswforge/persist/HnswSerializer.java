package dev.hnswforge.persist;

import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.zip.CRC32;

/**
 * Binary reader and writer for version 1 and version 2 {@link IndexSnapshot} files.
 *
 * <p>The format is little-endian with a fixed 128-byte header followed by flat
 * float32 vectors, int32 levels, int32 layer-0 adjacency, and int32 upper-layer
 * adjacency. Version 2 adds a fifth section of int32 tombstoned ids after the
 * upper-layer adjacency, optionally preceded by zero padding to keep its offset
 * 64-byte aligned. A CRC32 over every byte after the header protects all sections,
 * including the padding and the deleted-id section. Version 1 files have zeros in
 * the header slots used by version 2 for the deleted offset and length, and carry
 * no deleted ids. Writes go to a sibling {@code .tmp} file and are then moved into
 * place, optionally using an atomic move.
 *
 * <p>Thread-safety: stateless; all methods are safe for concurrent use on
 * different files, and concurrent reads of the same file are safe.
 */
public final class HnswSerializer {

    static final int HEADER_SIZE = 128;

    private static final int VERSION_1 = 1;
    private static final int VERSION_2 = 2;
    static final int OFF_DELETED = 104;
    static final int OFF_DELETED_LEN = 112;
    private static final int VERSION_2_RESERVED_START = 116;
    private static final byte[] MAGIC_BYTES = {
            'H', 'N', 'S', 'W', 'F', 'R', 'G', 0x00
    };
    private static final int OFF_VERSION = 8;
    private static final int OFF_ENDIANNESS = 12;
    private static final int OFF_DIMENSION = 16;
    private static final int OFF_METRIC = 20;
    private static final int OFF_M = 24;
    private static final int OFF_EF_CONSTRUCTION = 28;
    private static final int OFF_EF_SEARCH = 32;
    private static final int OFF_MAX_ELEMENTS = 36;
    private static final int OFF_SIZE = 40;
    private static final int OFF_ENTRY_POINT = 44;
    private static final int OFF_MAX_LEVEL = 48;
    private static final int OFF_SEED = 52;
    private static final int OFF_VECTORS = 60;
    private static final int OFF_LEVELS = 68;
    private static final int OFF_LAYER0 = 76;
    private static final int OFF_UPPER = 84;
    private static final int OFF_UPPER_LEN = 92;
    private static final int OFF_CRC = 100;
    private static final int RESERVED_START = 104;

    private HnswSerializer() {
    }

    /**
     * Writes a snapshot to {@code path}, replacing any existing file.
     *
     * <p>The snapshot is validated before any bytes are written. The file is first
     * written to a sibling {@code .tmp} file and then moved into place with an
     * atomic move where supported. On failure the temporary file is deleted and no
     * partial output is left at {@code path}. The result is a version 2 file that
     * includes the snapshot's tombstoned ids.
     *
     * <p>Thread-safety: safe for concurrent calls on different paths; concurrent
     * writes to the same path must be serialized by the caller.
     *
     * @param snapshot snapshot to serialize, must be non-null and internally valid
     * @param path     destination path, must name a file
     * @throws IOException              if writing or moving the file fails
     * @throws IllegalArgumentException if the snapshot or path is invalid
     * @throws NullPointerException     if {@code snapshot} or {@code path} is null
     */
    public static void save(IndexSnapshot snapshot, Path path) throws IOException {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        Objects.requireNonNull(path, "path must not be null");
        Path fileName = path.getFileName();
        if (fileName == null) {
            throw new IllegalArgumentException("path must name a file: " + path);
        }
        HnswParams params = Objects.requireNonNull(snapshot.params(), "snapshot params must not be null");
        int size = snapshot.size();
        int dimension = params.dimension();
        int maxElements = params.maxElements();
        int layer0Stride = 1 + params.maxM0();
        int upperStride = 1 + params.m();

        validateSnapshotShape(snapshot, params, layer0Stride, upperStride);
        long vectorsBytes;
        long levelsBytes;
        long layer0Bytes;
        long upperLen;
        long upperBytes;
        try {
            vectorsBytes = Math.multiplyExact((long) size * dimension, (long) Float.BYTES);
            levelsBytes = Math.multiplyExact((long) size, (long) Integer.BYTES);
            layer0Bytes = Math.multiplyExact((long) size * layer0Stride, (long) Integer.BYTES);
            upperLen = expectedUpperLength(snapshot.levels(), upperStride);
            upperBytes = Math.multiplyExact(upperLen, (long) Integer.BYTES);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("snapshot is too large to serialize", e);
        }
        long vectorsOff = HEADER_SIZE;
        long levelsOff = vectorsOff + vectorsBytes;
        long layer0Off = levelsOff + levelsBytes;
        long upperOff = layer0Off + layer0Bytes;
        long deletedLen = snapshot.deleted().length;
        long deletedOff;
        long deletedBytes;
        long total;
        try {
            deletedOff = align64(Math.addExact(upperOff, upperBytes));
            deletedBytes = Math.multiplyExact(deletedLen, (long) Integer.BYTES);
            total = Math.addExact(deletedOff, deletedBytes);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("snapshot is too large to serialize", e);
        }
        if (total > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("snapshot is too large to serialize: " + total + " bytes");
        }

        byte[] bytes = new byte[(int) total];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(0, MAGIC_BYTES);
        buffer.putInt(OFF_VERSION, VERSION_2);
        buffer.putInt(OFF_ENDIANNESS, 1);
        buffer.putInt(OFF_DIMENSION, dimension);
        buffer.putInt(OFF_METRIC, snapshot.params().metric().ordinal());
        buffer.putInt(OFF_M, params.m());
        buffer.putInt(OFF_EF_CONSTRUCTION, params.efConstruction());
        buffer.putInt(OFF_EF_SEARCH, params.efSearch());
        buffer.putInt(OFF_MAX_ELEMENTS, maxElements);
        buffer.putInt(OFF_SIZE, size);
        buffer.putInt(OFF_ENTRY_POINT, snapshot.entryPoint());
        buffer.putInt(OFF_MAX_LEVEL, snapshot.maxLevel());
        buffer.putLong(OFF_SEED, params.seed());
        buffer.putLong(OFF_VECTORS, vectorsOff);
        buffer.putLong(OFF_LEVELS, levelsOff);
        buffer.putLong(OFF_LAYER0, layer0Off);
        buffer.putLong(OFF_UPPER, upperOff);
        buffer.putLong(OFF_UPPER_LEN, upperLen);
        buffer.putLong(OFF_DELETED, deletedOff);
        buffer.putInt(OFF_DELETED_LEN, (int) deletedLen);

        buffer.position((int) vectorsOff);
        buffer.asFloatBuffer().put(snapshot.vectors());
        buffer.position((int) levelsOff);
        buffer.asIntBuffer().put(snapshot.levels());
        buffer.position((int) layer0Off);
        buffer.asIntBuffer().put(snapshot.layer0());
        buffer.position((int) upperOff);
        IntBuffer upperOutput = buffer.asIntBuffer();
        int written = 0;
        for (int id = 0; id < size; id++) {
            int level = snapshot.levels()[id];
            if (level == 0) {
                continue;
            }
            int[] slice = snapshot.upper()[id];
            upperOutput.put(slice);
            written += slice.length;
        }
        if (written != upperLen) {
            throw new IllegalArgumentException("snapshot upper adjacency length mismatch");
        }
        buffer.position((int) deletedOff);
        buffer.asIntBuffer().put(snapshot.deleted());

        CRC32 crc = new CRC32();
        crc.update(bytes, HEADER_SIZE, bytes.length - HEADER_SIZE);
        buffer.putInt(OFF_CRC, (int) crc.getValue());

        Path temp = path.resolveSibling(fileName.toString() + ".tmp");
        try {
            try (FileChannel channel = FileChannel.open(temp, StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                ByteBuffer output = ByteBuffer.wrap(bytes);
                while (output.hasRemaining()) {
                    channel.write(output);
                }
                channel.force(true);
            }
            try {
                Files.move(temp, path, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException | RuntimeException e) {
            try {
                Files.deleteIfExists(temp);
            } catch (IOException cleanup) {
                e.addSuppressed(cleanup);
            }
            throw e;
        }
    }

    /**
     * Reads and fully validates a version 1 or version 2 snapshot from {@code path}.
     *
     * <p>Every returned array is freshly allocated and independent of the file
     * contents. The header, CRC, section offsets, sizes, metric, levels, degrees,
     * neighbor ids and, for version 2, the tombstoned ids are all checked before a
     * snapshot is returned. Version 1 files return an empty deleted-id array.
     *
     * <p>Thread-safety: safe for concurrent reads of the same or different files.
     *
     * @param path source file, must be non-null
     * @return a freshly allocated snapshot
     * @throws IOException          if reading the file fails
     * @throws IndexFormatException if the file is malformed or inconsistent
     * @throws NullPointerException if {@code path} is null
     */
    public static IndexSnapshot load(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        byte[] bytes = Files.readAllBytes(path);
        Header header = parseHeader(bytes, bytes.length);
        verifyCrc(bytes, header);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int vectorCount = Math.toIntExact(header.vectorsBytes() / Float.BYTES);
        float[] vectors = new float[vectorCount];
        buffer.position((int) header.vectorsOff());
        buffer.asFloatBuffer().get(vectors);
        int[] levels = new int[header.size()];
        buffer.position((int) header.levelsOff());
        buffer.asIntBuffer().get(levels);
        validateLevels(levels, header.maxLevel(), header.upperStride(), header.upperLen());

        int[] layer0 = new int[Math.toIntExact(header.layer0Bytes() / Integer.BYTES)];
        buffer.position((int) header.layer0Off());
        IntBuffer layer0Buffer = buffer.asIntBuffer();
        layer0Buffer.get(layer0);
        for (int id = 0; id < header.size(); id++) {
            validateAdjacency(layer0, id * (1 + header.maxM0()),
                    header.maxM0(), header.size(), "layer 0", id);
        }

        int[][] upper = new int[header.maxElements()][];
        int[] sliceBuffer = new int[Math.toIntExact(header.upperLen())];
        IntBuffer upperBuffer = ByteBuffer.wrap(bytes, (int) header.upperOff(),
                (int) header.upperBytes()).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        upperBuffer.get(sliceBuffer);
        int cursor = 0;
        for (int id = 0; id < header.size(); id++) {
            int level = levels[id];
            if (level == 0) {
                continue;
            }
            int stride = header.upperStride();
            int[] slice = new int[Math.multiplyExact(level, stride)];
            System.arraycopy(sliceBuffer, cursor, slice, 0, slice.length);
            for (int layer = 1; layer <= level; layer++) {
                validateAdjacency(slice, (layer - 1) * stride, header.params().m(),
                        header.size(), "upper layer " + layer, id);
            }
            upper[id] = slice;
            cursor += slice.length;
        }
        int[] deleted = new int[header.deletedLen()];
        buffer.position((int) header.deletedOff());
        buffer.asIntBuffer().get(deleted);
        validateDeleted(deleted, header.size());
        return new IndexSnapshot(header.params(), header.size(), header.entryPoint(),
                header.maxLevel(), vectors, levels, layer0, upper, deleted);
    }

    static Header parseHeader(byte[] bytes, long fileSize) throws IndexFormatException {
        if (bytes.length < HEADER_SIZE || fileSize < HEADER_SIZE) {
            throw new IndexFormatException("truncated header: expected " + HEADER_SIZE
                    + " bytes, got " + fileSize);
        }
        for (int i = 0; i < MAGIC_BYTES.length; i++) {
            if (bytes[i] != MAGIC_BYTES[i]) {
                throw new IndexFormatException("bad magic at byte " + i);
            }
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int version = buffer.getInt(OFF_VERSION);
        if (version != VERSION_1 && version != VERSION_2) {
            throw new IndexFormatException("unsupported version: " + version
                    + ", expected " + VERSION_1 + " or " + VERSION_2);
        }
        int endianness = buffer.getInt(OFF_ENDIANNESS);
        if (endianness != 1) {
            throw new IndexFormatException("unsupported endianness: " + endianness);
        }
        int reservedStart = version == VERSION_1 ? RESERVED_START : VERSION_2_RESERVED_START;
        for (int i = reservedStart; i < HEADER_SIZE; i++) {
            if (bytes[i] != 0) {
                throw new IndexFormatException("nonzero reserved byte at offset " + i);
            }
        }
        int dimension = buffer.getInt(OFF_DIMENSION);
        int metricOrdinal = buffer.getInt(OFF_METRIC);
        if (metricOrdinal < 0 || metricOrdinal >= Metric.values().length) {
            throw new IndexFormatException("metric ordinal out of range: " + metricOrdinal);
        }
        Metric metric = Metric.values()[metricOrdinal];
        int maxElements = buffer.getInt(OFF_MAX_ELEMENTS);
        int size = buffer.getInt(OFF_SIZE);
        int entryPoint = buffer.getInt(OFF_ENTRY_POINT);
        int maxLevel = buffer.getInt(OFF_MAX_LEVEL);
        HnswParams params;
        try {
            params = new HnswParams(dimension, metric, buffer.getInt(OFF_M),
                    buffer.getInt(OFF_EF_CONSTRUCTION), buffer.getInt(OFF_EF_SEARCH),
                    maxElements, buffer.getLong(OFF_SEED));
        } catch (IllegalArgumentException e) {
            throw new IndexFormatException("invalid parameters in header: " + e.getMessage(), e);
        }
        if (size < 0 || size > maxElements) {
            throw new IndexFormatException("size out of range: " + size + ", capacity " + maxElements);
        }
        if (size == 0) {
            if (entryPoint != -1 || maxLevel != -1) {
                throw new IndexFormatException("empty index must use entryPoint=-1 and maxLevel=-1");
            }
        } else if (entryPoint < 0 || entryPoint >= size) {
            throw new IndexFormatException("entry point out of range: " + entryPoint);
        } else if (maxLevel < 0) {
            throw new IndexFormatException("maxLevel must be >= 0 for a nonempty index: " + maxLevel);
        }

        long vectorsOff = buffer.getLong(OFF_VECTORS);
        long levelsOff = buffer.getLong(OFF_LEVELS);
        long layer0Off = buffer.getLong(OFF_LAYER0);
        long upperOff = buffer.getLong(OFF_UPPER);
        long upperLen = buffer.getLong(OFF_UPPER_LEN);
        int crc = buffer.getInt(OFF_CRC);
        if (vectorsOff < HEADER_SIZE || (vectorsOff & 63L) != 0) {
            throw new IndexFormatException("vectors offset must be >= 128 and 64-byte aligned: " + vectorsOff);
        }
        long vectorsBytes;
        long levelsBytes;
        long layer0Bytes;
        long maxUpperLen;
        long upperEnd;
        long deletedOff = 0;
        int deletedLen = 0;
        long expectedLength;
        try {
            vectorsBytes = Math.multiplyExact((long) size * dimension, (long) Float.BYTES);
            levelsBytes = Math.multiplyExact((long) size, (long) Integer.BYTES);
            layer0Bytes = Math.multiplyExact((long) size * (1 + params.maxM0()), (long) Integer.BYTES);
            maxUpperLen = Math.multiplyExact(Math.multiplyExact((long) size, (long) maxLevel),
                    (long) (1 + params.m()));
            if (upperLen < 0 || upperLen > maxUpperLen) {
                throw new IndexFormatException("upper length out of range: " + upperLen);
            }
            if (levelsOff != Math.addExact(vectorsOff, vectorsBytes)) {
                throw new IndexFormatException("levels offset mismatch: " + levelsOff
                        + ", expected " + (vectorsOff + vectorsBytes));
            }
            if (layer0Off != Math.addExact(levelsOff, levelsBytes)) {
                throw new IndexFormatException("layer0 offset mismatch: " + layer0Off
                        + ", expected " + (levelsOff + levelsBytes));
            }
            if (upperOff != Math.addExact(layer0Off, layer0Bytes)) {
                throw new IndexFormatException("upper offset mismatch: " + upperOff
                        + ", expected " + (layer0Off + layer0Bytes));
            }
            upperEnd = Math.addExact(upperOff, Math.multiplyExact(upperLen, (long) Integer.BYTES));
            if (version == VERSION_2) {
                deletedOff = buffer.getLong(OFF_DELETED);
                deletedLen = buffer.getInt(OFF_DELETED_LEN);
                if (deletedLen < 0 || deletedLen > size) {
                    throw new IndexFormatException("version 2 deleted length out of range: "
                            + deletedLen + ", size " + size);
                }
                if (deletedOff < upperEnd || (deletedOff & 3L) != 0) {
                    throw new IndexFormatException("version 2 deleted offset out of range: "
                            + deletedOff + ", expected >= " + upperEnd);
                }
            }
            expectedLength = version == VERSION_2
                    ? Math.addExact(deletedOff, Math.multiplyExact((long) deletedLen, (long) Integer.BYTES))
                    : upperEnd;
        } catch (ArithmeticException e) {
            throw new IndexFormatException("section sizes overflow", e);
        }
        if (expectedLength != fileSize) {
            throw new IndexFormatException("file length mismatch: expected " + expectedLength
                    + " bytes, got " + fileSize);
        }
        return new Header(params, size, entryPoint, maxLevel, vectorsOff, levelsOff,
                layer0Off, upperOff, upperLen, crc, version, deletedOff, deletedLen);
    }

    static void verifyCrc(byte[] bytes, Header header) throws IndexFormatException {
        CRC32 crc = new CRC32();
        crc.update(bytes, HEADER_SIZE, bytes.length - HEADER_SIZE);
        if ((int) crc.getValue() != header.crc()) {
            throw new IndexFormatException("CRC mismatch: header " + Integer.toHexString(header.crc())
                    + ", computed " + Integer.toHexString((int) crc.getValue()));
        }
    }

    static void validateLevels(int[] levels, int maxLevel, int upperStride, long declaredUpperLen)
            throws IndexFormatException {
        long total = 0;
        try {
            for (int id = 0; id < levels.length; id++) {
                int level = levels[id];
                if (level < 0 || level > maxLevel) {
                    throw new IndexFormatException("level out of range at id " + id + ": " + level);
                }
                if (level > 0) {
                    total = Math.addExact(total, Math.multiplyExact((long) level, (long) upperStride));
                }
            }
        } catch (ArithmeticException e) {
            throw new IndexFormatException("upper length overflow", e);
        }
        if (total != declaredUpperLen) {
            throw new IndexFormatException("upper length mismatch: header says " + declaredUpperLen
                    + ", levels imply " + total);
        }
    }

    static void validateDeleted(int[] deleted, int size) throws IndexFormatException {
        int previous = -1;
        for (int id : deleted) {
            if (id < 0 || id >= size) {
                throw new IndexFormatException("deleted id out of range: " + id + ", size " + size);
            }
            if (id <= previous) {
                throw new IndexFormatException("deleted ids must be sorted and distinct: "
                        + id + " after " + previous);
            }
            previous = id;
        }
    }

    private static long align64(long value) throws ArithmeticException {
        return Math.addExact(value, 63L) & ~63L;
    }

    private static long expectedUpperLength(int[] levels, int upperStride) {
        long total = 0;
        for (int level : levels) {
            total += (long) level * upperStride;
        }
        return total;
    }

    private static void validateSnapshotShape(IndexSnapshot snapshot, HnswParams params,
                                              int layer0Stride, int upperStride) {
        int size = snapshot.size();
        int maxElements = params.maxElements();
        if (size < 0 || size > maxElements) {
            throw new IllegalArgumentException("size out of range: " + size + ", capacity " + maxElements);
        }
        if (size == 0) {
            if (snapshot.entryPoint() != -1 || snapshot.maxLevel() != -1) {
                throw new IllegalArgumentException("empty snapshot must use entryPoint=-1 and maxLevel=-1");
            }
        } else if (snapshot.entryPoint() < 0 || snapshot.entryPoint() >= size) {
            throw new IllegalArgumentException("entry point out of range: " + snapshot.entryPoint());
        } else if (snapshot.maxLevel() < 0) {
            throw new IllegalArgumentException("maxLevel must be >= 0 for a nonempty snapshot");
        }
        float[] vectors = Objects.requireNonNull(snapshot.vectors(), "vectors must not be null");
        int[] levels = Objects.requireNonNull(snapshot.levels(), "levels must not be null");
        int[] layer0 = Objects.requireNonNull(snapshot.layer0(), "layer0 must not be null");
        int[][] upper = Objects.requireNonNull(snapshot.upper(), "upper must not be null");
        if (vectors.length != (long) size * params.dimension()) {
            throw new IllegalArgumentException("vectors length mismatch: " + vectors.length);
        }
        if (levels.length != size) {
            throw new IllegalArgumentException("levels length mismatch: " + levels.length);
        }
        if (layer0.length != (long) size * layer0Stride) {
            throw new IllegalArgumentException("layer0 length mismatch: " + layer0.length);
        }
        if (upper.length != maxElements) {
            throw new IllegalArgumentException("upper length mismatch: " + upper.length);
        }
        int[] deleted = Objects.requireNonNull(snapshot.deleted(), "deleted must not be null");
        int previous = -1;
        for (int id : deleted) {
            if (id < 0 || id >= size) {
                throw new IllegalArgumentException("deleted id out of range: " + id
                        + ", size " + size);
            }
            if (id <= previous) {
                throw new IllegalArgumentException("deleted ids must be sorted and distinct: "
                        + id + " after " + previous);
            }
            previous = id;
        }
        for (float value : vectors) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException("vector values must not be NaN");
            }
        }
        for (int id = 0; id < size; id++) {
            int level = levels[id];
            if (level < 0 || level > snapshot.maxLevel()) {
                throw new IllegalArgumentException("level out of range at id " + id + ": " + level);
            }
            int degree = layer0[id * layer0Stride];
            if (degree < 0 || degree > params.maxM0()) {
                throw new IllegalArgumentException("layer 0 degree out of range at id " + id + ": " + degree);
            }
            for (int index = 0; index < degree; index++) {
                checkNeighbor(layer0[id * layer0Stride + 1 + index], size, id, "layer 0");
            }
            if (level == 0) {
                continue;
            }
            int[] slice = Objects.requireNonNull(upper[id], "upper[" + id + "] must not be null");
            if (slice.length != level * upperStride) {
                throw new IllegalArgumentException("upper[" + id + "] length mismatch: " + slice.length);
            }
            for (int layer = 1; layer <= level; layer++) {
                int offset = (layer - 1) * upperStride;
                int upperDegree = slice[offset];
                if (upperDegree < 0 || upperDegree > params.m()) {
                    throw new IllegalArgumentException("upper degree out of range at id " + id
                            + ", layer " + layer + ": " + upperDegree);
                }
                for (int index = 0; index < upperDegree; index++) {
                    checkNeighbor(slice[offset + 1 + index], size, id, "upper layer " + layer);
                }
            }
        }
    }

    private static void checkNeighbor(int neighbor, int size, int id, String layer) {
        if (neighbor < 0 || neighbor >= size) {
            throw new IllegalArgumentException(layer + " neighbor out of range at id " + id + ": " + neighbor);
        }
    }

    private static void validateAdjacency(int[] adjacency, int offset, int maxDegree,
                                          int size, String layer, int id)
            throws IndexFormatException {
        int degree = adjacency[offset];
        if (degree < 0 || degree > maxDegree) {
            throw new IndexFormatException(layer + " degree out of range at id " + id + ": " + degree);
        }
        for (int index = 0; index < degree; index++) {
            int neighbor = adjacency[offset + 1 + index];
            if (neighbor < 0 || neighbor >= size) {
                throw new IndexFormatException(layer + " neighbor out of range at id "
                        + id + ": " + neighbor);
            }
        }
    }

    record Header(HnswParams params, int size, int entryPoint, int maxLevel,
                  long vectorsOff, long levelsOff, long layer0Off, long upperOff,
                  long upperLen, int crc, int version, long deletedOff, int deletedLen) {

        int dimension() {
            return params.dimension();
        }

        int maxElements() {
            return params.maxElements();
        }

        int maxM0() {
            return params.maxM0();
        }

        int upperStride() {
            return 1 + params.m();
        }

        long vectorsBytes() {
            return (long) size * dimension() * Float.BYTES;
        }

        long levelsBytes() {
            return (long) size * Integer.BYTES;
        }

        long layer0Bytes() {
            return (long) size * (1 + maxM0()) * Integer.BYTES;
        }

        long upperBytes() {
            return upperLen * Integer.BYTES;
        }
    }
}
