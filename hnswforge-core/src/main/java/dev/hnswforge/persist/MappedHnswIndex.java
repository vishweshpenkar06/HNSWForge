package dev.hnswforge.persist;

import dev.hnswforge.distance.DistanceFunction;
import dev.hnswforge.distance.ScalarDistance;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import dev.hnswforge.index.VectorIndex;
import dev.hnswforge.util.BoundedMaxHeap;
import dev.hnswforge.util.MinHeap;
import dev.hnswforge.util.SearchScratch;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.CRC32;

/**
 * Read-only HNSW index backed by a memory-mapped version 1 snapshot.
 *
 * <p>Vectors and layer-0 adjacency are read directly from the mapped region at
 * little-endian byte offsets derived from the file header; levels and upper-layer
 * adjacency are copied into immutable heap arrays on open. The mapping is owned by
 * a shared {@link Arena} held by this instance and is released by {@link #close()}.
 * Neither {@link #add(float[])} nor {@link #save(Path)} is supported. Closing the
 * index invalidates the mapping; any subsequent operation throws
 * {@link IllegalStateException}.
 *
 * <p>Thread-safety: safe for concurrent searches. Mapped memory and copied graph
 * arrays are read-only, and each thread uses its own scratch buffers via
 * {@link ThreadLocal}. {@link #close()} must not race with in-flight searches.
 */
public final class MappedHnswIndex implements VectorIndex {

    private static final ValueLayout.OfInt INT_LE =
            ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final Arena arena;
    private final MemorySegment segment;
    private final HnswParams params;
    private final int size;
    private final int entryPoint;
    private final int maxLevel;
    private final long vectorsOff;
    private final long layer0Off;
    private final int layer0Stride;
    private final int upperStride;
    private final int[] levels;
    private final int[][] upper;
    private final DistanceFunction distance;
    private final ThreadLocal<SearchScratch> scratch;
    private final AtomicBoolean closed = new AtomicBoolean();

    private MappedHnswIndex(HnswSerializer.Header header, MemorySegment segment,
                            Arena arena, int[] levels, int[][] upper) {
        this.arena = arena;
        this.segment = segment;
        this.params = header.params();
        this.size = header.size();
        this.entryPoint = header.entryPoint();
        this.maxLevel = header.maxLevel();
        this.vectorsOff = header.vectorsOff();
        this.layer0Off = header.layer0Off();
        this.layer0Stride = 1 + params.maxM0();
        this.upperStride = 1 + params.m();
        this.levels = levels;
        this.upper = upper;
        this.distance = ScalarDistance.forMetric(params.metric());
        this.scratch = ThreadLocal.withInitial(() -> new SearchScratch(
                params.dimension(), params.maxElements(),
                Math.max(params.efSearch(), 1), params.maxM0()));
    }

    /**
     * Opens a version 1 snapshot for read-only searching.
     *
     * <p>The header, CRC, section offsets and level structure are validated before
     * the mapping is returned. On any failure the arena is released. The caller
     * must call {@link #close()} when the index is no longer needed.
     *
     * @param path snapshot file, must be non-null
     * @return a mapped, read-only index
     * @throws IOException          if the file cannot be read or mapped
     * @throws IndexFormatException if the file is malformed or inconsistent
     * @throws NullPointerException if {@code path} is null
     */
    public static MappedHnswIndex open(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            byte[] headerBytes = new byte[HnswSerializer.HEADER_SIZE];
            ByteBuffer headerBuffer = ByteBuffer.wrap(headerBytes);
            while (headerBuffer.hasRemaining()) {
                if (channel.read(headerBuffer, headerBuffer.position()) < 0) {
                    break;
                }
            }
            HnswSerializer.Header header = HnswSerializer.parseHeader(headerBytes, fileSize);
            Arena arena = Arena.ofShared();
            try {
                MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY,
                        0, fileSize, arena);
                verifyCrc(segment, header, fileSize);
                int[] levels = new int[header.size()];
                long cursor = header.levelsOff();
                for (int id = 0; id < levels.length; id++) {
                    levels[id] = segment.get(INT_LE, cursor + (long) id * Integer.BYTES);
                }
                HnswSerializer.validateLevels(levels, header.maxLevel(),
                        header.upperStride(), header.upperLen());
                int[][] upper = new int[header.maxElements()][];
                cursor = header.upperOff();
                for (int id = 0; id < levels.length; id++) {
                    int level = levels[id];
                    if (level == 0) {
                        continue;
                    }
                    int length = Math.multiplyExact(level, header.upperStride());
                    int[] slice = new int[length];
                    for (int index = 0; index < length; index++) {
                        slice[index] = segment.get(INT_LE, cursor + (long) index * Integer.BYTES);
                    }
                    cursor += (long) length * Integer.BYTES;
                    upper[id] = slice;
                }
                return new MappedHnswIndex(header, segment, arena, levels, upper);
            } catch (IOException | RuntimeException | Error e) {
                arena.close();
                throw e;
            }
        }
    }

    /**
     * Always rejects insertion; a mapped index cannot be modified.
     *
     * <p>Thread-safety: safe to call from any thread; throws consistently.
     *
     * @param vector ignored
     * @return never returns normally
     * @throws UnsupportedOperationException always, because the index is read-only
     * @throws IllegalStateException         if the index has been closed
     */
    @Override
    public int add(float[] vector) {
        ensureOpen();
        throw new UnsupportedOperationException("mapped index is read-only");
    }

    /**
     * Searches with the configured {@code efSearch} effort.
     *
     * <p>Thread-safety: safe for concurrent searches with per-thread scratch buffers.
     *
     * @param query query vector of length {@link #dimension()}
     * @param k     number of results, must be positive
     * @return results sorted by ascending distance, ties by ascending id
     * @throws IllegalArgumentException if the query or {@code k} is invalid
     * @throws IllegalStateException    if the index has been closed
     */
    @Override
    public List<SearchResult> search(float[] query, int k) {
        return search(query, k, params.efSearch());
    }

    /**
     * Searches with an explicit effort of {@code max(efSearch, k)}.
     *
     * <p>Thread-safety: safe for concurrent searches with per-thread scratch buffers.
     *
     * @param query    query vector of length {@link #dimension()}
     * @param k        number of results, must be positive
     * @param efSearch search effort, must be positive
     * @return results sorted by ascending distance, ties by ascending id
     * @throws IllegalArgumentException if the query, {@code k}, or {@code efSearch} is invalid
     * @throws IllegalStateException    if the index has been closed
     */
    @Override
    public List<SearchResult> search(float[] query, int k, int efSearch) {
        ensureOpen();
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        if (efSearch < 1) {
            throw new IllegalArgumentException("efSearch must be >= 1, got " + efSearch);
        }
        SearchScratch local = scratch.get();
        float[] prepared = prepare(query, local.query);
        if (size == 0) {
            return List.of();
        }
        int entry = greedyDescent(prepared);
        local.entryBuffer[0] = entry;
        int found = searchLayer(prepared, local, Math.max(efSearch, k));
        List<SearchResult> results = new ArrayList<>(Math.min(k, found));
        for (int index = 0; index < found && results.size() < k; index++) {
            results.add(new SearchResult(local.foundIds[index], local.foundDistances[index]));
        }
        return List.copyOf(results);
    }

    /**
     * Returns the number of vectors in the snapshot.
     *
     * <p>Thread-safety: reads an immutable field; safe for concurrent use.
     *
     * @return the snapshot size
     * @throws IllegalStateException if the index has been closed
     */
    @Override
    public int size() {
        ensureOpen();
        return size;
    }

    /**
     * Returns the vector dimension from the snapshot header.
     *
     * <p>Thread-safety: reads an immutable field; safe for concurrent use.
     *
     * @return the vector dimension
     * @throws IllegalStateException if the index has been closed
     */
    @Override
    public int dimension() {
        ensureOpen();
        return params.dimension();
    }

    /**
     * Returns the distance metric from the snapshot header.
     *
     * <p>Thread-safety: reads an immutable field; safe for concurrent use.
     *
     * @return the distance metric
     * @throws IllegalStateException if the index has been closed
     */
    @Override
    public Metric metric() {
        ensureOpen();
        return params.metric();
    }

    /**
     * Always rejects saving; a mapped index is a read-only view of its file.
     *
     * <p>Thread-safety: safe to call from any thread; throws consistently.
     *
     * @param path ignored
     * @throws UnsupportedOperationException always, because the index is read-only
     * @throws IllegalStateException         if the index has been closed
     */
    @Override
    public void save(Path path) throws IOException {
        ensureOpen();
        throw new UnsupportedOperationException("mapped index is read-only");
    }

    /**
     * Releases the shared arena that owns the memory mapping.
     *
     * <p>Thread-safety: idempotent; must not race with in-flight searches. After
     * the first call every other operation throws {@link IllegalStateException}.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            arena.close();
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("mapped index is closed");
        }
    }

    private static void verifyCrc(MemorySegment segment, HnswSerializer.Header header,
                                  long fileSize) throws IndexFormatException {
        CRC32 crc = new CRC32();
        byte[] chunk = new byte[64 * 1024];
        long position = HnswSerializer.HEADER_SIZE;
        while (position < fileSize) {
            int length = (int) Math.min(chunk.length, fileSize - position);
            MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, position, chunk, 0, length);
            crc.update(chunk, 0, length);
            position += length;
        }
        if ((int) crc.getValue() != header.crc()) {
            throw new IndexFormatException("CRC mismatch: header " + Integer.toHexString(header.crc())
                    + ", computed " + Integer.toHexString((int) crc.getValue()));
        }
    }

    private float[] prepare(float[] query, float[] target) {
        Objects.requireNonNull(query, "query must not be null");
        if (query.length != params.dimension()) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + params.dimension() + ", got " + query.length);
        }
        for (float value : query) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException("query values must not be NaN");
            }
        }
        System.arraycopy(query, 0, target, 0, params.dimension());
        if (params.metric() == Metric.COSINE) {
            normalize(target);
        }
        return target;
    }

    private static void normalize(float[] vector) {
        double sum = 0.0;
        for (float value : vector) {
            sum += (double) value * value;
        }
        if (sum == 0.0) {
            throw new IllegalArgumentException("cosine distance is undefined for a zero vector");
        }
        double norm = Math.sqrt(sum);
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (vector[i] / norm);
        }
    }

    private float distanceTo(int id, float[] query) {
        return distance.distance(query, segment,
                vectorsOff + (long) id * params.dimension() * Float.BYTES, params.dimension());
    }

    private int layer0Degree(int id) {
        return segment.get(INT_LE,
                layer0Off + (long) id * layer0Stride * Integer.BYTES);
    }

    private int layer0Neighbor(int id, int index) {
        return segment.get(INT_LE,
                layer0Off + ((long) id * layer0Stride + 1 + index) * Integer.BYTES);
    }

    private int upperDegree(int id, int layer) {
        int[] slice = upper[id];
        return slice == null ? 0 : slice[(layer - 1) * upperStride];
    }

    private int upperNeighbor(int id, int layer, int index) {
        return upper[id][(layer - 1) * upperStride + 1 + index];
    }

    private int greedyDescent(float[] query) {
        int current = entryPoint;
        float currentDistance = distanceTo(current, query);
        for (int layer = maxLevel; layer >= 1; layer--) {
            if (layer > levels[current]) {
                continue;
            }
            boolean improved;
            do {
                improved = false;
                int node = current;
                int count = upperDegree(node, layer);
                for (int index = 0; index < count; index++) {
                    int neighbor = upperNeighbor(node, layer, index);
                    float distanceToNeighbor = distanceTo(neighbor, query);
                    if (distanceToNeighbor < currentDistance) {
                        current = neighbor;
                        currentDistance = distanceToNeighbor;
                        improved = true;
                    }
                }
            } while (improved && layer <= levels[current]);
        }
        return current;
    }

    private int searchLayer(float[] query, SearchScratch local, int ef) {
        local.visited.begin();
        local.candidates.clear();
        local.results.reset(ef);
        local.ensureFound(ef);
        int entry = local.entryBuffer[0];
        if (entry < 0 || entry >= size || !local.visited.mark(entry)) {
            return 0;
        }
        float entryDistance = distanceTo(entry, query);
        local.candidates.push(entry, entryDistance);
        local.results.push(entry, entryDistance);
        while (local.candidates.size() > 0) {
            int current = local.candidates.pop();
            float currentDistance = local.candidates.popDistance();
            if (local.results.size() == ef && currentDistance > local.results.peekDistance()) {
                break;
            }
            int count = layer0Degree(current);
            for (int index = 0; index < count; index++) {
                int neighbor = layer0Neighbor(current, index);
                if (neighbor < 0 || neighbor >= size || !local.visited.mark(neighbor)) {
                    continue;
                }
                float distanceToNeighbor = distanceTo(neighbor, query);
                if (local.results.size() < ef || distanceToNeighbor < local.results.peekDistance()) {
                    local.candidates.push(neighbor, distanceToNeighbor);
                    local.results.push(neighbor, distanceToNeighbor);
                }
            }
        }
        int found = local.results.size();
        for (int index = found - 1; index >= 0; index--) {
            local.foundIds[index] = local.results.pop();
            local.foundDistances[index] = local.results.popDistance();
        }
        return found;
    }
}
