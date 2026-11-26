package dev.hnswforge.repoprism;

import dev.hnswforge.index.HnswIndex;
import dev.hnswforge.index.HnswParams;
import dev.hnswforge.index.Metric;
import dev.hnswforge.index.SearchResult;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * {@link VectorSearchBackend} backed by a cosine {@link HnswIndex}.
 *
 * <p>Embeddings are L2-normalized on insert and query. Chunk identifiers are mapped
 * bidirectionally to internal HNSW ids; re-upserting a chunk tombstones its previous
 * internal id and allocates a fresh one, so stale ids never reappear in search
 * results. Persisted state consists of {@code index.bin} (the serialized graph) and
 * {@code ids.jsonl} (one {@code chunkId<TAB>internalId} line per live chunk), both
 * written into the configured index directory.
 *
 * <p>Thread-safety: all mutations ({@link #upsert}, {@link #rebuild},
 * {@link #persist} and {@link #close}) take the write lock and all
 * {@link #search(float[], int)} calls take the read lock of a
 * {@link ReentrantReadWriteLock}; concurrent readers and serialized writers are safe.
 * The backing {@link HnswIndex} is itself concurrent but is guarded here so that its
 * internal ids and the id maps stay consistent. Instances are safe for concurrent use
 * by multiple threads until {@link #close()} is called.
 */
public final class HnswForgeBackend implements VectorSearchBackend, AutoCloseable {

    private static final String INDEX_FILE = "index.bin";
    private static final String IDS_FILE = "ids.jsonl";
    private static final String IDS_TEMP_FILE = "ids.jsonl.tmp";
    private static final int DEFAULT_M = 16;
    private static final int DEFAULT_EF_CONSTRUCTION = 200;
    private static final int DEFAULT_EF_SEARCH = 100;
    private static final long DEFAULT_SEED = 42L;

    private final Path indexDirectory;
    private final HnswParams params;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private int dimension;
    private HnswIndex index;
    private Map<String, Integer> chunkToId;
    private Map<Integer, String> idToChunk;
    private boolean closed;

    /**
     * Creates an empty backend with {@code m=16}, {@code efConstruction=200},
     * {@code efSearch=100} and {@code seed=42}.
     *
     * @param indexDirectory directory used by {@link #persist()} and {@link #load(Path, int)}
     * @param dimension      embedding dimension, must be positive
     * @throws IllegalArgumentException if {@code dimension} is not positive
     * @throws NullPointerException     if {@code indexDirectory} is null
     */
    public HnswForgeBackend(Path indexDirectory, int dimension) {
        this(indexDirectory, dimension, DEFAULT_M, DEFAULT_EF_CONSTRUCTION,
                DEFAULT_EF_SEARCH, DEFAULT_SEED);
    }

    /**
     * Creates an empty backend with explicit HNSW tuning parameters.
     *
     * @param indexDirectory directory used by {@link #persist()} and {@link #load(Path, int)}
     * @param dimension      embedding dimension, must be positive
     * @param m              maximum upper-layer degree, in {@code [2, 128]}
     * @param efConstruction insertion candidate list size, at least {@code m}
     * @param efSearch       search candidate list size, at least one
     * @param seed           seed for the deterministic level generator
     * @throws IllegalArgumentException if any numeric constraint is violated
     * @throws NullPointerException     if {@code indexDirectory} is null
     */
    public HnswForgeBackend(Path indexDirectory, int dimension, int m, int efConstruction,
                            int efSearch, long seed) {
        this.indexDirectory = Objects.requireNonNull(indexDirectory, "indexDirectory must not be null");
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, got " + dimension);
        }
        this.dimension = dimension;
        this.params = HnswParams.builder(dimension, Metric.COSINE)
                .m(m)
                .efConstruction(efConstruction)
                .efSearch(efSearch)
                .seed(seed)
                .build();
        this.index = new HnswIndex(params);
        this.chunkToId = new HashMap<>();
        this.idToChunk = new HashMap<>();
    }

    private HnswForgeBackend(Path indexDirectory, HnswParams params, HnswIndex index,
                             Map<String, Integer> chunkToId, Map<Integer, String> idToChunk) {
        this.indexDirectory = indexDirectory;
        this.params = params;
        this.dimension = params.dimension();
        this.index = index;
        this.chunkToId = chunkToId;
        this.idToChunk = idToChunk;
    }

    /**
     * Restores a backend from a directory written by {@link #persist()}.
     *
     * <p>Both {@code index.bin} and {@code ids.jsonl} must be present and mutually
     * consistent. Every mapped internal id must fall inside the restored index and
     * every chunk id and internal id must appear exactly once.
     *
     * <p>Thread-safety: the returned instance is safe for concurrent use.
     *
     * @param indexDirectory directory containing the persisted files
     * @param dimension      expected embedding dimension, must be positive
     * @return a new backend owning the restored index
     * @throws IOException              if a file is missing, unreadable, malformed or inconsistent
     * @throws IllegalArgumentException if {@code dimension} is not positive
     * @throws NullPointerException     if {@code indexDirectory} is null
     */
    public static HnswForgeBackend load(Path indexDirectory, int dimension) throws IOException {
        Objects.requireNonNull(indexDirectory, "indexDirectory must not be null");
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, got " + dimension);
        }
        Path indexFile = indexDirectory.resolve(INDEX_FILE);
        Path idsFile = indexDirectory.resolve(IDS_FILE);
        if (!Files.isRegularFile(indexFile)) {
            throw new IOException("Missing HNSW index file: " + indexFile);
        }
        if (!Files.isRegularFile(idsFile)) {
            throw new IOException("Missing chunk-id map file: " + idsFile);
        }
        HnswIndex restored;
        try {
            restored = HnswIndex.load(indexFile);
        } catch (IOException | RuntimeException failure) {
            throw new IOException("Failed to load HNSW index from " + indexFile + ": "
                    + failure.getMessage(), failure);
        }
        boolean complete = false;
        try {
            if (restored.dimension() != dimension) {
                throw new IOException("Index dimension mismatch in " + indexFile
                        + ": expected " + dimension + ", found " + restored.dimension());
            }
            if (restored.metric() != Metric.COSINE) {
                throw new IOException("Index metric mismatch in " + indexFile
                        + ": expected COSINE, found " + restored.metric());
            }
            List<String> lines;
            try {
                lines = Files.readAllLines(idsFile, StandardCharsets.UTF_8);
            } catch (IOException failure) {
                throw new IOException("Failed to read chunk-id map " + idsFile + ": "
                        + failure.getMessage(), failure);
            }
            Map<String, Integer> chunkToId = new HashMap<>();
            Map<Integer, String> idToChunk = new HashMap<>();
            for (int lineNumber = 1; lineNumber <= lines.size(); lineNumber++) {
                String line = lines.get(lineNumber - 1);
                int tab = line.indexOf('\t');
                if (tab <= 0 || tab != line.lastIndexOf('\t')) {
                    throw new IOException("Malformed " + IDS_FILE + " line " + lineNumber
                            + " in " + idsFile + ": expected one chunkId<TAB>internalId pair");
                }
                String chunkId = line.substring(0, tab);
                int id;
                try {
                    id = Integer.parseInt(line.substring(tab + 1));
                } catch (NumberFormatException failure) {
                    throw new IOException("Malformed internal id on " + IDS_FILE + " line "
                            + lineNumber + " in " + idsFile, failure);
                }
                if (id < 0 || id >= restored.size()) {
                    throw new IOException("Internal id out of range on " + IDS_FILE + " line "
                            + lineNumber + ": " + id + ", index size " + restored.size());
                }
                if (chunkToId.containsKey(chunkId)) {
                    throw new IOException("Duplicate chunk id on " + IDS_FILE + " line "
                            + lineNumber + ": " + chunkId);
                }
                if (idToChunk.containsKey(id)) {
                    throw new IOException("Duplicate internal id on " + IDS_FILE + " line "
                            + lineNumber + ": " + id);
                }
                chunkToId.put(chunkId, id);
                idToChunk.put(id, chunkId);
            }
            complete = true;
            return new HnswForgeBackend(indexDirectory, restored.params(), restored,
                    chunkToId, idToChunk);
        } finally {
            if (!complete) {
                restored.close();
            }
        }
    }

    /**
     * Inserts or replaces a chunk, tombstoning any previous internal id.
     *
     * <p>Thread-safety: takes the write lock; safe for concurrent callers.
     *
     * @param chunkId   non-blank chunk identifier without tab or newline characters
     * @param embedding non-null embedding of length {@link #dimension()}
     * @throws IllegalArgumentException if either argument is invalid
     * @throws IllegalStateException    if the backend has been closed
     */
    @Override
    public void upsert(String chunkId, float[] embedding) {
        validateChunkId(chunkId);
        float[] normalized = normalize(embedding);
        lock.writeLock().lock();
        try {
            ensureOpen();
            Integer previous = chunkToId.get(chunkId);
            if (previous != null) {
                index.delete(previous);
                idToChunk.remove(previous);
            }
            int id = index.add(normalized);
            chunkToId.put(chunkId, id);
            idToChunk.put(id, chunkId);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Returns the live chunks closest to the query embedding.
     *
     * <p>Scores are {@code 1 - cosineDistance}. Internal ids without a current chunk
     * mapping are discarded before sorting by descending score and chunk id.
     *
     * <p>Thread-safety: takes the read lock; safe for concurrent callers.
     *
     * @param queryEmbedding non-null embedding of length {@link #dimension()}
     * @param k              maximum number of results, must be positive
     * @return at most {@code k} results sorted by descending score
     * @throws IllegalArgumentException if the query is invalid or {@code k} is not positive
     * @throws IllegalStateException    if the backend has been closed
     */
    @Override
    public List<ScoredChunk> search(float[] queryEmbedding, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        float[] normalized = normalize(queryEmbedding);
        lock.readLock().lock();
        try {
            ensureOpen();
            int stale = Math.max(0, index.size() - chunkToId.size());
            int fetch = (int) Math.min(index.size(), (long) k + stale);
            if (fetch <= 0) {
                return List.of();
            }
            List<ScoredChunk> results = new ArrayList<>(Math.min(fetch, k));
            for (SearchResult result : index.search(normalized, fetch)) {
                String chunkId = idToChunk.get(result.id());
                if (chunkId == null) {
                    continue;
                }
                results.add(new ScoredChunk(chunkId, 1.0f - result.distance()));
            }
            results.sort(Comparator.comparingDouble(ScoredChunk::score).reversed()
                    .thenComparing(ScoredChunk::chunkId));
            if (results.size() > k) {
                return List.copyOf(results.subList(0, k));
            }
            return List.copyOf(results);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Replaces the current index with a fresh index built from the supplied chunks.
     *
     * <p>Thread-safety: takes the write lock; safe for concurrent callers.
     *
     * @param chunks non-null iterable of non-null, uniquely identified chunks
     * @throws IllegalArgumentException if a chunk, embedding or identifier is invalid or duplicated
     * @throws IllegalStateException    if the backend has been closed
     */
    @Override
    public void rebuild(Iterable<Chunk> chunks) {
        Objects.requireNonNull(chunks, "chunks must not be null");
        List<Chunk> snapshot = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (Chunk chunk : chunks) {
            Objects.requireNonNull(chunk, "chunk must not be null");
            validateChunkId(chunk.chunkId());
            if (!seen.add(chunk.chunkId())) {
                throw new IllegalArgumentException("duplicate chunkId: " + chunk.chunkId());
            }
            snapshot.add(chunk);
        }
        lock.writeLock().lock();
        try {
            ensureOpen();
            HnswIndex replacement = new HnswIndex(params);
            Map<String, Integer> nextChunkToId = new HashMap<>();
            Map<Integer, String> nextIdToChunk = new HashMap<>();
            boolean complete = false;
            try {
                for (Chunk chunk : snapshot) {
                    int id = replacement.add(normalize(chunk.embedding()));
                    nextChunkToId.put(chunk.chunkId(), id);
                    nextIdToChunk.put(id, chunk.chunkId());
                }
                complete = true;
            } finally {
                if (!complete) {
                    replacement.close();
                }
            }
            HnswIndex previous = index;
            index = replacement;
            chunkToId = nextChunkToId;
            idToChunk = nextIdToChunk;
            previous.close();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Writes {@code index.bin} and atomically replaces {@code ids.jsonl} in the
     * configured directory, creating the directory if needed.
     *
     * <p>Thread-safety: takes the write lock; safe for concurrent callers. Failures
     * are wrapped in {@link UncheckedIOException} rather than suppressed.
     *
     * @throws IllegalStateException  if the backend has been closed
     * @throws UncheckedIOException  if writing or moving either file fails
     */
    @Override
    public void persist() {
        lock.writeLock().lock();
        try {
            ensureOpen();
            try {
                Files.createDirectories(indexDirectory);
                index.save(indexDirectory.resolve(INDEX_FILE));
                Path target = indexDirectory.resolve(IDS_FILE);
                Path temp = indexDirectory.resolve(IDS_TEMP_FILE);
                StringBuilder content = new StringBuilder();
                for (Map.Entry<String, Integer> entry : chunkToId.entrySet()) {
                    content.append(entry.getKey()).append('\t')
                            .append(entry.getValue()).append('\n');
                }
                Files.writeString(temp, content, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE);
                try {
                    Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE);
                } catch (AtomicMoveNotSupportedException unsupported) {
                    Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
                }
            } catch (IOException failure) {
                throw new UncheckedIOException(
                        "Failed to persist HNSWForge index to " + indexDirectory, failure);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Releases the backing index. After this call every operation on this instance
     * fails with {@link IllegalStateException}; calling {@code close()} more than once
     * has no effect.
     *
     * <p>Thread-safety: takes the write lock; safe to call concurrently with operations.
     */
    @Override
    public void close() {
        lock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            index.close();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Returns the configured embedding dimension.
     *
     * <p>Thread-safety: reads a value fixed at construction; safe at any time.
     *
     * @return the configured embedding dimension
     */
    public int dimension() {
        return dimension;
    }

    /**
     * Returns the number of live chunks currently indexed.
     *
     * <p>Thread-safety: takes the read lock; safe for concurrent callers.
     *
     * @return the number of live chunks currently indexed
     * @throws IllegalStateException if the backend has been closed
     */
    public int liveCount() {
        lock.readLock().lock();
        try {
            ensureOpen();
            return chunkToId.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("backend is closed");
        }
    }

    private float[] normalize(float[] vector) {
        Objects.requireNonNull(vector, "embedding must not be null");
        if (vector.length != dimension) {
            throw new IllegalArgumentException("dimension mismatch: expected "
                    + dimension + ", got " + vector.length);
        }
        float[] copy = vector.clone();
        double sum = 0.0;
        for (float value : copy) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException("embedding values must not be NaN");
            }
            sum += (double) value * value;
        }
        if (sum == 0.0) {
            throw new IllegalArgumentException("cosine distance is undefined for a zero vector");
        }
        double norm = Math.sqrt(sum);
        for (int i = 0; i < copy.length; i++) {
            copy[i] = (float) (copy[i] / norm);
        }
        return copy;
    }

    private static void validateChunkId(String chunkId) {
        if (chunkId == null || chunkId.isBlank()) {
            throw new IllegalArgumentException("chunkId must not be null or blank");
        }
        if (chunkId.indexOf('\t') >= 0 || chunkId.indexOf('\n') >= 0
                || chunkId.indexOf('\r') >= 0) {
            throw new IllegalArgumentException("chunkId must not contain tabs or newlines: " + chunkId);
        }
    }
}
