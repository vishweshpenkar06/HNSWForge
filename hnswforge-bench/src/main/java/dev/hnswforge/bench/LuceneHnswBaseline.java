package dev.hnswforge.bench;

import dev.hnswforge.index.SearchResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;

/**
 * Lucene HNSW baseline used by the Phase 6 comparison harness.
 *
 * <p>Contract: {@link #addAll(float[][])} must be called exactly once before the
 * first {@link #search(float[], int)}. One document is indexed per vector with a
 * {@code KnnFloatVectorField("embedding", vector, EUCLIDEAN)} and a stored
 * {@code StringField("id", ...)} holding the insertion index; the segment is
 * force-merged into a single segment before queries run. Results carry ids that
 * match the input order and squared Euclidean distances in {@code SearchResult}.
 * Lucene's EUCLIDEAN similarity returns {@code score = 1 / (1 + distance)}, so the
 * reported distance is recovered as {@code (1 / score) - 1}. Lucene 10 exposes no
 * query-time {@code efSearch} knob; {@link #setEfSearch(int)} therefore accepts the
 * value for a uniform harness API but deliberately leaves Lucene's behavior
 * unchanged. The {@code m}, {@code efConstruction} and {@code maxElements}
 * constructor arguments are validated and retained for reporting; graph tuning is
 * left to Lucene's own defaults because the specified field constructor does not
 * expose {@code maxConn} or {@code beamWidth}.
 *
 * <p>Thread-safety: {@code addAll} must complete before any search begins and must
 * not be called twice. Thereafter {@code search} is safe for concurrent callers
 * because {@code IndexSearcher} and its stored-fields reader are thread-safe;
 * {@code setEfSearch} is a stateless no-op. {@code close} must be called only after
 * all searches have finished and is idempotent.
 */
public final class LuceneHnswBaseline implements ComparisonSystem {
    private static final String VECTOR_FIELD = "embedding";
    private static final String ID_FIELD = "id";

    private final int dimension;
    private final int m;
    private final int efConstruction;
    private final int maxElements;
    private final ByteBuffersDirectory directory;
    private final IndexWriter writer;
    private DirectoryReader reader;
    private IndexSearcher searcher;
    private int vectorCount;
    private boolean closed;

    /**
     * Creates an empty in-memory Lucene index for the given geometry.
     *
     * <p>Thread-safety: a newly constructed instance is owned by the calling
     * thread until it is handed to readers after {@code addAll}.
     *
     * @param dimension     vector dimension, must be positive
     * @param m             HNSW maximum degree, must be positive; recorded only
     * @param efConstruction HNSW construction effort, must be positive; recorded only
     * @param maxElements   maximum accepted vector count, must be positive
     * @throws IllegalArgumentException if any argument is not positive
     * @throws IllegalStateException if the Lucene directory or writer cannot be created
     */
    public LuceneHnswBaseline(int dimension, int m, int efConstruction, int maxElements) {
        if (dimension <= 0 || m <= 0 || efConstruction <= 0 || maxElements <= 0) {
            throw new IllegalArgumentException(
                    "dimension, m, efConstruction and maxElements must be positive");
        }
        this.dimension = dimension;
        this.m = m;
        this.efConstruction = efConstruction;
        this.maxElements = maxElements;
        try {
            this.directory = new ByteBuffersDirectory();
            this.writer = new IndexWriter(directory, new IndexWriterConfig());
        } catch (IOException exception) {
            throw new IllegalStateException("could not create the in-memory Lucene index", exception);
        }
    }

    /**
     * Adds all vectors and force-merges the result into a single segment.
     *
     * <p>Thread-safety: call once from a single thread, before any search.
     *
     * @param vectors equally sized, non-NaN vectors of length {@code dimension}
     * @throws IllegalArgumentException if {@code vectors} is null, empty, larger than
     *                                  {@code maxElements}, or contains an invalid vector
     * @throws IllegalStateException if the instance is closed, {@code addAll} was already
     *                               called, or Lucene fails to index the documents
     */
    public void addAll(float[][] vectors) {
        checkOpen();
        Objects.requireNonNull(vectors, "vectors must not be null");
        if (vectors.length == 0) {
            throw new IllegalArgumentException("vectors must not be empty");
        }
        if (vectors.length > maxElements) {
            throw new IllegalArgumentException(
                    "vector count exceeds maxElements: " + vectors.length + " > " + maxElements);
        }
        if (searcher != null) {
            throw new IllegalStateException("addAll may only be called once");
        }
        for (int i = 0; i < vectors.length; i++) {
            validateVector(vectors[i], i);
        }
        try {
            for (int i = 0; i < vectors.length; i++) {
                Document document = new Document();
                document.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[i],
                        VectorSimilarityFunction.EUCLIDEAN));
                document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
                writer.addDocument(document);
            }
            writer.forceMerge(1);
            reader = DirectoryReader.open(writer);
            searcher = new IndexSearcher(reader);
            vectorCount = vectors.length;
        } catch (IOException exception) {
            throw new IllegalStateException("could not build the Lucene index", exception);
        }
    }

    /**
     * Returns up to {@code k} nearest vectors with squared Euclidean distances.
     *
     * <p>Thread-safety: safe for concurrent callers after {@code addAll} completes.
     *
     * @param query query vector of length {@code dimension}, without NaN values
     * @param k     maximum number of results, must be positive
     * @return results sorted by ascending squared distance, then ascending id
     * @throws IllegalArgumentException if the query or {@code k} is invalid
     * @throws IllegalStateException if the instance is closed, no index has been built,
     *                               a stored id is malformed, or Lucene fails
     */
    public List<SearchResult> search(float[] query, int k) {
        checkOpen();
        validateVector(query, -1);
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        if (searcher == null) {
            return List.of();
        }
        TopDocs topDocs;
        try {
            topDocs = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, k), k);
        } catch (IOException exception) {
            throw new IllegalStateException("Lucene search failed", exception);
        }
        List<SearchResult> results = new ArrayList<>(topDocs.scoreDocs.length);
        try {
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                String storedId = searcher.storedFields().document(scoreDoc.doc).get(ID_FIELD);
                if (storedId == null) {
                    throw new IllegalStateException("stored id is missing for doc " + scoreDoc.doc);
                }
                int id;
                try {
                    id = Integer.parseInt(storedId);
                } catch (NumberFormatException exception) {
                    throw new IllegalStateException("malformed stored id: " + storedId, exception);
                }
                float score = scoreDoc.score;
                if (!Float.isFinite(score) || score <= 0f) {
                    throw new IllegalStateException("invalid Lucene EUCLIDEAN score: " + score);
                }
                results.add(new SearchResult(id, 1.0f / score - 1.0f));
            }
        } catch (IOException exception) {
            throw new IllegalStateException("could not read stored Lucene ids", exception);
        }
        results.sort(Comparator.comparingDouble(SearchResult::distance)
                .thenComparingInt(SearchResult::id));
        return results;
    }

    /**
     * Accepts a search-effort value without changing Lucene's query behavior.
     *
     * <p>Thread-safety: stateless and safe for concurrent callers.
     *
     * @param ignored requested search effort, must be positive
     * @throws IllegalArgumentException if {@code ignored} is not positive
     */
    public void setEfSearch(int ignored) {
        if (ignored <= 0) {
            throw new IllegalArgumentException("efSearch must be > 0, got " + ignored);
        }
    }

    /**
     * Estimates the in-memory index footprint as vectors times {@code dimension * 4 + 64}
     * bytes, covering float vectors and a fixed per-vector graph and metadata allowance.
     *
     * <p>Thread-safety: reads immutable state and is safe for concurrent callers.
     *
     * @return the non-negative byte estimate
     * @throws IllegalStateException if the instance is closed
     */
    public long indexBytesEstimate() {
        checkOpen();
        return (long) vectorCount * ((long) dimension * Float.BYTES + 64L);
    }

    /**
     * Releases the reader, writer and in-memory directory, in that order.
     *
     * <p>Thread-safety: call after all searches finish; repeated calls have no effect.
     *
     * @throws IllegalStateException if Lucene fails to release a resource
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        IOException failure = null;
        if (reader != null) {
            try {
                reader.close();
            } catch (IOException exception) {
                failure = exception;
            }
        }
        try {
            writer.close();
        } catch (IOException exception) {
            failure = accumulate(failure, exception);
        }
        try {
            directory.close();
        } catch (IOException exception) {
            failure = accumulate(failure, exception);
        }
        if (failure != null) {
            throw new IllegalStateException("could not close the Lucene index", failure);
        }
    }

    /**
     * Returns the configured maximum degree, recorded for reporting only.
     *
     * <p>Thread-safety: reads immutable state and is safe for concurrent callers.
     *
     * @return the constructor's {@code m} value
     */
    public int m() {
        return m;
    }

    /**
     * Returns the configured construction effort, recorded for reporting only.
     *
     * <p>Thread-safety: reads immutable state and is safe for concurrent callers.
     *
     * @return the constructor's {@code efConstruction} value
     */
    public int efConstruction() {
        return efConstruction;
    }

    private static IOException accumulate(IOException first, IOException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Lucene baseline is closed");
        }
    }

    private void validateVector(float[] vector, int index) {
        String name = index < 0 ? "query" : "vectors[" + index + "]";
        if (vector == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        if (vector.length != dimension) {
            throw new IllegalArgumentException(name + " has dimension " + vector.length
                    + " but expected " + dimension);
        }
        for (float value : vector) {
            if (Float.isNaN(value)) {
                throw new IllegalArgumentException(name + " must not contain NaN");
            }
        }
    }
}
