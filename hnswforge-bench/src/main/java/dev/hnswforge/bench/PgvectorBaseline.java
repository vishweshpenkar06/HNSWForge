package dev.hnswforge.bench;

import dev.hnswforge.index.SearchResult;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * pgvector HNSW baseline used by the Phase 6 comparison harness.
 *
 * <p>Contract: {@link #createSchema()} drops and recreates {@code items} with a
 * {@code vector(dimension)} column and an HNSW index using {@code vector_l2_ops}
 * with the constructor's {@code m} and {@code efConstruction}. {@link #addAll(float[][])}
 * inserts ids equal to input row indices in batches of 1000 using vector string
 * literals. {@link #search(float[], int)} orders by {@code <->} (L2) and reports the
 * squared L2 distance; the {@code index_bytes_est} value is {@code -1} because the
 * on-disk size is owned by PostgreSQL and is queried with server-side relation-size
 * functions rather than from the client. Every SQL failure is wrapped in an
 * {@link IllegalStateException} that tells the user to start the pgvector container.
 *
 * <p>Thread-safety: a JDBC {@code Connection} is not thread-safe, so instances must
 * be used from a single thread or externally synchronized. {@code close} is
 * idempotent and must run after all work. Measurements taken through this class
 * deliberately include network round trips and JDBC driver overhead, which is not
 * present in the in-process baselines; in-database {@code EXPLAIN ANALYZE} timings
 * would be needed to separate those costs.
 */
public final class PgvectorBaseline implements ComparisonSystem {
    private static final String URL = System.getProperty("pgvector.url",
            "jdbc:postgresql://localhost:5433/hnswforge");
    private static final String USER = "hnswforge";
    private static final String PASSWORD = "hnswforge";
    private static final String CONTAINER_HINT =
            "start the pgvector container with: docker compose -f scripts/docker-compose.pgvector.yml up -d";
    private static final int BATCH_SIZE = 1000;

    private final int dimension;
    private final int m;
    private final int efConstruction;
    private final Connection connection;
    private boolean closed;

    /**
     * Opens a JDBC connection to the local pgvector container.
     *
     * <p>Thread-safety: the instance is owned by the calling thread until used.
     *
     * @param dimension     vector dimension, must be positive
     * @param m             HNSW maximum degree, must be in {@code [2, 100]}
     * @param efConstruction HNSW construction effort, must be at least {@code m}
     * @throws IllegalArgumentException if an argument is invalid
     * @throws IllegalStateException if the connection cannot be opened; the message
     *                               explains how to start the container
     */
    public PgvectorBaseline(int dimension, int m, int efConstruction) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, got " + dimension);
        }
        if (m < 2 || m > 100) {
            throw new IllegalArgumentException("m must be in [2, 100], got " + m);
        }
        if (efConstruction < m) {
            throw new IllegalArgumentException("efConstruction must be >= m (" + m + ")");
        }
        this.dimension = dimension;
        this.m = m;
        this.efConstruction = efConstruction;
        try {
            this.connection = DriverManager.getConnection(URL, USER, PASSWORD);
        } catch (SQLException exception) {
            throw new IllegalStateException(
                    "could not connect to pgvector at " + URL + "; " + CONTAINER_HINT, exception);
        }
    }

    /**
     * Enables the {@code vector} extension and recreates the {@code items} table and index.
     *
     * <p>Thread-safety: call once from the owning thread before insertion.
     *
     * @throws IllegalStateException if the instance is closed or a statement fails;
     *                               the message explains how to start the container
     */
    public void createSchema() {
        checkOpen();
        try (Statement statement = connection.createStatement()) {
            statement.execute("CREATE EXTENSION IF NOT EXISTS vector");
            statement.execute("DROP TABLE IF EXISTS items");
            statement.execute("CREATE TABLE items (id int primary key, embedding vector("
                    + dimension + "))");
            statement.execute("CREATE INDEX items_embedding_hnsw_idx ON items "
                    + "USING hnsw (embedding vector_l2_ops) WITH (m = " + m
                    + ", ef_construction = " + efConstruction + ")");
        } catch (SQLException exception) {
            throw failure("could not create the pgvector schema", exception);
        }
    }

    /**
     * Inserts all vectors in batches of 1000, assigning ids from zero in input order.
     *
     * <p>Thread-safety: call once from the owning thread before searching.
     *
     * @param vectors equally sized, non-NaN vectors of length {@code dimension}
     * @throws IllegalArgumentException if {@code vectors} is null or contains an invalid vector
     * @throws IllegalStateException if the instance is closed or a batch fails;
     *                               the message explains how to start the container
     */
    public void addAll(float[][] vectors) {
        checkOpen();
        Objects.requireNonNull(vectors, "vectors must not be null");
        for (int i = 0; i < vectors.length; i++) {
            validateVector(vectors[i], i);
        }
        String sql = "INSERT INTO items (id, embedding) VALUES (?, ?::vector)";
        try (PreparedStatement statement = connection.prepareStatement(sql)) {
            int pending = 0;
            for (int i = 0; i < vectors.length; i++) {
                statement.setInt(1, i);
                statement.setString(2, vectorLiteral(vectors[i]));
                statement.addBatch();
                pending++;
                if (pending == BATCH_SIZE) {
                    statement.executeBatch();
                    pending = 0;
                }
            }
            if (pending > 0) {
                statement.executeBatch();
            }
        } catch (SQLException exception) {
            throw failure("could not insert vectors into pgvector", exception);
        }
    }

    /**
     * Sets the server-side {@code hnsw.ef_search} session parameter for later searches.
     *
     * <p>Thread-safety: must be called from the single thread owning the connection.
     *
     * @param ef search effort, must be positive
     * @throws IllegalArgumentException if {@code ef} is not positive
     * @throws IllegalStateException if the instance is closed or the statement fails;
     *                               the message explains how to start the container
     */
    public void setEfSearch(int ef) {
        checkOpen();
        if (ef <= 0) {
            throw new IllegalArgumentException("efSearch must be > 0, got " + ef);
        }
        try (Statement statement = connection.createStatement()) {
            statement.execute("SET hnsw.ef_search = " + ef);
        } catch (SQLException exception) {
            throw failure("could not set hnsw.ef_search", exception);
        }
    }

    /**
     * Returns up to {@code k} nearest ids with squared L2 distances.
     *
     * <p>Thread-safety: must be called from the single thread owning the connection.
     * Network and driver latency is included in the caller's timing.
     *
     * @param query query vector of length {@code dimension}, without NaN values
     * @param k     maximum number of results, must be positive
     * @return results ordered by ascending L2 distance
     * @throws IllegalArgumentException if the query or {@code k} is invalid
     * @throws IllegalStateException if the instance is closed or the query fails;
     *                               the message explains how to start the container
     */
    public List<SearchResult> search(float[] query, int k) {
        checkOpen();
        validateVector(query, -1);
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        String literal = vectorLiteral(query);
        String sql = "SELECT id, embedding <-> ?::vector AS distance FROM items "
                + "ORDER BY embedding <-> ?::vector LIMIT ?";
        try (PreparedStatement statement = connection.prepareStatement(sql)) {
            statement.setString(1, literal);
            statement.setString(2, literal);
            statement.setInt(3, k);
            try (ResultSet resultSet = statement.executeQuery()) {
                List<SearchResult> results = new ArrayList<>();
                while (resultSet.next()) {
                    double l2Distance = resultSet.getDouble("distance");
                    results.add(new SearchResult(resultSet.getInt("id"),
                            (float) (l2Distance * l2Distance)));
                }
                return results;
            }
        } catch (SQLException exception) {
            throw failure("pgvector search failed", exception);
        }
    }

    /**
     * Reports an unknown index size, because pgvector does not expose one through JDBC.
     *
     * <p>Thread-safety: reads no mutable state and is safe for concurrent callers.
     *
     * @return {@code -1}, meaning unknown; use {@code pg_relation_size} on the server
     *         for a real measurement
     */
    public long indexBytesEstimate() {
        return -1L;
    }

    /**
     * Closes the JDBC connection.
     *
     * <p>Thread-safety: call after all work; repeated calls have no effect.
     *
     * @throws IllegalStateException if the connection cannot be closed
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            connection.close();
        } catch (SQLException exception) {
            throw failure("could not close the pgvector connection", exception);
        }
    }

    private IllegalStateException failure(String operation, SQLException cause) {
        return new IllegalStateException(operation + " at " + URL + "; " + CONTAINER_HINT, cause);
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Pgvector baseline is closed");
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

    private String vectorLiteral(float[] vector) {
        StringBuilder literal = new StringBuilder(1 + vector.length * 12);
        literal.append('[');
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) {
                literal.append(',');
            }
            literal.append(Float.toString(vector[i]));
        }
        return literal.append(']').toString();
    }
}
