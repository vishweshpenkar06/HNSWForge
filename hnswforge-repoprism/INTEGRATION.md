# Wiring HNSWForge into RepoPrism

This module supplies a `VectorSearchBackend` implementation backed by HNSWForge and an
offline hit@6 evaluation CLI. HNSWForge replaces **only the dense/vector leg** of
retrieval: the lexical leg, the pgvector backend and the existing RRF fusion stay
unchanged.

## 1. Add the Maven dependency

RepoPrism consumes the module from JitPack (replace the placeholder owner/tag with a
real release coordinate):

```xml
<repositories>
  <repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
  </repository>
</repositories>

<dependency>
  <groupId>com.github.REPO_OWNER</groupId>
  <artifactId>hnswforge-repoprism</artifactId>
  <version>RELEASE_TAG</version>
</dependency>
```

The artifact depends only on `hnswforge-core`; no other runtime dependencies are added.

## 2. Declare the two conditional beans

Add one Spring `@Configuration` class to RepoPrism. pgvector remains the default when
the property is absent; HNSWForge is selected explicitly.

```java
@Configuration
public class RetrievalBackendConfiguration {

    private final ObjectProvider<HnswForgeBackend> hnswforgeBackend;

    public RetrievalBackendConfiguration(ObjectProvider<HnswForgeBackend> hnswforgeBackend) {
        this.hnswforgeBackend = hnswforgeBackend;
    }

    @Bean
    @ConditionalOnProperty(name = "retrieval.backend", havingValue = "pgvector",
            matchIfMissing = true)
    VectorSearchBackend pgvectorBackend(PgVectorProperties properties) {
        return new PgVectorSearchBackend(properties);
    }

    @Bean
    @ConditionalOnProperty(name = "retrieval.backend", havingValue = "hnswforge")
    HnswForgeBackend hnswforgeBackend(
            @Value("${retrieval.hnswforge.index-dir}") String indexDirectory,
            @Value("${retrieval.hnswforge.dimension:384}") int dimension) {
        return new HnswForgeBackend(Path.of(indexDirectory), dimension);
    }

    @PreDestroy
    void persistOnShutdown() {
        HnswForgeBackend backend = hnswforgeBackend.getIfAvailable();
        if (backend != null) {
            backend.persist();
            backend.close();
        }
    }
}
```

Existing code continues to inject `VectorSearchBackend`; the concrete bean changes
with `retrieval.backend`. The full constructor
`(Path, int dimension, int m, int efConstruction, int efSearch, long seed)` is available
when tuning is needed; defaults are `m=16`, `efConstruction=200`, `efSearch=100`, `seed=42`.

## 3. Configure the index

```properties
retrieval.backend=hnswforge
retrieval.hnswforge.index-dir=/var/lib/repoprism/hnswforge
retrieval.hnswforge.dimension=384
```

The dimension must match the embedding model (384 for the default RepoPrism dense
model). The directory is created on first `persist()` and must contain `index.bin` and
`ids.jsonl` for `HnswForgeBackend.load(directory, 384)` to succeed.
`persist()` writes both files atomically; `close()` releases the index but does **not**
persist, so call `persist()` first on shutdown as shown above.

## 4. Keep RRF fusion unchanged

Do not modify the fusion code. The dense leg now calls `upsert`, `search` and `rebuild`
on whichever `VectorSearchBackend` bean is active. Both backends return the same
`ScoredChunk` values, so reciprocal-rank fusion consumes identical shapes. When
switching backends, `rebuild` all chunks once (for example on startup when
`ids.jsonl` is missing) to populate the new index.

## 5. Run the evaluation with real exports

Export chunks, evaluation queries and gold judgments as flat JSONL from RepoPrism, then
run the CLI from the built module:

```bash
mvn -pl hnswforge-repoprism -am package
java -cp "hnswforge-repoprism/target/hnswforge-repoprism-0.1.0-SNAPSHOT.jar:hnswforge-core/target/hnswforge-core-0.1.0-SNAPSHOT.jar" \
  dev.hnswforge.repoprism.ReproprismEval \
  --embeddings chunks.jsonl --queries eval72.jsonl --gold gold.jsonl --out report.csv
```

Input formats (one flat JSON object per line):

```json
{"chunk_id":"src/Foo.java:12","embedding":[0.01,0.02]}
{"query_id":"q1","embedding":[0.03,0.04]}
{"query_id":"q1","chunk_ids":["src/Foo.java:12"]}
```

The tool prints a before/after hit@6 table for exact brute force, HNSWForge-only and
hybrid RRF, and optionally writes the same numbers as CSV. It deliberately does not
report pgvector-only hit@6: those numbers require a real RepoPrism pgvector export and
are not computed from the JSONL fixture.
