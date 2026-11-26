package dev.hnswforge.server;

import dev.hnswforge.index.Metric;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * HTTP-level tests for all four endpoints and their validation paths.
 *
 * <p>Each test starts a fresh server on an ephemeral port and stops it afterwards.
 * Thread-safety: tests are sequential and use a single {@link HttpClient}, which is
 * itself thread-safe; the server exercises concurrent request handling.
 */
class RestServerTest {
    @TempDir
    Path tempDir;

    private RestServer server;
    private HttpClient client;
    private URI baseUri;

    @BeforeEach
    void setUp() {
        server = new RestServer(0, 2, Metric.L2, 16);
        client = HttpClient.newHttpClient();
        baseUri = URI.create("http://localhost:" + server.port() + "/");
    }

    @AfterEach
    void tearDown() {
        server.stop(0);
    }

    @Test
    void healthReportsEmptyIndex() throws Exception {
        HttpResponse<String> response = get("/health");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"status\":\"ok\""), response.body());
        assertTrue(response.body().contains("\"size\":0"), response.body());
    }

    @Test
    void addingVectorsReturnsSequentialIds() throws Exception {
        HttpResponse<String> first = post("/vectors", "{\"vector\":[1.0,0.0]}");
        assertEquals(201, first.statusCode());
        assertEquals("{\"id\":0}", first.body());

        HttpResponse<String> second = post("/vectors", "{\"vector\":[0.0,1.0]}");
        assertEquals(201, second.statusCode());
        assertEquals("{\"id\":1}", second.body());

        HttpResponse<String> health = get("/health");
        assertEquals(200, health.statusCode());
        assertTrue(health.body().contains("\"size\":2"), health.body());
    }

    @Test
    void searchReturnsNearestVectorsWithExplicitEf() throws Exception {
        assertEquals(201, post("/vectors", "{\"vector\":[1.0,0.0]}").statusCode());
        assertEquals(201, post("/vectors", "{\"vector\":[0.0,1.0]}").statusCode());

        HttpResponse<String> response = post("/search", "{\"vector\":[1.0,0.0],\"k\":2,\"ef\":8}");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().startsWith("{\"results\":["), response.body());
        assertTrue(response.body().contains("{\"id\":0,\"distance\":0.0}"), response.body());
        assertTrue(response.body().contains("\"id\":1"), response.body());
    }

    @Test
    void searchWithoutEfUsesDefaultSearchEffort() throws Exception {
        assertEquals(201, post("/vectors", "{\"vector\":[3.0,4.0]}").statusCode());

        HttpResponse<String> response = post("/search", "{\"vector\":[3.0,4.0],\"k\":1}");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"id\":0"), response.body());
    }

    @Test
    void snapshotWritesIndexFile() throws Exception {
        assertEquals(201, post("/vectors", "{\"vector\":[1.0,2.0]}").statusCode());
        Path snapshot = tempDir.resolve("index.bin");
        String path = snapshot.toString().replace("\\", "\\\\");

        HttpResponse<String> response = post("/snapshot", "{\"path\":\"" + path + "\"}");
        assertEquals(200, response.statusCode());
        assertEquals("{\"saved\":true}", response.body());
        assertTrue(Files.exists(snapshot));
        assertTrue(Files.size(snapshot) > 0);
    }

    @Test
    void badInputsReturn400() throws Exception {
        assertBadRequest(post("/vectors", "{\"vector\":[1.0]}"));
        assertBadRequest(post("/vectors", "{\"vector\":[1.0,\"x\"]}"));
        assertBadRequest(post("/search", "{\"vector\":[1.0,2.0]}"));
        assertBadRequest(post("/search", "{\"vector\":[1.0,2.0],\"k\":0}"));
        assertBadRequest(post("/search", "{\"vector\":[1.0,2.0],\"k\":1,\"ef\":-1}"));
        assertBadRequest(post("/snapshot", "{\"path\":123}"));
        assertBadRequest(post("/snapshot", "not json"));
        assertBadRequest(get("/vectors"));
        assertBadRequest(client.send(HttpRequest.newBuilder(baseUri.resolve("health"))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build(), HttpResponse.BodyHandlers.ofString()));
    }

    private HttpResponse<String> get(String path) throws IOException, InterruptedException {
        return client.send(HttpRequest.newBuilder(baseUri.resolve(path)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> post(String path, String body) throws IOException, InterruptedException {
        return client.send(HttpRequest.newBuilder(baseUri.resolve(path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build(), HttpResponse.BodyHandlers.ofString());
    }

    private static void assertBadRequest(HttpResponse<String> response) {
        assertEquals(400, response.statusCode(), response.body());
        assertTrue(response.body().startsWith("{\"error\":\""), response.body());
    }
}
