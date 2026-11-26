package dev.hnswforge.eval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for the {@code sweep} subcommand.
 *
 * <p>Each test uses a private temporary directory and the deterministic
 * synthetic dataset, so runs are reproducible and independent.
 *
 * <p>Thread-safety: JUnit creates a fresh instance per test method.
 */
class SweepTest {

    private static final String HEADER =
            "dataset,n,dim,metric,m,efc,efs,k,recall,qps,p50_us,p95_us,p99_us,build_s,index_bytes_est";

    @TempDir
    Path tempDir;

    @Test
    void sweepWritesHeaderAndEightDataRows() throws IOException {
        Path out = tempDir.resolve("results.csv");

        int code = Main.run(new String[] {"sweep", "--dataset", "synthetic", "--n", "300",
                "--m", "4,8", "--efc", "40,80", "--efs", "10,20", "--k", "5",
                "--threads", "1", "--out", out.toString()});

        assertEquals(0, code);
        List<String> lines = Files.readAllLines(out);
        assertEquals(HEADER, lines.get(0));
        assertEquals(8, lines.size() - 1);
        for (int row = 1; row < lines.size(); row++) {
            String[] fields = lines.get(row).split(",", -1);
            assertEquals(15, fields.length, lines.get(row));
            double recall = Double.parseDouble(fields[8]);
            double qps = Double.parseDouble(fields[9]);
            assertTrue(recall >= 0.0 && recall <= 1.0, "recall out of range: " + recall);
            assertTrue(qps > 0.0, "qps must be positive: " + qps);
        }
    }

    @Test
    void invalidArgumentsReturnTwo() {
        Path out = tempDir.resolve("results.csv");
        assertEquals(2, Main.run(new String[] {"sweep", "--dataset", "bogus"}));
        assertEquals(2, Main.run(new String[] {"sweep", "--m", "4,x", "--out", out.toString()}));
        assertEquals(2, Main.run(new String[] {"sweep", "--m", "1", "--out", out.toString()}));
        assertEquals(2, Main.run(new String[] {"sweep", "--unknown", "1"}));
    }
}
