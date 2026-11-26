package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.hnswforge.util.SyntheticData;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Structural invariants of the HNSW graph. */
class HnswGraphInvariantsTest {

    @Test
    void degreesRespectBoundsAndLinksAreValid() {
        HnswParams params = HnswParams.builder(8, Metric.L2).m(8).efConstruction(100)
                .maxElements(5_000).seed(11L).build();
        float[][] vectors = SyntheticData.uniform(5_000, 8, 11L);
        HnswIndex index = HnswTestUtil.build(vectors, params);

        for (int id = 0; id < index.size(); id++) {
            int level = index.level(id);
            for (int layer = 0; layer <= level; layer++) {
                int count = index.neighborCount(id, layer);
                int bound = layer == 0 ? params.maxM0() : params.m();
                assertTrue(count <= bound, "degree " + count + " exceeds " + bound);
                Set<Integer> seen = new HashSet<>();
                for (int i = 0; i < count; i++) {
                    int neighbor = index.neighborAt(id, layer, i);
                    assertTrue(neighbor >= 0 && neighbor < index.size(), "invalid neighbor id");
                    assertNotEquals(id, neighbor, "self loop");
                    assertTrue(seen.add(neighbor), "duplicate neighbor");
                }
            }
        }
    }

    @Test
    void everyNodeIsReachableFromEntryPointAtLayerZero() {
        HnswParams params = HnswParams.builder(6, Metric.L2).m(8).efConstruction(100)
                .maxElements(2_000).seed(23L).build();
        float[][] vectors = SyntheticData.uniform(2_000, 6, 23L);
        HnswIndex index = HnswTestUtil.build(vectors, params);

        Set<Integer> visited = new HashSet<>();
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(index.entryPoint());
        visited.add(index.entryPoint());
        while (!queue.isEmpty()) {
            int current = queue.poll();
            for (int i = 0; i < index.neighborCount(current, 0); i++) {
                int neighbor = index.neighborAt(current, 0, i);
                if (visited.add(neighbor)) {
                    queue.add(neighbor);
                }
            }
        }
        assertEquals(index.size(), visited.size());
    }

    @Test
    void sameSeedAndOrderProduceIdenticalGraphs() {
        HnswParams params = HnswParams.builder(6, Metric.L2).m(8).efConstruction(60)
                .maxElements(1_000).seed(99L).build();
        float[][] vectors = SyntheticData.uniform(1_000, 6, 5L);
        HnswIndex first = HnswTestUtil.build(vectors, params);
        HnswIndex second = HnswTestUtil.build(vectors, params);
        assertEquals(first.entryPoint(), second.entryPoint());
        assertEquals(first.maxLevel(), second.maxLevel());
        for (int id = 0; id < first.size(); id++) {
            assertEquals(first.level(id), second.level(id));
            for (int layer = 0; layer <= first.level(id); layer++) {
                assertEquals(first.neighborCount(id, layer), second.neighborCount(id, layer));
                for (int i = 0; i < first.neighborCount(id, layer); i++) {
                    assertEquals(first.neighborAt(id, layer, i),
                            second.neighborAt(id, layer, i));
                }
            }
        }
    }

    @Test
    void differentSeedProducesDifferentGraph() {
        float[][] vectors = SyntheticData.uniform(500, 6, 5L);
        HnswIndex first = HnswTestUtil.build(vectors,
                HnswParams.builder(6, Metric.L2).m(8).efConstruction(60)
                        .maxElements(500).seed(1L).build());
        HnswIndex second = HnswTestUtil.build(vectors,
                HnswParams.builder(6, Metric.L2).m(8).efConstruction(60)
                        .maxElements(500).seed(2L).build());
        boolean anyDifference = false;
        for (int id = 0; id < first.size() && !anyDifference; id++) {
            for (int layer = 0; layer <= Math.min(first.level(id), second.level(id)); layer++) {
                for (int i = 0; i < Math.min(first.neighborCount(id, layer),
                        second.neighborCount(id, layer)); i++) {
                    if (first.neighborAt(id, layer, i) != second.neighborAt(id, layer, i)) {
                        anyDifference = true;
                    }
                }
            }
        }
        assertTrue(anyDifference || first.maxLevel() != second.maxLevel());
    }

    @Test
    void levelDistributionIsRoughlyGeometric() {
        HnswParams params = HnswParams.builder(6, Metric.L2).m(16).efConstruction(100)
                .maxElements(5_000).seed(77L).build();
        HnswIndex index = HnswTestUtil.build(
                SyntheticData.uniform(5_000, 6, 77L), params);
        int aboveZero = 0;
        int aboveOne = 0;
        for (int id = 0; id < index.size(); id++) {
            if (index.level(id) >= 1) {
                aboveZero++;
            }
            if (index.level(id) >= 2) {
                aboveOne++;
            }
        }
        assertTrue(aboveZero > 150 && aboveZero < 500, "level>=1 count was " + aboveZero);
        assertTrue(aboveOne > 0 && aboveOne < 60, "level>=2 count was " + aboveOne);
    }
}
