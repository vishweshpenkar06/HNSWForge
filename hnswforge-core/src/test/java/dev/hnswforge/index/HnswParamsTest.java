package dev.hnswforge.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** Tests for parameter validation and derived values. */
class HnswParamsTest {

    @Test
    void builderDefaults() {
        HnswParams params = HnswParams.builder(128, Metric.L2).build();
        assertEquals(128, params.dimension());
        assertEquals(Metric.L2, params.metric());
        assertEquals(16, params.m());
        assertEquals(200, params.efConstruction());
        assertEquals(64, params.efSearch());
        assertEquals(100_000, params.maxElements());
        assertEquals(42L, params.seed());
        assertEquals(NeighborSelection.HEURISTIC, params.neighborSelection());
    }

    @Test
    void builderAppliesOverrides() {
        HnswParams params = HnswParams.builder(8, Metric.COSINE)
                .m(8).efConstruction(100).efSearch(32).maxElements(500)
                .seed(7L).neighborSelection(NeighborSelection.SIMPLE).build();
        assertEquals(8, params.m());
        assertEquals(100, params.efConstruction());
        assertEquals(32, params.efSearch());
        assertEquals(500, params.maxElements());
        assertEquals(7L, params.seed());
        assertEquals(NeighborSelection.SIMPLE, params.neighborSelection());
    }

    @Test
    void sevenArgumentConstructorDefaultsToHeuristic() {
        HnswParams params = new HnswParams(4, Metric.INNER_PRODUCT, 4, 40, 8, 100, 1L);
        assertEquals(NeighborSelection.HEURISTIC, params.neighborSelection());
    }

    @Test
    void derivedValuesAreCorrect() {
        HnswParams params = HnswParams.builder(4, Metric.L2).m(16).build();
        assertEquals(32, params.maxM0());
        assertEquals(1.0 / Math.log(16), params.levelMult(), 1e-12);
    }

    @Test
    void invalidDimensionThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(0, Metric.L2).build());
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(-1, Metric.L2).build());
    }

    @Test
    void invalidMThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(2, Metric.L2).m(1).efConstruction(1).build());
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(2, Metric.L2).m(129).build());
    }

    @Test
    void invalidEfThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(2, Metric.L2).m(16).efConstruction(15).build());
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(2, Metric.L2).efSearch(0).build());
    }

    @Test
    void invalidCapacityThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> HnswParams.builder(2, Metric.L2).maxElements(0).build());
    }

    @Test
    void nullMetricThrows() {
        assertThrows(NullPointerException.class, () -> HnswParams.builder(2, null).build());
    }

    @Test
    void nullSelectionThrows() {
        assertThrows(NullPointerException.class,
                () -> HnswParams.builder(2, Metric.L2).neighborSelection(null).build());
    }
}
