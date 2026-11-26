#!/usr/bin/env bash
# Dataset URLs and dataset names may need updating for this machine; see scripts/download_datasets.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results docs/charts

JAVA_FLAGS=(--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xms4g -Xmx4g -XX:+UseParallelGC)
EVAL_CP="hnswforge-eval/target/classes:hnswforge-core/target/classes"
COMPARE_CP="hnswforge-bench/target/classes:hnswforge-core/target/classes:hnswforge-eval/target/classes"

printf 'building HNSWForge\n'
./mvnw -q verify

printf 'running Java sweep\n'
java "${JAVA_FLAGS[@]}" -cp "${EVAL_CP}" dev.hnswforge.eval.Main sweep \
  --dataset synthetic --n 10000 \
  --m 8,16,32 --efc 100,200 \
  --efs 10,20,40,80,160,320 --k 10 \
  --out results/sweep_synthetic.csv

printf 'running hnswlib baseline\n'
if ! python3 -c 'import hnswlib' >/dev/null 2>&1; then
  python3 -m pip install hnswlib
fi
python3 scripts/hnswlib_baseline.py --synthetic 10000 32 \
  --m 16 --efc 200 --efs 10,20,40,80,160,320 --k 10 \
  --out results/hnswlib.csv

printf 'running comparison harness\n'
if [[ -f hnswforge-bench/target/classes/dev/hnswforge/bench/CompareMain.class ]]; then
  java "${JAVA_FLAGS[@]}" -cp "${COMPARE_CP}" dev.hnswforge.bench.CompareMain \
    --dataset synthetic \
    --systems hnswforge,hnswlib \
    --efs 10,20,40,80,160,320 \
    --out results/compare.csv
else
  printf 'warning: CompareMain class not found; skipping results/compare.csv\n' >&2
fi

printf 'generating comparison charts\n'
if [[ -f results/compare.csv ]]; then
  WATERFALL_CSV="${WATERFALL_CSV:-}"
  if [[ -n "${WATERFALL_CSV}" ]]; then
    python3 scripts/plot_compare.py results/compare.csv "${WATERFALL_CSV}"
  else
    python3 scripts/plot_compare.py results/compare.csv
  fi
else
  printf 'warning: results/compare.csv not available; skipping charts\n' >&2
fi

printf 'benchmark run complete\n'
