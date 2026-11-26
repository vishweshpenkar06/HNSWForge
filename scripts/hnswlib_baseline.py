import argparse
import csv
import time
from pathlib import Path

import numpy as np

CSV_HEADER = [
    "dataset", "n", "dim", "metric", "m", "efc", "efs", "k", "recall",
    "qps", "p50_us", "p95_us", "p99_us", "build_s", "index_bytes_est",
]
QUERY_COUNT = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark hnswlib with the same CSV schema as the Java HNSWForge sweep."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthetic",
        nargs=2,
        type=int,
        metavar=("N", "DIM"),
        help="generate N uniform float32 vectors of dimension DIM and 200 nearby queries",
    )
    source.add_argument(
        "--base",
        metavar="FVECS",
        help="base vectors in .fvecs format; must be used together with --queries",
    )
    parser.add_argument(
        "--queries",
        metavar="FVECS",
        help="query vectors in .fvecs format; requires --base",
    )
    parser.add_argument("--m", type=int, default=16, help="HNSW maximum degree (default: 16)")
    parser.add_argument(
        "--efc", type=int, default=200, help="efConstruction, must be >= --m (default: 200)"
    )
    parser.add_argument(
        "--efs",
        default="10,20,40,80,160,320",
        help="comma-separated efSearch values (default: 10,20,40,80,160,320)",
    )
    parser.add_argument("--k", type=int, default=10, help="neighbours per query (default: 10)")
    parser.add_argument(
        "--out",
        default="results/hnswlib.csv",
        help="output CSV path (default: results/hnswlib.csv)",
    )
    args = parser.parse_args()

    if args.base is None and args.queries is not None:
        parser.error("--queries requires --base")
    if args.base is not None and args.queries is None:
        parser.error("--base requires --queries")
    if args.synthetic is not None:
        if args.synthetic[0] <= 0 or args.synthetic[1] <= 0:
            parser.error("--synthetic N and DIM must be positive integers")
    if args.m < 2:
        parser.error("--m must be at least 2")
    if args.efc < args.m:
        parser.error("--efc must be >= --m")
    if args.k <= 0:
        parser.error("--k must be a positive integer")
    if not args.out:
        parser.error("--out must not be empty")
    try:
        efs = [int(token.strip()) for token in args.efs.split(",")]
    except ValueError:
        parser.error("--efs must be a comma-separated list of positive integers")
    if not efs or any(value <= 0 for value in efs):
        parser.error("--efs must be a comma-separated list of positive integers")
    args.efs_values = efs
    return args


def read_fvecs(path):
    target = Path(path)
    if not target.is_file():
        raise SystemExit("error: fvecs file not found: " + str(target))
    raw = np.fromfile(target, dtype=np.uint32)
    if raw.size == 0:
        raise SystemExit("error: fvecs file is empty: " + str(target))
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise SystemExit("error: malformed fvecs file: " + str(target))
    count = raw.size // (dim + 1)
    vectors = raw.reshape(count, dim + 1)[:, 1:].copy().view(np.float32)
    return np.ascontiguousarray(vectors, dtype=np.float32)


def make_synthetic(n, dim):
    base = np.random.default_rng(123).random((n, dim), dtype=np.float32)
    query_rng = np.random.default_rng(456)
    anchors = query_rng.integers(0, n, size=QUERY_COUNT)
    noise = query_rng.normal(0.0, 0.05, size=(QUERY_COUNT, dim)).astype(np.float32)
    queries = np.ascontiguousarray(base[anchors] + noise, dtype=np.float32)
    return base, queries


def exact_truth(base, queries, k, batch=256):
    truth = np.empty((queries.shape[0], k), dtype=np.int64)
    base_norm = np.einsum("ij,ij->i", base, base)
    for start in range(0, queries.shape[0], batch):
        block = queries[start:start + batch]
        dists = np.einsum("ij,ij->i", block, block)[:, None] + base_norm[None, :] - 2.0 * (block @ base.T)
        top = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(dists, top, axis=1), axis=1)
        truth[start:start + block.shape[0]] = np.take_along_axis(top, order, axis=1).astype(np.int64)
    return truth


def recall_at_k(labels, truth, k):
    hits = 0
    for found, expected in zip(labels, truth):
        hits += len({int(value) for value in found[:k]} & {int(value) for value in expected[:k]})
    return hits / (labels.shape[0] * k)


def main():
    args = parse_args()
    try:
        import hnswlib
    except ImportError as exc:
        raise SystemExit("error: hnswlib is not installed; run: python3 -m pip install hnswlib") from exc

    if args.synthetic is not None:
        n, dim = args.synthetic
        dataset = "synthetic"
        base, queries = make_synthetic(n, dim)
    else:
        base = read_fvecs(args.base)
        queries = read_fvecs(args.queries)
        dataset = Path(args.base).stem
        n, dim = base.shape
    if queries.shape[1] != dim:
        raise SystemExit("error: query dimension does not match base dimension")
    if args.k > n:
        raise SystemExit("error: --k must not exceed the number of base vectors")
    if queries.shape[0] == 0:
        raise SystemExit("error: no query vectors available")

    truth = exact_truth(base, queries, args.k)
    index = hnswlib.Index(space="l2", dim=dim)
    build_start = time.perf_counter()
    index.init_index(max_elements=n, M=args.m, ef_construction=args.efc)
    index.add_items(base, num_threads=1)
    build_s = time.perf_counter() - build_start
    index_bytes = (
        n * dim * 4
        + n * (1 + 2 * args.m) * 4
        + (n // (args.m - 1)) * (1 + args.m) * 4
    )

    rows = []
    for efs in args.efs_values:
        index.set_ef(efs)
        latencies = np.empty(queries.shape[0], dtype=np.float64)
        labels = np.empty((queries.shape[0], args.k), dtype=np.int64)
        start = time.perf_counter()
        for i in range(queries.shape[0]):
            query_start = time.perf_counter()
            found, _ = index.knn_query(queries[i:i + 1], k=args.k, num_threads=1)
            latencies[i] = (time.perf_counter() - query_start) * 1_000_000.0
            labels[i] = found[0]
        elapsed = time.perf_counter() - start
        qps = queries.shape[0] / elapsed
        recall = recall_at_k(labels, truth, args.k)
        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        rows.append([
            dataset, n, dim, "L2", args.m, args.efc, efs, args.k,
            "%.6f" % recall, "%.6f" % qps, "%.3f" % p50, "%.3f" % p95,
            "%.3f" % p99, "%.6f" % build_s, index_bytes,
        ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    print("hnswlib baseline: dataset=%s n=%d dim=%d m=%d efc=%d k=%d rows=%d build_s=%.3f -> %s"
          % (dataset, n, dim, args.m, args.efc, args.k, len(rows), build_s, out))


if __name__ == "__main__":
    main()
