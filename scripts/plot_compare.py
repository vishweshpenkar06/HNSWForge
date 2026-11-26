import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cross-system HNSW comparison charts from a verify CSV."
    )
    parser.add_argument("compare_csv", metavar="COMPARE_CSV", help="comparison CSV to plot")
    parser.add_argument(
        "waterfall_csv",
        metavar="WATERFALL_CSV",
        nargs="?",
        help="optional optimization CSV with columns step,label,qps,build_s,recall",
    )
    return parser.parse_args()


def load_rows(path):
    target = Path(path)
    if not target.is_file():
        raise SystemExit("error: CSV file not found: " + str(target))
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def find_column(fields, *names):
    for name in names:
        if name in fields:
            return name
    return None


def save(figure, output_dir, name):
    target = output_dir / name
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    print("wrote " + str(target))


def plot_recall_vs_qps(series, output_dir):
    figure, axes = plt.subplots(figsize=(8, 6))
    for system, points in sorted(series.items()):
        points = sorted(points)
        axes.plot([point[0] for point in points],
                  [point[1] for point in points],
                  marker="o",
                  label=system)
    axes.set_yscale("log")
    axes.set_xlabel("Recall@10")
    axes.set_ylabel("QPS (log scale)")
    axes.set_title("Recall@10 vs QPS by system")
    axes.grid(True, which="both", linestyle=":", alpha=0.6)
    axes.legend()
    save(figure, output_dir, "recall_vs_qps_all.png")


def plot_latencies(series, fields, output_dir):
    p50_col = find_column(fields, "p50_us", "p50")
    p95_col = find_column(fields, "p95_us", "p95")
    p99_col = find_column(fields, "p99_us", "p99")
    labels = []
    values = [[], [], []]
    for system, points in sorted(series.items()):
        candidates = []
        for recall, _qps, row in points:
            try:
                p50 = float(row[p50_col])
                p95 = float(row[p95_col])
                p99 = float(row[p99_col])
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append((abs(recall - 0.95), p50, p95, p99))
        if not candidates:
            continue
        candidates.sort(key=lambda candidate: candidate[0])
        labels.append(system)
        for index in range(3):
            values[index].append(candidates[0][index + 1])
    if not labels:
        print("skip latency_percentiles.png: no latency columns with plottable values")
        return
    x = np.arange(len(labels))
    width = 0.25
    figure, axes = plt.subplots(figsize=(max(7, len(labels) * 2), 5))
    axes.bar(x - width, values[0], width, label="p50")
    axes.bar(x, values[1], width, label="p95")
    axes.bar(x + width, values[2], width, label="p99")
    axes.set_xticks(x)
    axes.set_xticklabels(labels, rotation=20, ha="right")
    axes.set_ylabel("Latency (us)")
    axes.set_title("Latency percentiles at recall closest to 0.95")
    axes.grid(axis="y", linestyle=":", alpha=0.6)
    axes.legend()
    save(figure, output_dir, "latency_percentiles.png")


def plot_waterfall(path, output_dir):
    if path is None:
        print("skip opt_waterfall.png: no waterfall CSV supplied")
        return
    rows, fields = load_rows(path)
    qps_col = find_column(fields, "qps")
    step_col = find_column(fields, "step")
    label_col = find_column(fields, "label", "step")
    bars = []
    for row in rows:
        try:
            qps = float(row[qps_col])
        except (KeyError, TypeError, ValueError):
            continue
        step = str(row.get(step_col, ""))
        label = str(row.get(label_col, "")) or step
        bars.append((step, label, qps))
    bars.sort(key=lambda bar: (0, int(bar[0])) if bar[0].isdigit() else (1, bar[0]))
    if not bars:
        print("skip opt_waterfall.png: no plottable rows in " + str(path))
        return
    figure, axes = plt.subplots(figsize=(max(8, len(bars) * 1.5), 5))
    axes.bar([bar[1] for bar in bars], [bar[2] for bar in bars])
    axes.set_ylabel("QPS")
    axes.set_title("Optimization waterfall")
    axes.grid(axis="y", linestyle=":", alpha=0.6)
    axes.tick_params(axis="x", rotation=20)
    save(figure, output_dir, "opt_waterfall.png")


def plot_size_charts(rows, system_col, n_col, mem_col, build_col, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        try:
            n = float(row[n_col])
            mem = float(row[mem_col])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[row[system_col]].append((n, mem, row))
    if not grouped:
        print("skip build_time_vs_size.png and memory_per_1m.png: no plottable size rows")
        return

    if build_col is not None:
        figure, axes = plt.subplots(figsize=(8, 6))
        for system, points in sorted(grouped.items()):
            pairs = []
            for n, _mem, row in points:
                try:
                    pairs.append((n, float(row[build_col])))
                except (KeyError, TypeError, ValueError):
                    continue
            pairs.sort()
            if pairs:
                axes.plot([pair[0] for pair in pairs],
                          [pair[1] for pair in pairs],
                          marker="o",
                          label=system)
        axes.set_xscale("log")
        axes.set_xlabel("Vectors (n)")
        axes.set_ylabel("Build time (s)")
        axes.set_title("Build time vs index size")
        axes.grid(True, which="both", linestyle=":", alpha=0.6)
        axes.legend()
        save(figure, output_dir, "build_time_vs_size.png")
    else:
        print("skip build_time_vs_size.png: no build_s column")

    labels = []
    gigabytes = []
    for system, points in sorted(grouped.items()):
        n, mem, _row = max(points, key=lambda point: point[0])
        if n <= 0:
            continue
        labels.append(system)
        gigabytes.append(mem / n * 1_000_000 / (1024 ** 3))
    if not labels:
        print("skip memory_per_1m.png: no positive index sizes")
        return
    figure, axes = plt.subplots(figsize=(max(7, len(labels) * 2), 5))
    axes.bar(labels, gigabytes)
    axes.set_ylabel("Estimated index size (GiB per 1M vectors)")
    axes.set_title("Memory per 1M vectors")
    axes.grid(axis="y", linestyle=":", alpha=0.6)
    save(figure, output_dir, "memory_per_1m.png")


def main():
    args = parse_args()
    rows, fields = load_rows(args.compare_csv)
    system_col = find_column(fields, "system", "dataset", "backend")
    recall_col = find_column(fields, "recall", "recall@10", "recall_10")
    qps_col = find_column(fields, "qps")
    if system_col is None or recall_col is None or qps_col is None:
        raise SystemExit("error: compare CSV needs system, recall, and qps columns")

    series = defaultdict(list)
    for row in rows:
        try:
            recall = float(row[recall_col])
            qps = float(row[qps_col])
        except (KeyError, TypeError, ValueError):
            continue
        if qps <= 0:
            continue
        series[row[system_col]].append((recall, qps, row))
    if not series:
        raise SystemExit("error: no plottable rows in " + str(args.compare_csv))

    output_dir = Path(__file__).resolve().parent.parent / "docs" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_recall_vs_qps(series, output_dir)
    plot_latencies(series, fields, output_dir)
    plot_waterfall(args.waterfall_csv, output_dir)

    n_col = find_column(fields, "n")
    mem_col = find_column(fields, "index_bytes_est")
    build_col = find_column(fields, "build_s")
    if n_col is None or mem_col is None:
        print("skip build_time_vs_size.png and memory_per_1m.png: n or index_bytes_est column absent")
    else:
        plot_size_charts(rows, system_col, n_col, mem_col, build_col, output_dir)


if __name__ == "__main__":
    main()
