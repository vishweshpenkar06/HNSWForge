import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 scripts/plot_results.py <sweep-results.csv>")
    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        raise SystemExit("error: results file not found: " + str(csv_path))

    combos = defaultdict(list)
    dataset: str = ""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            qps = float(row["qps"])
            recall = float(row["recall"])
            if qps > 0:
                combos[(int(row["m"]), int(row["efc"]))].append((recall, qps))

    if not combos:
        raise SystemExit("error: no plottable rows in " + str(csv_path))

    output_dir = Path(__file__).resolve().parent.parent / "docs" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("recall_vs_qps_" + dataset + ".png")

    figure, axes = plt.subplots(figsize=(8, 6))
    for (m, efc), points in sorted(combos.items()):
        points.sort()
        axes.plot([point[0] for point in points],
                  [point[1] for point in points],
                  marker="o",
                  label="m=%d, efc=%d" % (m, efc))
    axes.set_yscale("log")
    axes.set_xlabel("Recall")
    axes.set_ylabel("QPS (log scale)")
    axes.set_title("Recall vs QPS (" + dataset + ")")
    axes.grid(True, which="both", linestyle=":", alpha=0.6)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print("wrote " + str(output))


if __name__ == "__main__":
    main()
