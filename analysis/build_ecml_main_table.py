#!/usr/bin/env python3
"""Export the verified controlled camera-ready result table to LaTeX."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATASETS = ["ML-1M", "Amazon Fine Food", "Amazon Books", "Amazon CDs", "Yelp", "MIND"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/tables/camera_ready_main_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/extended_fusion_summary.tex"),
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled full-ranking performance with the same BPR-MF backbone and frozen text features.}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Metric & ML-1M & Amazon FF & Books & CDs & Yelp & MIND \\",
        r"\midrule",
    ]
    previous_metric = None
    for row in rows:
        if previous_metric is not None and row["metric"] != previous_metric:
            lines.append(r"\midrule")
        values = [row[dataset] or "---" for dataset in DATASETS]
        lines.append(f'{row["method"]} & {row["metric"]} & ' + " & ".join(values) + r" \\")
        previous_metric = row["metric"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
