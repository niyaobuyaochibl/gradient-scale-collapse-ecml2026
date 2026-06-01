#!/usr/bin/env python3
"""Plot gradient-ratio dynamics and summary statistics from training logs."""

from __future__ import annotations

import csv
import re
from statistics import mean, pstdev
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RATIO_LINE = re.compile(r"ratio_text_cf=([0-9.]+)")
EPOCH_LINE = re.compile(r"Epoch \[(\d+)/(\d+)\]")


def parse_log(path: Path) -> List[Tuple[int, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pairs: List[Tuple[int, float]] = []
    current_epoch = None
    for line in text.splitlines():
        epoch_match = EPOCH_LINE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        ratio_match = RATIO_LINE.search(line)
        if ratio_match and current_epoch is not None:
            pairs.append((current_epoch, float(ratio_match.group(1))))
    # deduplicate per epoch (keep latest)
    merged: Dict[int, float] = {}
    for e, r in pairs:
        merged[e] = r
    return sorted(merged.items(), key=lambda x: x[0])


def first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def main() -> None:
    # Prefer latest 60-epoch run logs where available.
    logs: Dict[str, Path] = {}
    ml1m = first_existing(
        [
            Path("data/extended_fusion/logs/route_c_core/ml1m/gbaf/seed42.log"),
        ]
    )
    amazon = first_existing(
        [
            Path("data/extended_fusion/logs/route_c_core/amazonff/gbaf/seed42.log"),
        ]
    )
    yelp = first_existing(
        [
            Path("data/extended_fusion/logs/yelp_mind_60ep_parallel/yelp_gbaf_seed42.log"),
            Path("data/extended_fusion/logs/yelp_gbaf_probe_seed42.log"),
        ]
    )
    mind = first_existing(
        [
            Path("data/extended_fusion/logs/yelp_mind_60ep_parallel/mind_gbaf_seed42.log"),
            Path("data/extended_fusion/logs/yelp_mind_60ep_parallel/mind_gbaf_seed123.log"),
        ]
    )
    if ml1m:
        logs["GBAF (ML-1M)"] = ml1m
    if amazon:
        logs["GBAF (Amazon)"] = amazon
    if yelp:
        logs["GBAF (Yelp)"] = yelp
    if mind:
        logs["GBAF (MIND)"] = mind
    out_dir = Path("outputs/")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.4, 4.2))
    stats_rows: List[Dict[str, str]] = []
    for label, path in logs.items():
        if not path.exists():
            continue
        series = parse_log(path)
        if not series:
            continue
        x = [e for e, _ in series]
        y = [r for _, r in series]
        plt.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=label)
        stats_rows.append(
            {
                "series": label,
                "log_path": str(path),
                "epochs": str(len(series)),
                "ratio_mean": f"{mean(y):.6f}",
                "ratio_std": f"{pstdev(y):.6f}",
                "ratio_min": f"{min(y):.6f}",
                "ratio_max": f"{max(y):.6f}",
                "mean_abs_dev_from_1": f"{mean([abs(v - 1.0) for v in y]):.6f}",
            }
        )
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="gray", label="Target ratio")
    plt.xlabel("Epoch")
    plt.ylabel(r"$r_t=\|g_{text}\|_2/(\|g_{CF}\|_2+\epsilon)$")
    plt.title("Branch Gradient-Ratio Dynamics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "gradient_ratio_dynamics.pdf")
    plt.savefig(out_dir / "gradient_ratio_dynamics.png", dpi=240)
    summary_csv = out_dir / "gradient_ratio_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "log_path",
                "epochs",
                "ratio_mean",
                "ratio_std",
                "ratio_min",
                "ratio_max",
                "mean_abs_dev_from_1",
            ],
        )
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)
    print(f"[OK] wrote {out_dir / 'gradient_ratio_dynamics.pdf'}")
    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()
