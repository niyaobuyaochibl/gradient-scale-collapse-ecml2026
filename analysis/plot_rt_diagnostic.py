#!/usr/bin/env python3
"""Plot r_t deviation vs fixed-fusion degradation/recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_ROOT = Path("data/extended_fusion/results")
RATIO_CSV = Path("outputs/")
FIG_DIR = Path("outputs/")
TABLE_DIR = Path("outputs/")


def mean_recall(run_dir: Path) -> float:
    vals = []
    for p in sorted(run_dir.glob("results_seed*.json")):
        d = json.loads(p.read_text())
        vals.append(float(d.get("test_recall@10", d.get("best_val_recall@10", np.nan))))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ratio = pd.read_csv(RATIO_CSV)
    rt_map: Dict[str, float] = {}
    for _, r in ratio.iterrows():
        series = str(r["series"])
        if "(ML-1M)" in series:
            rt_map["MovieLens-1M"] = float(r["mean_abs_dev_from_1"])
        elif "(Amazon)" in series:
            rt_map["Amazon"] = float(r["mean_abs_dev_from_1"])
        elif "(Yelp)" in series:
            rt_map["Yelp"] = float(r["mean_abs_dev_from_1"])
        elif "(MIND)" in series:
            rt_map["MIND"] = float(r["mean_abs_dev_from_1"])

    runs: Dict[str, Tuple[Path, Path, Path]] = {
        "MovieLens-1M": (RESULT_ROOT / "movielens_lightgcn_only", RESULT_ROOT / "ml1m/fixed", RESULT_ROOT / "movielens_gbaf"),
        "Amazon": (RESULT_ROOT / "amazon_lightgcn_only", RESULT_ROOT / "amazonff/fixed", RESULT_ROOT / "amazon_gbaf"),
        "Yelp": (RESULT_ROOT / "yelp_lightgcn_only", RESULT_ROOT / "yelp/fixed", RESULT_ROOT / "yelp/gbaf"),
        "MIND": (RESULT_ROOT / "mind_lightgcn_only", RESULT_ROOT / "mind/fixed", RESULT_ROOT / "mind/gbaf"),
    }

    rows = []
    for ds, (cf_dir, fixed_dir, gbaf_dir) in runs.items():
        cf = mean_recall(cf_dir)
        fixed = mean_recall(fixed_dir)
        gbaf = mean_recall(gbaf_dir)
        rows.append(
            {
                "dataset": ds,
                "rt_dev": rt_map.get(ds, np.nan),
                "fixed_vs_cf_pct": (fixed - cf) / cf * 100.0 if cf > 0 else np.nan,
                "gbaf_vs_fixed_pct": (gbaf - fixed) / fixed * 100.0 if fixed > 0 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLE_DIR / "rt_diagnostic_points.csv"
    df.to_csv(out_csv, index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, r in df.iterrows():
        ax.scatter(r["rt_dev"], r["fixed_vs_cf_pct"], s=70, label=r["dataset"])
        ax.annotate(r["dataset"], (r["rt_dev"], r["fixed_vs_cf_pct"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    ax.set_xlabel(r"Mean |$r_t - 1$|")
    ax.set_ylabel("Fixed vs CF-only Recall@10 (%)")
    ax.set_title("Diagnostic relation: gradient imbalance vs fusion risk")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    pdf = FIG_DIR / "rt_diagnostic_scatter.pdf"
    png = FIG_DIR / "rt_diagnostic_scatter.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {pdf}")
    print(f"[OK] wrote {png}")


if __name__ == "__main__":
    main()
