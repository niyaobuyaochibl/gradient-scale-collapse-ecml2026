#!/usr/bin/env python3
"""Analyze per-epoch correlation between gradient ratio r_t and validation Recall@10.

Produces:
  1. A 2x2 panel figure: per-epoch r_t (left y) and val recall (right y) for 4 datasets
  2. A scatter figure: epoch-level r_t vs val_recall across all datasets/seeds
  3. Summary CSV with Pearson/Spearman correlations per dataset
"""

from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Log directories
LOG_DIRS = {
    "ML-1M": {
        "gbaf": list(Path("data/extended_fusion/logs").glob("*/movielens_gbaf_seed*.log")) +
                list(Path("data/extended_fusion/logs").glob("*/ml1m_gbaf_seed*.log")),
        "fixed": list(Path("data/extended_fusion/logs").glob("*/movielens_fixed*_seed*.log")) +
                 list(Path("data/extended_fusion/logs").glob("*/ml1m_fixed*_seed*.log")),
    },
    "Amazon": {
        "gbaf": list(Path("data/extended_fusion/logs").glob("*/amazon*gbaf_seed*.log")),
        "fixed": list(Path("data/extended_fusion/logs").glob("*/amazon*fixed*_seed*.log")),
    },
    "Yelp": {
        "gbaf": list(Path("data/extended_fusion/logs").glob("*/yelp_gbaf_seed*.log")),
        "fixed": list(Path("data/extended_fusion/logs").glob("*/yelp_fixed*_seed*.log")),
    },
    "MIND": {
        "gbaf": list(Path("data/extended_fusion/logs").glob("*/mind_gbaf_seed*.log")),
        "fixed": list(Path("data/extended_fusion/logs").glob("*/mind_fixed*_seed*.log")),
    },
}

FIG_DIR = Path("outputs/")
TABLE_DIR = Path("outputs/")

RATIO_RE = re.compile(r"ratio_text_cf=([0-9.]+)")
EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
VAL_RECALL_RE = re.compile(r"Val Recall@10:\s*([0-9.]+)")


def parse_log(path: Path) -> Tuple[List[int], List[float], List[float]]:
    """Parse a training log and extract per-epoch r_t and val recall."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    
    epoch_ratios: Dict[int, List[float]] = {}
    epoch_recalls: Dict[int, float] = {}
    current_epoch = 0
    
    for line in text.splitlines():
        em = EPOCH_RE.search(line)
        if em:
            current_epoch = int(em.group(1))
            continue
        
        rm = RATIO_RE.search(line)
        if rm and current_epoch > 0:
            epoch_ratios.setdefault(current_epoch, []).append(float(rm.group(1)))
        
        vm = VAL_RECALL_RE.search(line)
        if vm and current_epoch > 0:
            epoch_recalls[current_epoch] = float(vm.group(1))
    
    # Average ratios per epoch, align with recalls
    epochs = sorted(set(epoch_ratios.keys()) & set(epoch_recalls.keys()))
    rt_vals = [np.mean(epoch_ratios[e]) for e in epochs]
    rc_vals = [epoch_recalls[e] for e in epochs]
    
    return epochs, rt_vals, rc_vals


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    # ======== Figure 1: Dual-axis time series (1x2, Yelp + MIND only) ========
    plot_datasets = ["Yelp", "MIND"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    
    for idx, ds_name in enumerate(plot_datasets):
        method_logs = LOG_DIRS[ds_name]
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Plot GBAF logs
        gbaf_logs = method_logs.get("gbaf", [])
        for lf in gbaf_logs[:5]:  # max 5 seeds
            epochs, rts, rcs = parse_log(lf)
            if not epochs:
                continue
            seed = re.search(r"seed(\d+)", str(lf))
            seed_str = seed.group(1) if seed else "?"
            alpha = 0.4 if len(gbaf_logs) > 1 else 0.8
            ax.plot(epochs, rts, color="tab:red", alpha=alpha, linewidth=0.8)
            ax2.plot(epochs, rcs, color="tab:blue", alpha=alpha, linewidth=0.8)
            
            # Collect for correlation
            for e, r, rc in zip(epochs, rts, rcs):
                all_records.append({
                    "dataset": ds_name, "method": "GBAF", "seed": seed_str,
                    "epoch": e, "r_t": r, "val_recall": rc,
                })
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$r_t$ (gradient ratio)", color="tab:red")
        ax2.set_ylabel("Val Recall@10", color="tab:blue")
        ax.set_title(ds_name)
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.grid(axis='x', alpha=0.2)
    
    # Also collect data from non-plotted datasets for correlation
    for ds_name, method_logs in LOG_DIRS.items():
        if ds_name in plot_datasets:
            continue
        gbaf_logs = method_logs.get("gbaf", [])
        for lf in gbaf_logs[:5]:
            epochs, rts, rcs = parse_log(lf)
            if not epochs:
                continue
            seed = re.search(r"seed(\d+)", str(lf))
            seed_str = seed.group(1) if seed else "?"
            for e, r, rc in zip(epochs, rts, rcs):
                all_records.append({
                    "dataset": ds_name, "method": "GBAF", "seed": seed_str,
                    "epoch": e, "r_t": r, "val_recall": rc,
                })

    fig.suptitle("Per-Epoch Gradient Ratio $r_t$ and Validation Recall@10 (GBAF)", y=1.02)
    fig.tight_layout()
    out1 = FIG_DIR / "rt_recall_timeseries.pdf"
    fig.savefig(out1)
    fig.savefig(FIG_DIR / "rt_recall_timeseries.png", dpi=220)
    print(f"[OK] wrote {out1}")
    plt.close(fig)
    
    # ======== Correlation analysis ========
    summary_rows = []
    
    for ds_name in LOG_DIRS:
        ds_data = [r for r in all_records if r["dataset"] == ds_name]
        if len(ds_data) < 5:
            continue
        rts = [r["r_t"] for r in ds_data]
        rcs = [r["val_recall"] for r in ds_data]
        
        # Deviation from balance |r_t - 1|
        devs = [abs(r - 1.0) for r in rts]
        
        pearson_r, pearson_p = stats.pearsonr(devs, rcs)
        spearman_r, spearman_p = stats.spearmanr(devs, rcs)
        
        summary_rows.append({
            "dataset": ds_name,
            "n_points": len(ds_data),
            "mean_rt": np.mean(rts),
            "std_rt": np.std(rts),
            "pearson_r_dev_vs_recall": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r_dev_vs_recall": spearman_r,
            "spearman_p": spearman_p,
        })
        
        print(f"  {ds_name}: n={len(ds_data)}, Pearson(|r_t-1|, recall)={pearson_r:.3f} (p={pearson_p:.4f}), "
              f"Spearman={spearman_r:.3f} (p={spearman_p:.4f})")
    
    # Write summary CSV
    csv_path = TABLE_DIR / "rt_recall_correlation.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[OK] wrote {csv_path}")


if __name__ == "__main__":
    main()
