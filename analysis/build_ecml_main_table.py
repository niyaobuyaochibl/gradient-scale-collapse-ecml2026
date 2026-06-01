#!/usr/bin/env python3
"""Build unified ECML main table from custom runs and MMRec baselines."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RESULT_ROOT = Path("data/extended_fusion/results")
MMREC_ROOT = RESULT_ROOT / "mmrec"
OUT_DIR = Path("outputs/")

CUSTOM_RUNS: Dict[str, Dict[str, Path]] = {
    "MovieLens-1M": {
        "LightGCN-only": RESULT_ROOT / "movielens_lightgcn_only",
        "Fixed": RESULT_ROOT / "ml1m/fixed",
        "GBAF": RESULT_ROOT / "movielens_gbaf",
        "GBAF-Adaptive": RESULT_ROOT / "movielens_gbaf_adaptive",
    },
    "Amazon": {
        "LightGCN-only": RESULT_ROOT / "amazon_lightgcn_only",
        "Fixed": RESULT_ROOT / "amazonff/fixed",
        "GBAF": RESULT_ROOT / "amazon_gbaf",
        "GBAF-Adaptive": RESULT_ROOT / "amazon_gbaf_adaptive",
    },
    "Yelp": {
        "LightGCN-only": RESULT_ROOT / "yelp_lightgcn_only",
        "Fixed": RESULT_ROOT / "yelp/fixed",
        "GBAF": RESULT_ROOT / "yelp/gbaf",
        "GBAF-Adaptive": RESULT_ROOT / "yelp/gbaf_adaptive",
    },
    "MIND": {
        "LightGCN-only": RESULT_ROOT / "mind_lightgcn_only",
        "Fixed": RESULT_ROOT / "mind/fixed",
        "GBAF": RESULT_ROOT / "mind/gbaf",
        "GBAF-Adaptive": RESULT_ROOT / "mind/gbaf_adaptive",
    },
}

MMREC_DS = {
    "MovieLens-1M": "ecml_ml1m",
    "Amazon": "ecml_amazonff",
    "Yelp": "ecml_yelp",
    "MIND": "ecml_mind",
}
MMREC_MODELS = ["BM3", "LATTICE", "FREEDOM"]


def parse_custom(run_dir: Path) -> Tuple[int, float, float, float, float]:
    rs: List[float] = []
    ns: List[float] = []
    for p in sorted(run_dir.glob("results_seed*.json")):
        data = json.loads(p.read_text())
        rs.append(float(data.get("test_recall@10", data.get("best_val_recall@10", np.nan))))
        ns.append(float(data.get("test_ndcg@10", data.get("best_val_ndcg@10", np.nan))))
    if not rs:
        return 0, np.nan, np.nan, np.nan, np.nan
    return len(rs), float(np.mean(rs)), float(np.std(rs)), float(np.mean(ns)), float(np.std(ns))


def parse_mmrec(ds: str, model: str) -> Tuple[int, float, float, float, float]:
    rs: List[float] = []
    ns: List[float] = []
    log_paths = sorted((MMREC_ROOT / ds / model).glob("seed*/train.log"))
    for p in log_paths:
        t = p.read_text(errors="ignore")
        all_pairs = re.findall(r"best test:.*?recall@10:\s*([0-9.]+).*?ndcg@10:\s*([0-9.]+)", t, flags=re.I | re.S)
        if all_pairs:
            r, n = all_pairs[-1]
            rs.append(float(r))
            ns.append(float(n))
            continue
        m = re.search(r"BEST.*?Test:.*?recall@10:\s*([0-9.]+).*?ndcg@10:\s*([0-9.]+)", t, flags=re.S)
        if m:
            rs.append(float(m.group(1)))
            ns.append(float(m.group(2)))
    if not rs:
        return 0, np.nan, np.nan, np.nan, np.nan
    return len(rs), float(np.mean(rs)), float(np.std(rs)), float(np.mean(ns)), float(np.std(ns))


def fmt(mean: float, std: float, n: int) -> str:
    if np.isnan(mean):
        return "--"
    if n <= 1:
        return f"{mean:.4f}"
    return f"{mean:.4f}$\\pm${std:.4f}"


def main() -> None:
    rows = []
    for dataset, methods in CUSTOM_RUNS.items():
        for method, run_dir in methods.items():
            n, rm, rs, nm, ns = parse_custom(run_dir)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": n,
                    "recall_mean": rm,
                    "recall_std": rs,
                    "ndcg_mean": nm,
                    "ndcg_std": ns,
                    "source": "custom",
                }
            )
        ds_mmrec = MMREC_DS[dataset]
        for method in MMREC_MODELS:
            n, rm, rs, nm, ns = parse_mmrec(ds_mmrec, method)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": n,
                    "recall_mean": rm,
                    "recall_std": rs,
                    "ndcg_mean": nm,
                    "ndcg_std": ns,
                    "source": "mmrec",
                }
            )

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "main_results_unified.csv"
    df.to_csv(csv_path, index=False)

    # Wide table for LaTeX insertion.
    method_order = ["LightGCN-only", "Fixed", "GBAF", "GBAF-Adaptive", "BM3", "LATTICE", "FREEDOM"]
    datasets = ["MovieLens-1M", "Amazon", "Yelp", "MIND"]
    lines = []
    for method in method_order:
        row = [method]
        for ds in datasets:
            rec = df[(df["dataset"] == ds) & (df["method"] == method)]
            if rec.empty:
                row.extend(["--", "--"])
                continue
            r = rec.iloc[0]
            row.append(fmt(float(r["recall_mean"]), float(r["recall_std"]), int(r["n"])))
            row.append(fmt(float(r["ndcg_mean"]), float(r["ndcg_std"]), int(r["n"])))
        lines.append(row)

    table_df = pd.DataFrame(
        lines,
        columns=[
            "Method",
            "ML-1M R@10",
            "ML-1M N@10",
            "Amazon R@10",
            "Amazon N@10",
            "Yelp R@10",
            "Yelp N@10",
            "MIND R@10",
            "MIND N@10",
        ],
    )
    tex_path = OUT_DIR / "main_results_unified.tex"
    tex_path.write_text(table_df.to_latex(index=False, escape=False), encoding="utf-8")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {tex_path}")


if __name__ == "__main__":
    main()
