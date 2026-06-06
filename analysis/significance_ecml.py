#!/usr/bin/env python3
"""Recompute paired five-seed tests for the camera-ready comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

RUN_DIRS = {
    "MovieLens-1M": {"BPR-MF": "movielens_lightgcn_only", "Fixed": "ml1m/fixed", "GBAF": "movielens_gbaf"},
    "Amazon Fine Food": {"BPR-MF": "amazon_lightgcn_only", "Fixed": "amazonff/fixed", "GBAF": "amazon_gbaf"},
    "Amazon Books": {"BPR-MF": "amazon_books_lightgcn_only", "Fixed": "amazon_books_fixed_baseline", "GBAF": "amazon_books_gbaf"},
    "Amazon CDs": {"BPR-MF": "amazon_cds_lightgcn_only", "Fixed": "amazon_cds_fixed_baseline", "GBAF": "amazon_cds_gbaf"},
    "Yelp": {"BPR-MF": "yelp_lightgcn_only", "Fixed": "yelp/fixed", "GBAF": "yelp/gbaf"},
    "MIND": {"BPR-MF": "mind_lightgcn_only", "Fixed": "mind/fixed", "GBAF": "mind/gbaf"},
}


def load_recall_by_seed(run_dir: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    for path in sorted(run_dir.glob("results_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        values[seed] = float(payload.get("test_recall@10", payload["recall@10"]))
    return values


def paired(a: list[float], b: list[float]) -> tuple[float, float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    t_stat, p_value = stats.ttest_rel(aa, bb)
    differences = aa - bb
    effect = differences.mean() / differences.std(ddof=1)
    return float(t_stat), float(p_value), float(effect)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True, help="Directory containing per-seed result folders")
    parser.add_argument("--output", type=Path, default=Path("outputs/significance_ecml.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset, methods in RUN_DIRS.items():
        loaded = {name: load_recall_by_seed(args.result_root / rel) for name, rel in methods.items()}
        for baseline in ("Fixed", "BPR-MF"):
            common = sorted(set(loaded["GBAF"]) & set(loaded[baseline]))
            if len(common) != 5:
                raise RuntimeError(f"{dataset}: expected 5 paired seeds for GBAF vs {baseline}, found {len(common)}")
            t_stat, p_value, effect = paired(
                [loaded["GBAF"][seed] for seed in common],
                [loaded[baseline][seed] for seed in common],
            )
            rows.append({
                "dataset": dataset,
                "comparison": f"GBAF vs {baseline}",
                "n_seeds": len(common),
                "recall_t": t_stat,
                "recall_p": p_value,
                "recall_cohen_d": effect,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
