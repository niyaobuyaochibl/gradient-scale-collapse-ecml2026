#!/usr/bin/env python3
"""Paired significance tests for ECML paper comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

RESULT_ROOT = Path("data/extended_fusion/results")
OUT_DIR = Path("outputs/")

RUNS: Dict[str, Dict[str, Path]] = {
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

PAIRS = [("GBAF", "Fixed"), ("GBAF", "LightGCN-only"), ("GBAF-Adaptive", "Fixed")]


def load_by_seed(run_dir: Path) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for p in sorted(run_dir.glob("results_seed*.json")):
        data = json.loads(p.read_text())
        seed = int(data.get("seed"))
        out[seed] = (
            float(data.get("test_recall@10", data.get("best_val_recall@10", np.nan))),
            float(data.get("test_ndcg@10", data.get("best_val_ndcg@10", np.nan))),
        )
    return out


def paired(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    t, p = stats.ttest_rel(aa, bb, nan_policy="omit")
    d = aa - bb
    eff = float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan")
    return float(t), float(p), eff


def main() -> None:
    rows = []
    for ds, methods in RUNS.items():
        cache = {m: load_by_seed(path) for m, path in methods.items()}
        for a, b in PAIRS:
            A = cache[a]
            B = cache[b]
            common = sorted(set(A.keys()) & set(B.keys()))
            if len(common) < 2:
                continue
            rec_a = [A[s][0] for s in common]
            rec_b = [B[s][0] for s in common]
            nd_a = [A[s][1] for s in common]
            nd_b = [B[s][1] for s in common]
            tr, pr, dr = paired(rec_a, rec_b)
            tn, pn, dn = paired(nd_a, nd_b)
            rows.append(
                {
                    "dataset": ds,
                    "A": a,
                    "B": b,
                    "n_seeds": len(common),
                    "recall_t": tr,
                    "recall_p": pr,
                    "recall_cohen_d": dr,
                    "ndcg_t": tn,
                    "ndcg_p": pn,
                    "ndcg_cohen_d": dn,
                }
            )

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "significance_ecml.csv"
    md_path = OUT_DIR / "significance_ecml.md"
    df.to_csv(csv_path, index=False)
    lines = ["# ECML Significance Tests", ""]
    if df.empty:
        lines.append("No valid paired comparisons found.")
    else:
        lines.append(df.to_string(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
