"""Paired significance tests (t-test) and effect sizes across seeds.

Compares selected variant pairs per dataset using results_seed*.json files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


RESULTS_ROOT = Path("data/extended_fusion/results")

PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "ml1m": [
        ("fixed", "attention_pop"),
        ("fixed", "attention_concat"),
        ("fixed", "attention_pop_user"),
        ("attention_pop", "attention_pop_user"),
    ],
    "amazonff": [
        ("fixed", "attention_pop"),
        ("fixed", "attention_concat"),
        ("fixed", "attention_pop_user"),
        ("attention_pop", "attention_pop_user"),
    ],
    "yelp": [
        ("fixed", "attention_pop"),
        ("fixed", "attention_concat"),
        ("fixed", "attention_pop_user"),
        ("attention_pop", "attention_pop_user"),
    ],
    "mind": [
        ("fixed", "attention_pop"),
        ("fixed", "attention_concat"),
        ("attention_pop", "attention_concat"),
        ("fixed", "attention_pop_user"),
        ("attention_pop", "attention_pop_user"),
    ],
}


def load_metrics(dataset: str, variant: str) -> Dict[int, Dict[str, float]]:
    root = RESULTS_ROOT / dataset / variant
    out: Dict[int, Dict[str, float]] = {}
    for path in sorted(root.glob("results_seed*.json")):
        data = json.loads(path.read_text())
        seed = int(data.get("seed"))
        out[seed] = {"recall": float(data.get("recall@10", np.nan)), "ndcg": float(data.get("ndcg@10", np.nan))}
    return out


def paired_test(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    d = a_arr - b_arr
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr, nan_policy="omit")
    cohen_d = float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan")
    return float(t_stat), float(p_val), cohen_d


def analyze(dataset: str, pairs: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for va, vb in pairs:
        A = load_metrics(dataset, va)
        B = load_metrics(dataset, vb)
        common_seeds = sorted(set(A.keys()) & set(B.keys()))
        if not common_seeds:
            continue
        a_rec = [A[s]["recall"] for s in common_seeds]
        b_rec = [B[s]["recall"] for s in common_seeds]
        a_ndcg = [A[s]["ndcg"] for s in common_seeds]
        b_ndcg = [B[s]["ndcg"] for s in common_seeds]
        tr, pr, dr = paired_test(a_rec, b_rec)
        tn, pn, dn = paired_test(a_ndcg, b_ndcg)
        rows.append({
            "dataset": dataset,
            "A": va,
            "B": vb,
            "seeds": len(common_seeds),
            "recall_t": tr,
            "recall_p": pr,
            "recall_d": dr,
            "ndcg_t": tn,
            "ndcg_p": pn,
            "ndcg_d": dn,
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired t-tests for variant differences")
    parser.add_argument("--datasets", nargs="*", default=["ml1m", "amazonff", "yelp", "mind"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for ds in args.datasets:
        frames.append(analyze(ds, PAIRS.get(ds, [])))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_dir = RESULTS_ROOT.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "significance_summary.md"
    md.write_text("\n".join(["# Paired Significance Tests", "", df.to_markdown(index=False)]), encoding="utf-8")
    print(f"✅ Wrote {md}")


if __name__ == "__main__":
    main()



