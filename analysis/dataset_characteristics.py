"""Compute dataset-level characteristics to support analysis in the paper.

Outputs:
  - tables/dataset_characteristics.csv
  - tables/dataset_characteristics.tex
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Reuse training utilities
import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_fusion import load_data  # type: ignore  # noqa: E402
from models import compute_item_recency  # type: ignore  # noqa: E402
import yaml  # noqa: E402


CONFIG_MAP: Dict[str, str] = {
    "ml1m": str(PROJECT_ROOT / "configs/extended/movielens_fixed_baseline.yaml"),
    "amazonff": str(PROJECT_ROOT / "configs/extended/amazon_fixed_baseline.yaml"),
    "yelp": str(PROJECT_ROOT / "configs/extended/yelp_fixed_baseline.yaml"),
    "mind": str(PROJECT_ROOT / "configs/extended/mind_fixed_baseline.yaml"),
}


def _to_pairs(data) -> np.ndarray:
    if hasattr(data, "values"):
        arr = data[["user_idx", "item_idx"]].values  # type: ignore[attr-defined]
        return arr.astype(np.int64, copy=False)
    # assume array-like of (user_id, item_id)
    return np.asarray(data, dtype=np.int64)


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)


def top_k_share(x: np.ndarray, frac: float) -> float:
    if x.size == 0:
        return float("nan")
    x = np.sort(x)[::-1]
    k = max(1, int(math.ceil(frac * x.size)))
    return float(np.sum(x[:k]) / np.sum(x))


def compute_characteristics(dataset: str) -> Dict[str, float]:
    cfg_path = Path(CONFIG_MAP[dataset])
    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)
    config_dir = cfg_path.parent

    train_data, val_data, test_data, item_embeddings, stats = load_data(
        config["data"]["data_dir"], config, config_dir
    )

    pairs = _to_pairs(train_data)
    user_ids = pairs[:, 0]
    item_ids = pairs[:, 1]

    n_train = pairs.shape[0]
    n_users = int(stats.get("n_users", len(np.unique(user_ids))))
    n_items = int(stats.get("n_items", len(np.unique(item_ids))))

    # Per-user activity statistics
    user_counts_full = np.bincount(user_ids, minlength=n_users)
    active_user_counts = user_counts_full[user_counts_full > 0]
    user_mean = float(np.mean(active_user_counts)) if active_user_counts.size else float("nan")
    user_std = float(np.std(active_user_counts, ddof=1)) if active_user_counts.size > 1 else float("nan")
    user_gini = gini_coefficient(active_user_counts) if active_user_counts.size else float("nan")

    # Item popularity statistics
    item_counts_full = np.bincount(item_ids, minlength=n_items)
    active_mask = item_counts_full > 0
    active_item_counts = item_counts_full[active_mask]
    active_items = int(active_item_counts.size)
    item_gini = gini_coefficient(active_item_counts) if active_item_counts.size else float("nan")
    item_top1_share = top_k_share(active_item_counts, 0.01) if active_item_counts.size else float("nan")

    pop_entropy = float("nan")
    if active_item_counts.size:
        probs = active_item_counts.astype(np.float64) / active_item_counts.sum()
        entropy = -(probs * np.log(probs + 1e-12)).sum()
        pop_entropy = float(entropy / (math.log(active_item_counts.size) + 1e-12))

    coverage = active_items / n_items if n_items > 0 else float("nan")

    # Text embedding statistics
    embeddings: List[np.ndarray] = []
    if hasattr(item_embeddings, "items"):
        iterator = item_embeddings.items()
    elif isinstance(item_embeddings, dict):
        iterator = item_embeddings.items()
    else:
        iterator = enumerate(item_embeddings)

    for _, vec in iterator:
        if vec is None:
            continue
        arr = vec.detach().cpu().numpy() if hasattr(vec, "detach") else np.asarray(vec)
        if arr.size == 0:
            continue
        embeddings.append(arr.astype(np.float32, copy=False))

    norm_mean = float("nan")
    norm_std = float("nan")
    norm_snr = float("nan")
    text_sep = float("nan")
    if embeddings:
        mat_full = np.stack(embeddings, axis=0)
        norms = np.linalg.norm(mat_full, axis=1)
        norm_mean = float(np.mean(norms))
        norm_std = float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0
        mean_vec = mat_full.mean(axis=0)
        signal_power = float(np.dot(mean_vec, mean_vec))
        noise_power = float(np.var(mat_full, axis=0).mean())
        norm_snr = float(signal_power / (noise_power + 1e-8)) if noise_power > 0 else float("inf")

        sample_size = min(len(mat_full), 4000)
        rng = np.random.default_rng(0)
        if sample_size < len(mat_full):
            idx = rng.choice(len(mat_full), size=sample_size, replace=False)
            mat = mat_full[idx]
        else:
            mat = mat_full
        norms_mat = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        normed = mat / norms_mat
        cosine = normed @ normed.T
        m = cosine.shape[0]
        if m > 1:
            tri = cosine[np.triu_indices(m, k=1)]
            text_sep = float(1.0 - tri.mean()) if tri.size else 0.0
        else:
            text_sep = 0.0

    recency = compute_item_recency(train_data, n_items).numpy()
    recency_active = recency[active_mask]
    recency_mean = float(recency_active.mean()) if recency_active.size else float("nan")
    recency_std = float(recency_active.std(ddof=1)) if recency_active.size > 1 else float("nan")

    return {
        "dataset": dataset,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "user_mean_interactions": user_mean,
        "user_std_interactions": user_std,
        "user_gini": user_gini,
        "item_active": active_items,
        "item_coverage": coverage,
        "item_gini": item_gini,
        "item_top1pct_share": item_top1_share,
        "item_entropy": pop_entropy,
        "embed_norm_mean": norm_mean,
        "embed_norm_std": norm_std,
        "embed_norm_snr": norm_snr,
        "embed_text_separability": text_sep,
        "recency_mean": recency_mean,
        "recency_std": recency_std,
    }


def write_outputs(rows: Iterable[Dict[str, float]]) -> Tuple[Path, Path]:
    import csv

    table_dir = PROJECT_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / "dataset_characteristics.csv"
    fields = [
        "dataset",
        "n_users",
        "n_items",
        "n_train",
        "user_mean_interactions",
        "user_std_interactions",
        "user_gini",
        "item_active",
        "item_coverage",
        "item_gini",
        "item_top1pct_share",
        "item_entropy",
        "embed_norm_mean",
        "embed_norm_std",
        "embed_norm_snr",
        "embed_text_separability",
        "recency_mean",
        "recency_std",
    ]
    rows = list(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # LaTeX (compact)
    tex_path = table_dir / "dataset_characteristics.tex"
    def fmt(x: float, pct: bool = False) -> str:
        if x != x or x is None:  # NaN
            return "--"
        return f"{(x*100.0):.2f}\\%" if pct else f"{x:.2f}"

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Dataset characteristics (train split). Pop. Entropy denotes normalized item-popularity entropy; Text Sep.= $1-\\bar{\\cos}$ over sampled pairs.}",
        "  \\label{tab:dataset-characteristics}",
        "  \\begin{tabular}{lrrrrrrrrrr}",
        "    \\toprule",
        "    Dataset & Users & Items & Train & User Gini & Item Gini & Pop Ent. & Top 1\\% share & Text SNR & Text Sep. & Recency $\\mu$ " + "\\\\",
        "    \\midrule",
    ]
    for r in rows:
        line = (
            f"    {r['dataset']} & {r['n_users']} & {r['n_items']} & {r['n_train']} & "
            f"{fmt(r['user_gini'])} & {fmt(r['item_gini'])} & {fmt(r['item_entropy'])} & "
            f"{fmt(r['item_top1pct_share'], pct=True)} & {fmt(r['embed_norm_snr'])} & "
            f"{fmt(r['embed_text_separability'])} & {fmt(r['recency_mean'])} "
        ) + "\\\\"
        lines.append(line)
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    return csv_path, tex_path


def main() -> None:
    datasets = ["ml1m", "amazonff", "yelp", "mind"]
    rows = [compute_characteristics(ds) for ds in datasets]
    csv_path, tex_path = write_outputs(rows)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote LaTeX: {tex_path}")


if __name__ == "__main__":
    main()


