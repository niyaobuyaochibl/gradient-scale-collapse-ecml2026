"""Fit simple diagnostic boundaries relating dataset statistics to fusion gains."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = Path("data/extended_fusion/results")
DATASET_FEATURE_CSV = PROJECT_ROOT / "tables" / "dataset_characteristics.csv"

DATASETS = ["ml1m", "amazonff", "yelp", "mind"]
VARIANTS = [
    "fixed",
    "attention_pop",
    "attention_ctx",
    "attention_ctx_lora",
    "concat_mlp",
]


def load_dataset_features() -> Dict[str, Dict[str, float]]:
    features: Dict[str, Dict[str, float]] = {}
    with DATASET_FEATURE_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            feats = {k: float(row[k]) for k in row if k != "dataset"}
            features[dataset] = feats
    return features


def load_variant_metrics(dataset: str, variant: str) -> Tuple[float, float]:
    variant_dir = RESULT_ROOT / dataset / variant
    recall_vals: List[float] = []
    ndcg_vals: List[float] = []
    if not variant_dir.exists():
        return float("nan"), float("nan")
    for file in variant_dir.glob("results_seed*.json"):
        with file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        recall_vals.append(float(payload.get("recall@10", float("nan"))))
        ndcg_vals.append(float(payload.get("ndcg@10", float("nan"))))
    if not recall_vals:
        return float("nan"), float("nan")
    return float(np.nanmean(recall_vals)), float(np.nanmean(ndcg_vals))


def collect_metric_table() -> Dict[str, Dict[str, Dict[str, float]]]:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in DATASETS:
        table[dataset] = {}
        for variant in VARIANTS:
            recall, ndcg = load_variant_metrics(dataset, variant)
            table[dataset][variant] = {"recall@10": recall, "ndcg@10": ndcg}
    return table


def fit_linear_boundary(
    x_matrix: np.ndarray,
    y_vector: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return coefficients (including intercept) and R^2."""

    A = np.concatenate([np.ones((x_matrix.shape[0], 1)), x_matrix], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, y_vector, rcond=None)
    preds = A @ coeffs
    ss_res = float(np.sum((y_vector - preds) ** 2))
    ss_tot = float(np.sum((y_vector - y_vector.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return coeffs, r2


def main() -> None:
    dataset_features = load_dataset_features()
    metric_table = collect_metric_table()

    feature_names = [
        "item_entropy",
        "item_top1pct_share",
        "item_gini",
        "user_gini",
        "embed_text_separability",
        "recency_mean",
    ]

    rows: List[Dict[str, float]] = []
    x_rows: List[List[float]] = []
    y_recall: List[float] = []
    y_ndcg: List[float] = []

    for dataset in DATASETS:
        feats = dataset_features[dataset]
        fixed_metrics = metric_table[dataset].get("fixed", {})
        ctx_metrics = metric_table[dataset].get("attention_ctx", {})
        ctx_lora_metrics = metric_table[dataset].get("attention_ctx_lora", {})

        delta_recall = ctx_metrics["recall@10"] - fixed_metrics["recall@10"]
        delta_ndcg = ctx_metrics["ndcg@10"] - fixed_metrics["ndcg@10"]
        delta_recall_lora = ctx_lora_metrics["recall@10"] - fixed_metrics["recall@10"]
        delta_ndcg_lora = ctx_lora_metrics["ndcg@10"] - fixed_metrics["ndcg@10"]

        feature_vector = [float(feats.get(name, float("nan"))) for name in feature_names]

        row = {
            "dataset": dataset,
            "delta_recall_ctx_fixed": delta_recall,
            "delta_ndcg_ctx_fixed": delta_ndcg,
            "delta_recall_ctx_lora_fixed": delta_recall_lora,
            "delta_ndcg_ctx_lora_fixed": delta_ndcg_lora,
            **{f"feat_{name}": feature_vector[idx] for idx, name in enumerate(feature_names)},
        }
        rows.append(row)

        if not (np.all(np.isfinite(feature_vector)) and np.isfinite(delta_recall) and np.isfinite(delta_ndcg)):
            continue

        x_rows.append(feature_vector)
        y_recall.append(delta_recall)
        y_ndcg.append(delta_ndcg)

    if x_rows:
        X = np.asarray(x_rows, dtype=np.float64)
        y_recall_arr = np.asarray(y_recall, dtype=np.float64)
        y_ndcg_arr = np.asarray(y_ndcg, dtype=np.float64)

        coeffs_recall, r2_recall = fit_linear_boundary(X, y_recall_arr)
        coeffs_ndcg, r2_ndcg = fit_linear_boundary(X, y_ndcg_arr)
    else:
        X = np.zeros((0, len(feature_names)), dtype=np.float64)
        coeffs_recall = np.zeros(len(feature_names) + 1)
        coeffs_ndcg = np.zeros(len(feature_names) + 1)
        r2_recall = float("nan")
        r2_ndcg = float("nan")

    table_dir = PROJECT_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    coeff_csv = table_dir / "diagnostic_boundaries.csv"
    with coeff_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "intercept", *feature_names, "r2"])
        writer.writerow([
            "delta_recall",
            coeffs_recall[0],
            *coeffs_recall[1:],
            r2_recall,
        ])
        writer.writerow([
            "delta_ndcg",
            coeffs_ndcg[0],
            *coeffs_ndcg[1:],
            r2_ndcg,
        ])

    # Save detailed dataset rows
    detailed_csv = table_dir / "diagnostic_dataset_deltas.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if X.shape[0] > 0:
        item_entropy = X[:, feature_names.index("item_entropy")]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(item_entropy, y_recall_arr, color="tab:blue")
        entropy_line = np.linspace(item_entropy.min(), item_entropy.max(), 100)
        slope = coeffs_recall[feature_names.index("item_entropy") + 1]
        intercept = coeffs_recall[0]
        ax.plot(entropy_line, intercept + slope * entropy_line, color="tab:orange", label="Linear fit")
        for dataset, x_val, y_val in zip(DATASETS, item_entropy, y_recall_arr):
            ax.text(x_val, y_val, dataset, fontsize=9, ha="right", va="bottom")
        ax.set_xlabel("Item Entropy")
        ax.set_ylabel("Δ Recall@10 (Context - Fixed)")
        ax.set_title("Diagnostic boundary vs. item entropy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        fig_dir = PROJECT_ROOT / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_path = fig_dir / "diagnostic_boundaries.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
    else:
        plot_path = None

    # Write LaTeX table summarizing coefficients
    tex_path = table_dir / "diagnostic_boundaries.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write(
            "  \\caption{Linear diagnostic boundaries: Δ metrics vs. dataset features. Coefficients correspond to standardized features.}\n"
        )
        f.write("  \\label{tab:diagnostic-boundaries}\n")
        f.write("  \\begin{tabular}{lrrrrrrrrr}\n")
        f.write("    \\toprule\n")
        header = ["Target", "Intercept", *feature_names, "$R^2$"]
        f.write("    " + " & ".join(header) + " \\\\" + "\n")
        f.write("    \\midrule\n")
        f.write(
            "    delta\\_recall & "
            + " & ".join(f"{val:.3f}" for val in coeffs_recall)
            + f" & {r2_recall:.3f} \\\\"
            + "\n"
        )
        f.write(
            "    delta\\_ndcg & "
            + " & ".join(f"{val:.3f}" for val in coeffs_ndcg)
            + f" & {r2_ndcg:.3f} \\\\"
            + "\n"
        )
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Coefficients written to {coeff_csv}")
    print(f"Detailed deltas written to {detailed_csv}")
    if plot_path is not None:
        print(f"Figure saved to {plot_path}")
    else:
        print("Insufficient data for diagnostic plot; skipping figure.")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()

