"""Aggregate stratified evaluation into LaTeX tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "experiments" / "extended" / "analysis"
TABLE_DIR = PROJECT_ROOT / "tables"

DATASETS = ["ml1m", "amazonff", "yelp", "mind"]
ITEM_SLICE_PREFIX = "item_"
USER_SLICE_PREFIX = "user_"
VARIANT_ORDER = ["fixed", "attention_pop", "attention_ctx", "attention_ctx_lora"]


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def load_stratified(dataset: str) -> pd.DataFrame:
    csv_path = ANALYSIS_DIR / f"stratified_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    return df


def format_cell(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std) or std == 0:
        return f"{mean*100:.2f}"
    return f"{mean*100:.2f}$\\pm${std*100:.2f}"


def build_table(rows: List[Dict[str, str]], caption: str, label: str, columns: List[str]) -> str:
    col_spec = "l" + "c" * (len(columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    " + " & ".join(columns) + " \\\\",
        "    \\midrule",
    ]

    for row in rows:
        values = [row[col] for col in columns]
        lines.append("    " + " & ".join(values) + " \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def aggregate() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    item_rows: List[Dict[str, str]] = []
    user_rows: List[Dict[str, str]] = []

    for dataset in DATASETS:
        df = load_stratified(dataset)
        agg = df.groupby(["variant", "slice"]).agg(
            recall_mean=("recall@10", "mean"),
            recall_std=("recall@10", "std"),
            ndcg_mean=("ndcg@10", "mean"),
            ndcg_std=("ndcg@10", "std"),
        ).reset_index()

        for variant in VARIANT_ORDER:
            subset = agg[agg["variant"] == variant]
            if subset.empty:
                continue
            row_label = latex_escape(f"{dataset} / {variant}")
            item_low = subset[subset["slice"] == ITEM_SLICE_PREFIX + "low"]
            item_mid = subset[subset["slice"] == ITEM_SLICE_PREFIX + "mid"]
            item_high = subset[subset["slice"] == ITEM_SLICE_PREFIX + "high"]

            user_low = subset[subset["slice"] == USER_SLICE_PREFIX + "low"]
            user_mid = subset[subset["slice"] == USER_SLICE_PREFIX + "mid"]
            user_high = subset[subset["slice"] == USER_SLICE_PREFIX + "high"]

            item_rows.append(
                {
                    "Dataset / Variant": row_label,
                    "Low": format_cell(
                        item_low["recall_mean"].values[0] if not item_low.empty else float("nan"),
                        item_low["recall_std"].values[0] if not item_low.empty else float("nan"),
                    ),
                    "Mid": format_cell(
                        item_mid["recall_mean"].values[0] if not item_mid.empty else float("nan"),
                        item_mid["recall_std"].values[0] if not item_mid.empty else float("nan"),
                    ),
                    "High": format_cell(
                        item_high["recall_mean"].values[0] if not item_high.empty else float("nan"),
                        item_high["recall_std"].values[0] if not item_high.empty else float("nan"),
                    ),
                }
            )

            user_rows.append(
                {
                    "Dataset / Variant": row_label,
                    "Low": format_cell(
                        user_low["recall_mean"].values[0] if not user_low.empty else float("nan"),
                        user_low["recall_std"].values[0] if not user_low.empty else float("nan"),
                    ),
                    "Mid": format_cell(
                        user_mid["recall_mean"].values[0] if not user_mid.empty else float("nan"),
                        user_mid["recall_std"].values[0] if not user_mid.empty else float("nan"),
                    ),
                    "High": format_cell(
                        user_high["recall_mean"].values[0] if not user_high.empty else float("nan"),
                        user_high["recall_std"].values[0] if not user_high.empty else float("nan"),
                    ),
                }
            )

    item_table = build_table(
        item_rows,
        "Item-popularity stratified Recall@10 (mean$\\pm$std across seeds).",
        "tab:stratified-item",
        ["Dataset / Variant", "Low", "Mid", "High"],
    )
    user_table = build_table(
        user_rows,
        "User-activity stratified Recall@10 (mean$\\pm$std across seeds).",
        "tab:stratified-user",
        ["Dataset / Variant", "Low", "Mid", "High"],
    )

    (TABLE_DIR / "stratified_item.tex").write_text(item_table, encoding="utf-8")
    (TABLE_DIR / "stratified_user.tex").write_text(user_table, encoding="utf-8")


if __name__ == "__main__":
    aggregate()

