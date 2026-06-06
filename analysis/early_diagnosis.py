#!/usr/bin/env python3
"""Recompute the camera-ready leave-one-dataset-out early-diagnosis results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL_EPOCHS = {
    "ml1m": 50,
    "amazonff": 50,
    "amazon_books": 50,
    "amazon_cds": 50,
    "yelp": 60,
    "mind": 60,
}


def evaluate_lodo(df: pd.DataFrame, k: int) -> tuple[dict[str, float], pd.DataFrame]:
    subset = df[df["k"] == k].dropna(subset=["dev_k", "val_delta_k"]).copy()
    prediction_rows: list[dict[str, object]] = []
    for held_out in sorted(subset["dataset"].unique()):
        train = subset[subset["dataset"] != held_out]
        test = subset[subset["dataset"] == held_out]
        threshold = float(train["dev_k"].median())
        for _, row in test.iterrows():
            prediction = int(row["val_delta_k"] < 0 and row["dev_k"] > threshold)
            prediction_rows.append({
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "k": k,
                "threshold": threshold,
                "prediction": prediction,
                "label": int(row["collapse_label"]),
            })
    predictions = pd.DataFrame(prediction_rows)
    y_pred = predictions["prediction"].to_numpy(dtype=int)
    y_true = predictions["label"].to_numpy(dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    saved = sum(
        TOTAL_EPOCHS[row.dataset] - k
        for row in predictions.itertuples()
        if row.prediction == 1 and row.label == 1
    )
    metrics = {
        "k": k,
        "n_cases": len(predictions),
        "n_collapse": int(y_true.sum()),
        "accuracy": (tp + tn) / len(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "saved_epochs_total": saved,
        "saved_epochs_per_true_positive": saved / tp if tp else 0.0,
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("results/tables/early_diagnosis_features.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("results/tables/early_diagnosis_metrics_by_k.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("results/tables/early_diagnosis_lodo_predictions.csv"))
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    metrics_rows = []
    prediction_frames = []
    for k in (3, 5, 10):
        metrics, predictions = evaluate_lodo(features, k)
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(args.metrics_output, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.predictions_output, index=False)
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(f"[OK] wrote {args.metrics_output}")
    print(f"[OK] wrote {args.predictions_output}")


if __name__ == "__main__":
    main()
