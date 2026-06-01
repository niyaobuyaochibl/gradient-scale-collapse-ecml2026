"""Stratified evaluation by item popularity and user activity across datasets/variants.

Produces Markdown and optional LaTeX tables summarizing Recall@10 / NDCG@10 for
tercile groups (low/mid/high) using saved best checkpoints.
"""

from __future__ import annotations

import argparse
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_fusion import (  # type: ignore
    load_data,
    build_model,
    evaluate,
)
from models import (  # type: ignore
    compute_item_popularity,
    compute_user_activity,
    compute_item_cf_confidence,
    compute_item_text_quality,
    compute_item_recency,
)


@dataclass
class DatasetConfig:
    dataset: str
    variant_to_config: Dict[str, Path]
    variant_to_results_dir: Dict[str, Path]


CONFIGS: Dict[str, DatasetConfig] = {
    "ml1m": DatasetConfig(
        dataset="ml1m",
        variant_to_config={
            "fixed": PROJECT_ROOT / "configs/extended/movielens_fixed_baseline.yaml",
            "lightgcn_only": PROJECT_ROOT / "configs/extended/movielens_lightgcn_only.yaml",
            "gbaf": PROJECT_ROOT / "configs/extended/movielens_gbaf.yaml",
            "gbaf_adaptive": PROJECT_ROOT / "configs/extended/movielens_gbaf_adaptive.yaml",
            "gbafv2": PROJECT_ROOT / "configs/extended/movielens_gbafv2.yaml",
            "attention_pop": PROJECT_ROOT / "configs/extended/movielens_attention_pop.yaml",
            "attention_pop_user": PROJECT_ROOT / "configs/extended/movielens_attention_pop_user.yaml",
            "attention_concat": PROJECT_ROOT / "configs/extended/movielens_attention_concat.yaml",
            "attention_ctx": PROJECT_ROOT / "configs/extended/movielens_attention_ctx.yaml",
            "attention_ctx_lora": PROJECT_ROOT / "configs/extended/movielens_attention_ctx_lora.yaml",
        },
        variant_to_results_dir={
            "fixed": Path("results/ml1m/fixed"),
            "lightgcn_only": Path("results/movielens_lightgcn_only"),
            "gbaf": Path("results/movielens_gbaf"),
            "gbaf_adaptive": Path("results/movielens_gbaf_adaptive"),
            "gbafv2": Path("results/movielens_gbafv2"),
            "attention_pop": Path("results/ml1m/attention_pop"),
            "attention_pop_user": Path("results/ml1m/attention_pop_user"),
            "attention_concat": Path("results/ml1m/attention_concat"),
            "attention_ctx": Path("results/ml1m/attention_ctx"),
            "attention_ctx_lora": Path("results/ml1m/attention_ctx_lora"),
        },
    ),
    "amazonff": DatasetConfig(
        dataset="amazonff",
        variant_to_config={
            "fixed": PROJECT_ROOT / "configs/extended/amazon_fixed_baseline.yaml",
            "lightgcn_only": PROJECT_ROOT / "configs/extended/amazon_lightgcn_only.yaml",
            "gbaf": PROJECT_ROOT / "configs/extended/amazon_gbaf.yaml",
            "gbaf_adaptive": PROJECT_ROOT / "configs/extended/amazon_gbaf_adaptive.yaml",
            "gbafv2": PROJECT_ROOT / "configs/extended/amazon_gbafv2.yaml",
            "attention_pop": PROJECT_ROOT / "configs/extended/amazon_attention_pop.yaml",
            "attention_pop_user": PROJECT_ROOT / "configs/extended/amazon_attention_pop_user.yaml",
            "attention_concat": PROJECT_ROOT / "configs/extended/amazon_attention_concat.yaml",
            "attention_ctx": PROJECT_ROOT / "configs/extended/amazon_attention_ctx.yaml",
            "attention_ctx_lora": PROJECT_ROOT / "configs/extended/amazon_attention_ctx_lora.yaml",
        },
        variant_to_results_dir={
            "fixed": Path("results/amazonff/fixed"),
            "lightgcn_only": Path("results/amazon_lightgcn_only"),
            "gbaf": Path("results/amazon_gbaf"),
            "gbaf_adaptive": Path("results/amazon_gbaf_adaptive"),
            "gbafv2": Path("results/amazon_gbafv2"),
            "attention_pop": Path("results/amazonff/attention_pop"),
            "attention_pop_user": Path("results/amazonff/attention_pop_user"),
            "attention_concat": Path("results/amazonff/attention_concat"),
            "attention_ctx": Path("results/amazonff/attention_ctx"),
            "attention_ctx_lora": Path("results/amazonff/attention_ctx_lora"),
        },
    ),
    "yelp": DatasetConfig(
        dataset="yelp",
        variant_to_config={
            "fixed": PROJECT_ROOT / "configs/extended/yelp_fixed_baseline.yaml",
            "lightgcn_only": PROJECT_ROOT / "configs/extended/yelp_lightgcn_only.yaml",
            "gbaf": PROJECT_ROOT / "configs/extended/yelp_gbaf.yaml",
            "gbaf_adaptive": PROJECT_ROOT / "configs/extended/yelp_gbaf_adaptive.yaml",
            "gbafv2": PROJECT_ROOT / "configs/extended/yelp_gbafv2.yaml",
            "attention_pop": PROJECT_ROOT / "configs/extended/yelp_attention_pop.yaml",
            "attention_pop_user": PROJECT_ROOT / "configs/extended/yelp_attention_pop_user.yaml",
            "attention_concat": PROJECT_ROOT / "configs/extended/yelp_attention_concat.yaml",
            "attention_ctx": PROJECT_ROOT / "configs/extended/yelp_attention_ctx.yaml",
            "attention_ctx_lora": PROJECT_ROOT / "configs/extended/yelp_attention_ctx_lora.yaml",
        },
        variant_to_results_dir={
            "fixed": Path("results/yelp/fixed"),
            "lightgcn_only": Path("results/yelp_lightgcn_only"),
            "gbaf": Path("results/yelp/gbaf"),
            "gbaf_adaptive": Path("results/yelp/gbaf_adaptive"),
            "gbafv2": Path("results/yelp/gbafv2"),
            "attention_pop": Path("results/yelp/attention_pop"),
            "attention_pop_user": Path("results/yelp/attention_pop_user"),
            "attention_concat": Path("results/yelp/attention_concat"),
            "attention_ctx": Path("results/yelp/attention_ctx"),
            "attention_ctx_lora": Path("results/yelp/attention_ctx_lora"),
        },
    ),
    "mind": DatasetConfig(
        dataset="mind",
        variant_to_config={
            "fixed": PROJECT_ROOT / "configs/extended/mind_fixed_baseline.yaml",
            "lightgcn_only": PROJECT_ROOT / "configs/extended/mind_lightgcn_only.yaml",
            "gbaf": PROJECT_ROOT / "configs/extended/mind_gbaf.yaml",
            "gbaf_adaptive": PROJECT_ROOT / "configs/extended/mind_gbaf_adaptive.yaml",
            "gbafv2": PROJECT_ROOT / "configs/extended/mind_gbafv2.yaml",
            "attention_pop": PROJECT_ROOT / "configs/extended/mind_attention_pop.yaml",
            "attention_pop_user": PROJECT_ROOT / "configs/extended/mind_attention_pop_user.yaml",
            "attention_concat": PROJECT_ROOT / "configs/extended/mind_attention_concat.yaml",
            "attention_ctx": PROJECT_ROOT / "configs/extended/mind_attention_ctx.yaml",
            "attention_ctx_lora": PROJECT_ROOT / "configs/extended/mind_attention_ctx_lora.yaml",
        },
        variant_to_results_dir={
            "fixed": Path("results/mind/fixed"),
            "lightgcn_only": Path("results/mind_lightgcn_only"),
            "gbaf": Path("results/mind/gbaf"),
            "gbaf_adaptive": Path("results/mind/gbaf_adaptive"),
            "gbafv2": Path("results/mind/gbafv2"),
            "attention_pop": Path("results/mind/attention_pop"),
            "attention_pop_user": Path("results/mind/attention_pop_user"),
            "attention_concat": Path("results/mind/attention_concat"),
            "attention_ctx": Path("results/mind/attention_ctx"),
            "attention_ctx_lora": Path("results/mind/attention_ctx_lora"),
        },
    ),
}


def tercile_thresholds(values: np.ndarray) -> Tuple[float, float]:
    p33 = float(np.percentile(values, 33))
    p67 = float(np.percentile(values, 67))
    return p33, p67


def assign_groups(array: np.ndarray, p33: float, p67: float) -> np.ndarray:
    # 0: low, 1: mid, 2: high
    groups = np.zeros_like(array, dtype=np.int64)
    groups[array > p33] = 1
    groups[array > p67] = 2
    return groups


def slice_pairs_by_item_group(pairs: np.ndarray, item_groups: np.ndarray) -> Dict[str, np.ndarray]:
    # Bound-check indices to avoid out-of-range errors if sparse mappings exist
    valid_mask = (pairs[:, 1] >= 0) & (pairs[:, 1] < len(item_groups))
    pairs = pairs[valid_mask]
    idx = pairs[:, 1].astype(int)
    low = pairs[item_groups[idx] == 0]
    mid = pairs[item_groups[idx] == 1]
    high = pairs[item_groups[idx] == 2]
    return {"low": low, "mid": mid, "high": high}


def slice_pairs_by_user_group(pairs: np.ndarray, user_groups: np.ndarray) -> Dict[str, np.ndarray]:
    valid_mask = (pairs[:, 0] >= 0) & (pairs[:, 0] < len(user_groups))
    pairs = pairs[valid_mask]
    idx = pairs[:, 0].astype(int)
    low = pairs[user_groups[idx] == 0]
    mid = pairs[user_groups[idx] == 1]
    high = pairs[user_groups[idx] == 2]
    return {"low": low, "mid": mid, "high": high}


def load_best_model_path(result_dir: Path, seed: int) -> Path:
    return result_dir / f"best_model_seed{seed}.pt"


def eval_subset(model, pairs: np.ndarray, item_texts, n_items: int, n_users: int, device: torch.device) -> Tuple[float, float]:
    if len(pairs) == 0:
        return float("nan"), float("nan")
    # bound check
    mask = (pairs[:, 0] >= 0) & (pairs[:, 0] < n_users) & (pairs[:, 1] >= 0) & (pairs[:, 1] < n_items)
    pairs = pairs[mask]
    if len(pairs) == 0:
        return float("nan"), float("nan")
    # Use the same evaluation function but pass numpy array as-is
    recall, ndcg = evaluate(model, pairs, item_texts, n_items, k=10, device=device)
    return float(recall), float(ndcg)


def stratified_eval(dataset_key: str, variants: Iterable[str], seeds: Iterable[int], output_dir: Path) -> Path:
    cfg = CONFIGS[dataset_key]
    records: List[Dict] = []

    for variant in variants:
        if variant not in cfg.variant_to_config:
            continue
        config_path = cfg.variant_to_config[variant]
        with config_path.open("r") as handle:
            config = yaml.safe_load(handle)

        device = resolve_device(config)

        # Load data and build item/user groups from training counts
        train, val, test, item_texts, stats = load_data(config["data"]["data_dir"], config, config_path.parent)

        test_pairs = test.values[:, [0, 1]] if hasattr(test, "values") else np.asarray(test)
        try:
            limit = max(0, int(globals().get("_TEST_LIMIT", 0)))
        except Exception:
            limit = 0
        if limit and len(test_pairs) > limit:
            test_pairs = test_pairs[:limit]

        item_pop_tensor = compute_item_popularity(train, stats["n_items"])
        item_pop = item_pop_tensor.numpy()
        p33_i, p67_i = tercile_thresholds(item_pop)
        item_groups = assign_groups(item_pop, p33_i, p67_i)

        user_act_tensor = compute_user_activity(train, stats["n_users"])
        user_act = user_act_tensor.numpy()
        p33_u, p67_u = tercile_thresholds(user_act)
        user_groups = assign_groups(user_act, p33_u, p67_u)

        by_item = slice_pairs_by_item_group(test_pairs, item_groups)
        by_user = slice_pairs_by_user_group(test_pairs, user_groups)

        for seed in seeds:
            result_dir = cfg.variant_to_results_dir.get(variant)
            if result_dir is None:
                continue
            checkpoint = load_best_model_path(result_dir, seed)
            if not checkpoint.exists():
                continue
            model = build_model(config, device=device)
            prepare_attention_features(model, config, train, item_texts)
            state = torch.load(str(checkpoint), map_location=device)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()

            for group_name, subset in by_item.items():
                r, n = eval_subset(model, subset, item_texts, stats["n_items"], stats["n_users"], device)
                records.append({
                    "dataset": dataset_key,
                    "variant": variant,
                    "seed": seed,
                    "slice": f"item_{group_name}",
                    "recall@10": r,
                    "ndcg@10": n,
                })
            for group_name, subset in by_user.items():
                r, n = eval_subset(model, subset, item_texts, stats["n_items"], stats["n_users"], device)
                records.append({
                    "dataset": dataset_key,
                    "variant": variant,
                    "seed": seed,
                    "slice": f"user_{group_name}",
                    "recall@10": r,
                    "ndcg@10": n,
                })

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"stratified_{dataset_key}.csv"
    df.to_csv(csv_path, index=False)

    # Aggregate mean±std per (dataset, variant, slice)
    agg = df.groupby(["dataset", "variant", "slice"]).agg(
        recall_mean=("recall@10", "mean"),
        recall_std=("recall@10", "std"),
        ndcg_mean=("ndcg@10", "mean"),
        ndcg_std=("ndcg@10", "std"),
    ).reset_index()
    try:
        markdown_table = agg.to_markdown(index=False, floatfmt=('.0f', '.3f', '.3f', '.3f', '.3f'))
    except Exception:
        markdown_table = agg.to_string(index=False)
    md_lines = [f"# Stratified Evaluation: {dataset_key}", "", markdown_table]
    md_path = output_dir / f"stratified_{dataset_key}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ Wrote {csv_path} and {md_path}")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified evaluation by item popularity and user activity")
    parser.add_argument("--datasets", nargs="*", default=["ml1m", "amazonff", "yelp", "mind"])
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["lightgcn_only", "fixed", "gbaf", "gbaf_adaptive"],
    )
    parser.add_argument("--seeds", nargs="*", default=["42", "123", "999", "2024", "2025"]) 
    parser.add_argument("--limit-test", type=int, default=5000, help="Limit number of test interactions (head)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments/extended/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds]
    out_dir = Path(args.output_dir)
    # Inject limit into module global by monkey patching within this simple script
    global _TEST_LIMIT
    _TEST_LIMIT = int(args.limit_test)
    for ds in args.datasets:
        stratified_eval(ds, args.variants, seeds, out_dir)


def resolve_device(config: Dict) -> torch.device:
    device_cfg = config.get("device", "auto")
    if isinstance(device_cfg, str) and device_cfg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def prepare_attention_features(
    model: torch.nn.Module,
    config: Dict,
    train_data,
    item_embeddings,
) -> None:
    if not isinstance(model, torch.nn.Module):
        return
    if not hasattr(model, "features"):
        return
    features = set(getattr(model, "features", ()) or [])
    feature_params = config.get("model", {}).get("feature_params", {})
    n_items = config["model"]["n_items"]
    n_users = config["model"]["n_users"]

    # compute shared stats once
    popularity_tensor = compute_item_popularity(train_data, n_items)
    model.set_item_popularity(popularity_tensor)

    if "cf_confidence" in features:
        method = str(feature_params.get("cf_confidence_method", "log")).lower()
        cf_confidence = compute_item_cf_confidence(popularity_tensor, method=method)
        model.set_item_cf_confidence(cf_confidence)

    if "text_quality" in features:
        text_quality = compute_item_text_quality(item_embeddings, n_items)
        model.set_item_text_quality(text_quality)

    if "recency" in features:
        recency = compute_item_recency(train_data, n_items)
        model.set_item_recency(recency)

    if getattr(model, "context_features", None) and "user_activity" in model.context_features:
        user_activity = compute_user_activity(train_data, n_users)
        model.set_user_activity(user_activity)


if __name__ == "__main__":
    main()


