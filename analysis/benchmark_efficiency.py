"""Benchmark compute efficiency for fusion variants and plot Pareto trade-offs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FlopCountAnalysis = None  # fallback: disable FLOPs if fvcore is unavailable
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path("data/extended_fusion/results")


from train_fusion import (  # noqa: E402
    RecommendationDataset,
    build_model,
    load_data,
)
from models import (  # noqa: E402
    AttentionFusion,
    compute_item_popularity,
    compute_user_activity,
    compute_item_cf_confidence,
    compute_item_text_quality,
    compute_item_recency,
)


CONFIG_MAP: Dict[str, Dict[str, str]] = {
    "ml1m": {
        "fixed": "configs/extended/movielens_fixed_baseline.yaml",
        "attention_concat": "configs/extended/movielens_attention_concat.yaml",
        "attention_pop": "configs/extended/movielens_attention_pop.yaml",
        "attention_pop_user": "configs/extended/movielens_attention_pop_user.yaml",
        "attention_ctx": "configs/extended/movielens_attention_ctx.yaml",
        "attention_ctx_lora": "configs/extended/movielens_attention_ctx_lora.yaml",
    },
    "amazonff": {
        "fixed": "configs/extended/amazon_fixed_baseline.yaml",
        "attention_concat": "configs/extended/amazon_attention_concat.yaml",
        "attention_pop": "configs/extended/amazon_attention_pop.yaml",
        "attention_pop_user": "configs/extended/amazon_attention_pop_user.yaml",
        "attention_ctx": "configs/extended/amazon_attention_ctx.yaml",
        "attention_ctx_lora": "configs/extended/amazon_attention_ctx_lora.yaml",
    },
    "yelp": {
        "fixed": "configs/extended/yelp_fixed_baseline.yaml",
        "attention_concat": "configs/extended/yelp_attention_concat.yaml",
        "attention_pop": "configs/extended/yelp_attention_pop.yaml",
        "attention_pop_user": "configs/extended/yelp_attention_pop_user.yaml",
        "attention_ctx": "configs/extended/yelp_attention_ctx.yaml",
        "attention_ctx_lora": "configs/extended/yelp_attention_ctx_lora.yaml",
    },
    "mind": {
        "fixed": "configs/extended/mind_fixed_baseline.yaml",
        "attention_concat": "configs/extended/mind_attention_concat.yaml",
        "attention_pop": "configs/extended/mind_attention_pop.yaml",
        "attention_pop_user": "configs/extended/mind_attention_pop_user.yaml",
        "attention_ctx": "configs/extended/mind_attention_ctx.yaml",
        "attention_ctx_lora": "configs/extended/mind_attention_ctx_lora.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compute efficiency and plot Pareto trade-offs.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(CONFIG_MAP.keys()),
        help="Datasets to benchmark (default: all)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["fixed", "attention_concat", "attention_pop", "attention_pop_user", "attention_ctx", "attention_ctx_lora"],
        help="Model variants to benchmark (default: all)",
    )
    parser.add_argument("--sample-batches", type=int, default=4, help="Number of batches to sample for benchmarking")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up iterations before timing")
    parser.add_argument("--iters", type=int, default=8, help="Timed iterations per variant")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures"),
        help="Directory to store benchmark outputs/plots (default: figures)",
    )
    return parser.parse_args()


def _prepare_model_state(
    model: torch.nn.Module,
    train_data,
    item_embeddings,
    config: Dict,
) -> None:
    if isinstance(model, AttentionFusion):
        n_items = config["model"]["n_items"]
        n_users = config["model"]["n_users"]
        feature_params = config["model"].get("feature_params", {})

        popularity = compute_item_popularity(train_data, n_items)
        model.set_item_popularity(popularity)

        features = set(getattr(model, "features", ()) or [])
        if "cf_confidence" in features:
            method = str(feature_params.get("cf_confidence_method", "log")).lower()
            cf_conf = compute_item_cf_confidence(popularity, method=method)
            model.set_item_cf_confidence(cf_conf)

        if "text_quality" in features:
            model.set_item_text_quality(compute_item_text_quality(item_embeddings, n_items))

        if "recency" in features:
            model.set_item_recency(compute_item_recency(train_data, n_items))

        if getattr(model, "context_features", None) and "user_activity" in model.context_features:
            activity = compute_user_activity(train_data, n_users)
            model.set_user_activity(activity)


def _make_batch_loader(
    train_data,
    item_embeddings,
    config: Dict,
    sample_batches: int,
) -> Iterable[Dict[str, torch.Tensor]]:
    batch_size = int(config["training"]["batch_size"])
    negative_sampling = int(config["training"].get("negative_sampling", 4))
    total_needed = batch_size * sample_batches

    if hasattr(train_data, "iloc"):
        subset = train_data.iloc[:total_needed]
    else:
        subset = train_data[:total_needed]

    dataset = RecommendationDataset(
        subset,
        item_embeddings,
        config["model"]["n_items"],
        negative_sampling=negative_sampling,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader


def _training_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    user_ids = batch["user_id"].to(device)
    pos_items = batch["pos_item"].to(device)
    neg_items = batch["neg_items"].to(device)
    pos_texts = batch["pos_text"].to(device).float()
    neg_texts = batch["neg_texts"].to(device).float()

    batch_size, n_neg = neg_items.shape

    optimizer.zero_grad(set_to_none=True)

    pos_scores = model(user_ids, pos_items, pos_texts)

    user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg).reshape(-1)
    neg_items_flat = neg_items.reshape(-1)
    neg_texts_flat = neg_texts.reshape(-1, neg_texts.size(-1))

    neg_scores = model(user_ids_expanded, neg_items_flat, neg_texts_flat).reshape(batch_size, n_neg)

    pos_scores = pos_scores.unsqueeze(1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

    loss.backward()
    optimizer.step()

    cache = {
        "user_ids": user_ids,
        "pos_items": pos_items,
        "pos_texts": pos_texts,
        "user_ids_expanded": user_ids_expanded,
        "neg_items_flat": neg_items_flat,
        "neg_texts_flat": neg_texts_flat,
    }

    return float(loss.item()), cache


def _compute_flops(model: torch.nn.Module, cache: Dict[str, torch.Tensor]) -> float:
    if FlopCountAnalysis is None:
        return float("nan")
    model.eval()
    with torch.no_grad():
        pos_flops = FlopCountAnalysis(
            model,
            (cache["user_ids"], cache["pos_items"], cache["pos_texts"]),
        ).total()
        neg_flops = FlopCountAnalysis(
            model,
            (cache["user_ids_expanded"], cache["neg_items_flat"], cache["neg_texts_flat"]),
        ).total()
    model.train()
    return float(pos_flops + neg_flops)


def _load_performance_metrics(dataset: str, variant: str) -> Tuple[float, float]:
    result_dir = RESULTS_ROOT / dataset / variant
    files = sorted(result_dir.glob("results_seed*.json"))
    recalls: List[float] = []
    ndcgs: List[float] = []
    for file in files:
        with file.open() as f:
            data = json.load(f)
        recalls.append(float(data.get("recall@10", 0.0)))
        ndcgs.append(float(data.get("ndcg@10", 0.0)))

    recall_mean = sum(recalls) / len(recalls) if recalls else float("nan")
    ndcg_mean = sum(ndcgs) / len(ndcgs) if ndcgs else float("nan")
    return recall_mean, ndcg_mean


def benchmark_variant(
    dataset: str,
    variant: str,
    config_path: Path,
    args: argparse.Namespace,
    dataset_cache: Dict[str, Tuple],
    device: torch.device,
) -> Dict:
    with config_path.open() as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent

    if dataset not in dataset_cache:
        train_data, val_data, test_data, item_embeddings, stats = load_data(
            config["data"]["data_dir"], config, config_dir
        )
        dataset_cache[dataset] = (train_data, val_data, test_data, item_embeddings, stats)
    else:
        train_data, val_data, test_data, item_embeddings, stats = dataset_cache[dataset]

    model = build_model(config, device)
    model.to(device)
    _prepare_model_state(model, train_data, item_embeddings, config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    loader = _make_batch_loader(train_data, item_embeddings, config, args.sample_batches)
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise RuntimeError(f"No training data available for {dataset}/{variant}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up iterations
    for _ in range(args.warmup):
        _training_step(model, batch, optimizer, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    cache_sample = None
    for _ in range(args.iters):
        loss, cache_sample = _training_step(model, batch, optimizer, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / max(args.iters, 1)
    peak_memory = float("nan")
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MiB

    total_flops = _compute_flops(model, cache_sample)
    params = sum(p.numel() for p in model.parameters())

    recall_mean, ndcg_mean = _load_performance_metrics(dataset, variant)

    record = {
        "dataset": dataset,
        "variant": variant,
        "step_time_ms": avg_time * 1000.0,
        "peak_mem_mib": peak_memory,
        "flops_g": total_flops / 1e9,
        "params_m": params / 1e6,
        "recall_mean": recall_mean,
        "ndcg_mean": ndcg_mean,
    }

    # Cleanup to release GPU memory between variants
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record


def plot_pareto(df: pd.DataFrame, output_dir: Path) -> Path:
    datasets = df["dataset"].unique()
    datasets = sorted(datasets, key=lambda x: ["ml1m", "amazonff", "yelp", "mind"].index(x) if x in {"ml1m", "amazonff", "yelp", "mind"} else len(datasets))

    # Use a large figure for print clarity; avoid constrained_layout due to colorbar + grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(df["peak_mem_mib"].min()), float(df["peak_mem_mib"].max()))
    marker_map = {
        "fixed": "o",  # circle
        "attention_pop": "s",  # square
        "attention_pop_user": "^",  # triangle
        "attention_concat": "D",  # diamond
        "attention_ctx": "v",
        "attention_ctx_lora": "P",
    }
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    axes = axes.flatten()
    scatter = None

    for ax, dataset in zip(axes, datasets):
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            ax.axis("off")
            continue

        # Use step time (ms) as X axis to avoid stacking when FLOPs are missing
        x_vals = subset["step_time_ms"].astype(float)
        x_label = "Step time (ms)"

        # draw per-variant with distinct markers
        scatter = None
        for variant_name in subset["variant"].unique():
            sv = subset[subset["variant"] == variant_name]
        scatter = ax.scatter(
                sv["step_time_ms"].astype(float),
                sv["recall_mean"].astype(float),
                s=30 + sv["params_m"].fillna(0.0) * 6.0,
                c=sv["peak_mem_mib"],
                cmap=cmap,
                norm=norm,
            alpha=0.85,
            edgecolors="k",
                linewidths=0.8,
                marker=marker_map.get(variant_name, "o"),
            )
        # 不再绘制文本标签，由 marker 形状 + 图例传达方法信息

        ax.set_title(dataset.upper(), fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Recall@10", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused axes if any
    for ax in axes[len(datasets):]:
        ax.axis("off")

    # Place colorbar at the bottom (horizontal) to avoid covering subplots
    if scatter is not None:
        fig.subplots_adjust(bottom=0.18)  # make room at bottom for colorbar
        cax = fig.add_axes([0.15, 0.09, 0.7, 0.03])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
        cbar.set_label("Peak memory (MiB)", fontsize=12)

    # Add legends at figure-level to avoid covering any subplot
    # 形状图例（方法）
    shape_handles = [
        plt.Line2D([0], [0], marker=marker_map["fixed"], color="k", linestyle="None", markerfacecolor="none", markersize=8, label="Fixed"),
        plt.Line2D([0], [0], marker=marker_map["attention_pop"], color="k", linestyle="None", markerfacecolor="none", markersize=8, label="Popularity-gated"),
        plt.Line2D([0], [0], marker=marker_map["attention_pop_user"], color="k", linestyle="None", markerfacecolor="none", markersize=8, label="Pop+User"),
        plt.Line2D([0], [0], marker=marker_map["attention_concat"], color="k", linestyle="None", markerfacecolor="none", markersize=8, label="Concat-MLP"),
    ]
    fig.legend(
        handles=shape_handles,
        title="Variant (marker shape)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=True,
    )

    # Marker size 图例（参数量）
    example_sizes = [0.6, 3.7, 10.0]  # representative params (M)
    size_scale = lambda pm: 30 + pm * 6.0
    size_handles = [plt.scatter([], [], s=size_scale(s), edgecolors="k", facecolors="none") for s in example_sizes]
    size_labels = [f"Params ≈ {s:g} M" for s in example_sizes]
    fig.legend(
        size_handles,
        size_labels,
        title="Marker size",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.94),
        frameon=True,
    )

    # Global caption describing encodings
    # Removed extra encoding footer per request; legends and colorbar already convey encodings clearly.

    fig.suptitle("Pareto Trade-offs: Recall vs Compute Cost", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.02, 0.18, 0.98, 0.94))  # leave headroom for top legend

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "pareto_tradeoffs.pdf"
    # Save vector PDF for print clarity; also export a PNG for quick viewing
    fig.savefig(plot_path)
    fig.savefig((output_dir / "pareto_tradeoffs.png"), dpi=320)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    dataset_cache: Dict[str, Tuple] = {}
    records: List[Dict] = []

    # Prefer existing metrics if available to avoid recomputation
    csv_path = output_dir / "efficiency_metrics.csv"
    legacy_csv = PROJECT_ROOT / "experiments/extended/analysis/efficiency_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Using existing metrics at {csv_path}")
    elif legacy_csv.exists():
        df = pd.read_csv(legacy_csv)
        print(f"Using existing metrics at {legacy_csv}")
    else:
        for dataset in args.datasets:
            if dataset not in CONFIG_MAP:
                print(f"Unknown dataset '{dataset}', skipping.")
                continue
            for variant in args.variants:
                config_path = Path(CONFIG_MAP[dataset].get(variant, ""))
                if not config_path.exists():
                    print(f"Missing config for {dataset}/{variant}, skipping.")
                    continue
                print(f"Benchmarking {dataset} / {variant} ...")
                record = benchmark_variant(dataset, variant, config_path, args, dataset_cache, device)
                records.append(record)
        if not records:
            print("No benchmark results generated.")
            return
        df = pd.DataFrame(records).sort_values(["dataset", "variant"]).reset_index(drop=True)

    # Write metrics next to the plot outputs for reproducibility.
    df.to_csv(csv_path, index=False)

    plot_path = plot_pareto(df, output_dir)

    try:
        md_table = df.to_markdown(index=False)
    except Exception:
        md_table = df.to_string(index=False)

    md_lines = [
        "# Efficiency Benchmarks",
        "",
        f"Device: `{device}`",
        "",
        md_table,
        "",
        f"Plot saved to `{plot_path}`",
    ]
    (output_dir / "efficiency_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"✅ Wrote benchmark table to {csv_path}")
    print(f"✅ Pareto plot saved to {plot_path}")


if __name__ == "__main__":
    main()


