#!/usr/bin/env python3
"""Pre-compute item text embeddings with multiple LLM encoders."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta": "sentence-transformers/all-distilroberta-v1",
    "simcse": "princeton-nlp/sup-simcse-bert-base-uncased",
}


DATASETS: Dict[str, Dict[str, str]] = {
    "ml1m": {
        "text_path": "datasets/ml-1m/item_texts.pkl",
        "output_key": "ml-1m",
    },
    "amazonff": {
        "text_path": "datasets/amazon_fine_food_300k/item_texts.pkl",
        "output_key": "amazon_fine_food",
    },
    "yelp": {
        "text_path": "datasets/yelp/item_texts.pkl",
        "output_key": "yelp",
    },
    "mind": {
        "text_path": "datasets/mind/item_texts.pkl",
        "output_key": "mind",
    },
}


def resolve_model_name(alias: str) -> str:
    if alias in MODEL_ALIASES:
        return MODEL_ALIASES[alias]
    return alias


def load_texts(path: Path) -> Dict[int, str]:
    with path.open("rb") as f:
        return pickle.load(f)


def ensure_hf_endpoint(use_mirror: bool) -> None:
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    else:
        os.environ.pop("HF_ENDPOINT", None)


def load_model(model_name: str, device: str, cache_dir: Optional[Path]) -> SentenceTransformer:
    kwargs = {"device": device}
    if cache_dir is not None:
        kwargs["cache_folder"] = str(cache_dir)
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        model = SentenceTransformer(model_name, **kwargs)
        print(f"✅ Loaded {model_name} from local cache")
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        print(f"⬇️  Downloading {model_name} from HuggingFace...")
        model = SentenceTransformer(model_name, **kwargs)
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def compute_embeddings(model: SentenceTransformer, item_texts: Dict[int, str], batch_size: int) -> Dict[int, torch.Tensor]:
    item_ids = list(item_texts.keys())
    texts = [item_texts[i] for i in item_ids]
    embeddings: Dict[int, torch.Tensor] = {}

    for idx in tqdm(range(0, len(texts), batch_size), desc="Encoding", ncols=90):
        batch_ids = item_ids[idx: idx + batch_size]
        batch_texts = texts[idx: idx + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        for item_id, emb in zip(batch_ids, batch_embeddings):
            embeddings[item_id] = torch.from_numpy(emb).float()

    return embeddings


def save_embeddings(embeddings: Dict[int, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(embeddings, f)


def update_manifest(manifest_path: Path, record: Dict[str, Union[str, int, float]]) -> None:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = []
    manifest = [entry for entry in manifest if entry.get("output_path") != record["output_path"]]
    manifest.append(record)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings for recommender experiments")
    parser.add_argument("dataset", choices=DATASETS.keys(), help="Dataset key (ml1m/amazonff/yelp/mind)")
    parser.add_argument("encoder", help="Encoder alias or HF model name (minilm/mpnet/roberta/simcse)")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--output-dir", default="data/extended_fusion/embeddings", help="Root directory for outputs")
    parser.add_argument("--cache-dir", default=None, help="Optional HuggingFace cache directory")
    parser.add_argument("--mirror", action="store_true", help="Use HuggingFace mirror endpoint")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embedding file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_cfg = DATASETS[args.dataset]
    text_path = Path(dataset_cfg["text_path"])
    if not text_path.exists():
        raise FileNotFoundError(f"Item text file not found: {text_path}")

    encoder_name = resolve_model_name(args.encoder)
    encoder_slug = Path(encoder_name).name.replace("_", "-") if "/" in encoder_name else encoder_name
    output_root = Path(args.output_dir)
    output_path = output_root / dataset_cfg["output_key"] / f"{encoder_slug}.pkl"

    if output_path.exists() and not args.force:
        print(f"⚠️  Output already exists at {output_path}. Use --force to overwrite.")
        return

    ensure_hf_endpoint(args.mirror)
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    print("📥 Loading item texts...")
    item_texts = load_texts(text_path)
    print(f"   Loaded {len(item_texts)} items from {text_path}")

    model = load_model(encoder_name, args.device, cache_dir)

    print("🔨 Computing embeddings...")
    embeddings = compute_embeddings(model, item_texts, args.batch_size)
    example_dim = embeddings[next(iter(embeddings))].shape[0]
    print(f"   Generated embeddings for {len(embeddings)} items (dim={example_dim})")

    print(f"💾 Saving embeddings to {output_path}")
    save_embeddings(embeddings, output_path)

    manifest_path = output_root / "embedding_manifest.json"
    update_manifest(
        manifest_path,
        {
            "dataset": args.dataset,
            "encoder": encoder_name,
            "encoder_alias": args.encoder,
            "output_path": str(output_path),
            "num_items": len(embeddings),
            "dimension": example_dim,
            "batch_size": args.batch_size,
        },
    )
    print(f"✅ Manifest updated at {manifest_path}")


if __name__ == "__main__":
    main()


