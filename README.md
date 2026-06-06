# Gradient-Scale Collapse in Additive Text-CF Fusion

Official artifact for the ECML-PKDD 2026 Applied Data Science Track paper:

> **Gradient-Scale Collapse in Additive Text-CF Fusion: Early Diagnosis and Minimal Repair**  
> Yunan Zhang, Jingjing Fan, and Yanxiao Liu

## What This Repository Contains

This repository is a minimal reproducibility artifact containing the executable training and evaluation code, final configurations, analysis scripts, and aggregate result tables used for the paper. It studies a failure mode of additive text-CF fusion in which branch-scale imbalance can make text injection harmful, then evaluates an early diagnostic and a minimal branch-normalization repair.

The executable CF backbone is a latent-factor model that learns user and item embeddings with the pairwise BPR ranking objective. It does not perform graph propagation.

## Main Components

1. **Fixed additive fusion:** `e_i^CF + 0.5 e_i^text`.
2. **GBAF:** an item-conditioned scalar gate using normalized popularity, CF confidence, and their interaction.
3. **BranchNorm:** independent unit-norm rescaling of the CF-embedding and text-projection parameter gradients. The implementation key is `grad_balance.method: independent`.
4. **Concat-MLP:** a non-gated architecture control.
5. **Early diagnosis:** a leave-one-dataset-out rule based on first-`K`-epoch gradient-ratio trajectories and validation gaps.

## Datasets

| Dataset | Domain | Users | Items | Train interactions |
|---|---|---:|---:|---:|
| MovieLens-1M | Movies | 6,040 | 3,416 | 797,275 |
| Amazon Fine Food | Food reviews | 6,690 | 1,769 | 43,112 |
| Amazon Books | Book reviews | 12,129 | 14,710 | 291,042 |
| Amazon CDs | Music reviews | 15,592 | 16,184 | 414,228 |
| Yelp | Local business reviews | 99,165 | 56,696 | 2,173,523 |
| MIND | News | 50,000 | 7,713 | 185,581 |

Raw datasets are not redistributed. Place processed `train.pkl`, `val.pkl`, `test.pkl`, and `stats.json` files under `datasets/<dataset>/`. Place frozen Sentence-BERT embeddings under `embeddings/<dataset>/all-MiniLM-L6-v2.pkl`.

## Final Configurations

Paper-facing configurations are in [`configs/final`](configs/final). These are the only released experiment configurations and use portable paths and paper terminology.

| Paper method | Configuration pattern |
|---|---|
| CF-only latent-factor model | `configs/final/*_cf_only.yaml` |
| Fixed `lambda=0.5` | `configs/final/*_fixed.yaml` |
| BranchNorm-only | `configs/final/*_branch_norm.yaml` |
| GBAF | `configs/final/*_gbaf.yaml` |
| Concat-MLP | `configs/final/*_concat_mlp.yaml` |

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After preparing a dataset and its text embeddings:

```bash
python code/train_fusion.py \
  --config configs/final/yelp_gbaf.yaml \
  --seed 42 \
  --output_dir results/yelp/gbaf/seed_42
```

Run the five paper seeds (`42`, `123`, `999`, `2024`, `2025`) for each method. The paper uses full ranking over all items with training interactions masked, reporting Recall@10 and NDCG@10.

## Verified Results and Analysis

- `results/tables/camera_ready_main_results.csv`: controlled six-dataset means reported in the paper.
- `results/tables/camera_ready_ablation_results.csv`: full ablation and repair aggregates.
- `results/tables/significance_camera_ready.csv`: paired five-seed tests reported in the supplement.
- `analysis/build_ecml_main_table.py`: exports a LaTeX table from the verified controlled CSV.
- `analysis/significance_ecml.py`: recomputes paired tests when raw per-seed JSON files are available.
- `analysis/early_diagnosis.py`: reproduces the LODO diagnosis metrics and scheduled-epoch savings from the included 30-case feature table.
- `scripts/run_camera_ready_experiments.sh`: runs the controlled final configurations over the five paper seeds.
- `scripts/run_all_precomputes.sh`: creates MiniLM embeddings for all six datasets.

The aggregate values in `results/tables/` were cross-checked against the per-seed experiment outputs used to build the camera-ready paper. Large datasets, embeddings, raw run logs, checkpoints, and model weights are excluded from Git.

## Repository Layout

```text
code/                 Training and model implementation
configs/final/        Authoritative camera-ready experiment configs
analysis/             Aggregation, significance, and diagnostic scripts
scripts/              Experiment and preprocessing helpers
results/tables/       Verified aggregate paper tables
```

## Citation

```bibtex
@inproceedings{zhang2026gradientscale,
  title     = {Gradient-Scale Collapse in Additive Text-CF Fusion: Early Diagnosis and Minimal Repair},
  author    = {Zhang, Yunan and Fan, Jingjing and Liu, Yanxiao},
  booktitle = {Machine Learning and Knowledge Discovery in Databases: Applied Data Science Track},
  year      = {2026},
  publisher = {Springer}
}
```

## License

MIT License. See [`LICENSE`](LICENSE).
