# Gradient-Scale Collapse in Additive Text-CF Fusion: Early Diagnosis and Minimal Repair

Official code repository for the ECML-PKDD 2026 Applied Data Science Track paper:

> **Gradient-Scale Collapse in Additive Text-CF Fusion: Early Diagnosis and Minimal Repair**  
> Yunan Zhang, Jingjing Fan, and Yanxiao Liu  
> ECML-PKDD 2026 Applied Data Science Track

## Overview

This repository contains the code, configurations, analysis scripts, and processed result tables used in the paper. The project studies **Gradient-Scale Collapse** in additive text-CF fusion for recommender systems and provides:

1. **Phenomenon identification**: additive text-CF fusion can collapse when text-branch scale dominates additive training and learned-gate updates.
2. **Architecture contrast**: concatenation fusion (Concat-MLP), which has no scalar gate, avoids this failure on collapse-prone datasets, showing that the issue is architecture-specific rather than caused by text quality alone.
3. **Early diagnosis protocol**: a lightweight diagnostic based on first-`K`-epoch gradient-ratio trajectories `r_t` between text and CF branches.
4. **Minimal repair**: branch-wise gradient normalization that controls gradient scale imbalance with minimal changes to an existing CF pipeline.

## Datasets

The paper evaluates six public datasets under a unified full-ranking protocol.

| Dataset | Domain | Users | Items | Train interactions |
|---------|--------|------:|------:|-------------------:|
| MovieLens-1M | Movies | 6,040 | 3,416 | 797,275 |
| Amazon Fine Food | Food reviews | 6,690 | 1,769 | 43,112 |
| Amazon Books | Book reviews | 12,129 | 14,710 | 291,042 |
| Amazon CDs | Music reviews | 15,592 | 16,184 | 414,228 |
| Yelp | Local business reviews | 99,165 | 56,696 | 2,173,523 |
| MIND | News | 50,000 | 7,713 | 185,581 |

Raw datasets are not redistributed in this repository. Use the scripts in `code/datasets/` to preprocess each public dataset. Place processed files (`train.pkl`, `val.pkl`, `test.pkl`, `stats.json`) under `datasets/<dataset_name>/`. Place pre-computed Sentence-BERT embeddings under `embeddings/<dataset_name>/all-MiniLM-L6-v2.pkl`.

## Fusion Architectures

| Method | Config key | Description |
|--------|------------|-------------|
| CF-only | `lightgcn_only` | LightGCN backbone without text features |
| Fixed `lambda=0.5` | `fixed_baseline` | Additive text-CF fusion with a fixed equal-weight text coefficient |
| Gated additive | `attention_pop_user` | Learned scalar gate conditioned on popularity and user/activity signals |
| GradNorm | `gradnorm_only` | Additive fusion with branch-wise gradient normalization |
| Concat-MLP | `concat_mlp` | Non-gated concatenation fusion used as the architecture control |

## Quick Start

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pre-compute Text Embeddings

```bash
python scripts/precompute_llm_embeddings.py --dataset ml-1m --encoder minilm --device cuda
```

### 3. Run a Single Experiment

```bash
python code/train_fusion.py \
    --config configs/movielens_attention_pop_user.yaml \
    --seed 42 \
    --output_dir results/ml1m/attention_pop_user/seed_42
```

### 4. Run Experiment Suites

```bash
bash scripts/run_concat_experiments.sh
bash scripts/run_extended_experiments.sh ml1m
```

### 5. Analysis

```bash
python analysis/significance_ecml.py
python analysis/plot_gradient_ratio.py
python analysis/diagnostic_boundaries.py
python analysis/build_ecml_main_table.py
```

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   ├── train_fusion.py
│   ├── models/
│   │   ├── attention_variants.py
│   │   └── gbaf.py
│   └── datasets/
│       ├── prepare_mind_dataset.py
│       ├── prepare_yelp_dataset.py
│       ├── prepare_amazon_books_dataset.py
│       ├── process_amazon_cds.py
│       └── ...
├── configs/
├── analysis/
├── scripts/
└── results/tables/
```

## Evaluation Protocol

- **Ranking protocol**: full ranking over all candidate items with training interactions masked.
- **Metrics**: Recall@10 and NDCG@10 in the main paper; additional metrics are produced by analysis scripts.
- **Seeds**: `42`, `123`, `999`, `2024`, and `2025`.
- **Early stopping**: validation Recall@10 with patience 10.
- **Text encoder**: frozen Sentence-BERT `all-MiniLM-L6-v2` embeddings unless otherwise specified.

## Key Configuration Options

Each YAML config controls the model, gradient balancing, and training settings:

```yaml
model:
  type: "attention_fusion"    # fixed_fusion | attention_fusion | concat_mlp_fusion
  embedding_dim: 64
  text_dim: 384

grad_balance:
  method: "none"              # none | gradnorm | independent | pcgrad
  log_grads: true
  target_ratio: 1.0

training:
  loss_type: "bpr"
  batch_size: 1024
  learning_rate: 0.003
  num_epochs: 50
  early_stopping_patience: 10
  negative_sampling: 4
```

## Reproducibility Notes

The repository includes the scripts used to aggregate tables, run significance tests, and generate diagnostic plots. Large raw datasets, processed interaction files, embeddings, checkpoints, and model weights are intentionally excluded from version control. The `.gitignore` file documents the expected local directories for those artifacts.

## Citation

```bibtex
@inproceedings{zhang2026gradientscale,
  title     = {Gradient-Scale Collapse in Additive Text-CF Fusion: Early Diagnosis and Minimal Repair},
  author    = {Zhang, Yunan and Fan, Jingjing and Liu, Yanxiao},
  booktitle = {Proceedings of ECML PKDD 2026},
  year      = {2026}
}
```

## License

This code is released under the MIT License. See `LICENSE` for details.
