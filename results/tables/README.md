# Camera-ready result tables

These tables mirror the ECML-PKDD 2026 ADS camera-ready paper.
They replace earlier draft aggregation tables that used intermediate scales or
partial datasets.

- `dataset_characteristics.tex` / `camera_ready_dataset_characteristics.csv`: dataset statistics used in the paper.
- `extended_fusion_summary.tex` / `camera_ready_main_results.csv`: main Recall@10 and NDCG@10 table.
- `concat_mlp_5seed.tex`: final Concat-MLP architecture-control results.
- `diagnostic_boundaries.tex`: final early-diagnosis summary.
- `camera_ready_ablation_results.csv`: ablation values from the paper and supplement.

- `significance_camera_ready.csv`: paired five-seed tests reported in the supplement.
- `early_diagnosis_features.csv`: 30 dataset-seed cases at K in {3,5,10}.
- `early_diagnosis_metrics_by_k.csv` / `early_diagnosis_lodo_predictions.csv`: outputs reproduced by `analysis/early_diagnosis.py`.
