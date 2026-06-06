# Camera-ready result tables

These CSV files contain the aggregate values used in the ECML-PKDD 2026 ADS paper and the inputs/outputs of the included diagnostic analysis.

- `camera_ready_dataset_characteristics.csv`: dataset statistics used in the paper.
- `camera_ready_main_results.csv`: main Recall@10 and NDCG@10 results.
- `camera_ready_ablation_results.csv`: ablation values from the paper and supplement.
- `significance_camera_ready.csv`: paired five-seed tests reported in the supplement.
- `early_diagnosis_features.csv`: 30 dataset-seed cases at K in {3,5,10}.
- `early_diagnosis_metrics_by_k.csv` / `early_diagnosis_lodo_predictions.csv`: outputs reproduced by `analysis/early_diagnosis.py`.
