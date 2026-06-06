# Final Camera-Ready Configurations

These are the paper-facing configurations used for the controlled experiments in the camera-ready manuscript.

- Paths are relative to this directory: datasets are expected in `../../datasets/` and embeddings in `../../embeddings/`.
- `*_cf_only.yaml` uses `fixed_fusion` with `lambda_fixed: 0.0`; this is the paper's latent-factor CF baseline, optimized with the pairwise BPR objective and without graph propagation.
- `*_branch_norm.yaml` uses `grad_balance.method: independent`; the paper names this operation **BranchNorm**.
- `*_gbaf_no_branch_norm.yaml`, `*_gbaf_no_confidence_gate.yaml`, and `*_gbaf_no_regularization.yaml` are the component ablations.
- Yelp and MIND use 60-epoch schedules for additive models. Their Concat-MLP controls use 30 epochs and patience 3, as reported in the paper.

Run five seeds: `42`, `123`, `999`, `2024`, and `2025`.
