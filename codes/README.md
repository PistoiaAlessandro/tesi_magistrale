# Experiment scripts

Run in numerical order. Full description in [`docs/pipeline.md`](../docs/pipeline.md).

| # | Script | Language | Phase | Content |
|---|--------|----------|-------|---------|
| 00 | `code_00_working_paths.py` / `.R` | Py / R | Setup | Project paths and folder creation |
| 01 | `code_01_partitioning.ipynb` | Python | Partitioning | AGE groups, 70/30 split, 5 folds, MaxAbsScaler |
| 02 | `code_02_preprocess1.ipynb` | Python | Pre-processing | Reweighing, Disparate Impact Remover (λ = 0.5…1.0) |
| 03 | `code_03_preprocess2.R` | R | Pre-processing | Train glm / rf / xgbTree / nnet on RW & DIR data |
| 04 | `code_04_preprocess3.R` | R | Pre-processing | Tune λ on validation EMP, save results |
| 05 | `code_05_inprocess1.ipynb` | Python | In-processing | Prejudice Remover (η), Meta-Fair Classifier (τ) |
| 06 | `code_06_inprocess2.ipynb` | Python | In-processing | Adversarial Debiasing (50 epochs) |
| 07 | `code_07_inprocess3.R` | R | In-processing | Tuning, EMPfrac cut-off, evaluation |
| 08 | `code_08_postprocess1.R` | R | Post-processing | Base-model scores |
| 09 | `code_09_postprocess2.R` | R | Post-processing | Equalized Odds Post-processing |
| 10 | `code_10_postprocess3.R` | R | Post-processing | Platt Scaling by group |
| 11 | `code_11_postprocess4.ipynb` | Python | Post-processing | Reject Option Classification (θ = 0.1, 0.2, 0.3) |
| 12 | `code_12_postprocess5.R` | R | Post-processing | Evaluation of ROC-corrected predictions |
| 13 | `code_13_unconstrained.R` | R | Benchmark | Unconstrained profit-maximising models |
| 14 | `code_14_results.R` | R | Results | Merge all results, Pareto frontier |

> The scripts are based on the open-source framework of Kozodoi, Jacob & Lessmann (2022) — [kozodoi/Fair_Credit_Scoring](https://github.com/kozodoi/Fair_Credit_Scoring) — and were run and adapted for the thesis experiments.
