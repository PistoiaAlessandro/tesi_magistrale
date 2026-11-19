# R helper functions

Sourced by the R scripts in `codes/`.

| File | Function | Purpose |
|------|----------|---------|
| `93_*` | `PlotParetoFront()` | Profit vs. fairness Pareto frontier (non-dominated points) |
| `94_evaluate.R` | `evaluate()` | AUC, EMP, IND, SP, SF in one call |
| `95_fairness_metrics.R` | `fairness_metric()` | Group disparity for a chosen criterion |
| `96_emp_summary.R` | `emp_summary()` | Wrapper around `EMP::empCreditScoring()` (EMP, EMPfrac) |
| `97_caret_settings.R` | `caret_settings()` | caret `trainControl`: parallel CV, class probabilities |
| `98_param_grids.R` | `param_grids()` | Hyper-parameter grids for glm / rf / xgbTree / nnet |
| `99_compute_profit.R` | `compute_profit()` | Total profit, average profit, profit per euro lent |

A tested Python counterpart of the fairness and profit metrics is in [`src/fairness_metrics.py`](../src/fairness_metrics.py).
