# Results

Generated outputs (not versioned):

- `intermediate/` — predictions of each fairness processor, e.g. `PR_predictions_*.csv`, `MA_predictions_*.csv`, plus base-model scores used by the post-processors.
- `final/` — aggregated metrics table (datasets × folds × base models × processors) with AUC, EMP, profit per euro lent, IND, SP, SF, and the Pareto-frontier plots produced by `code_14_results.R`.

A summary of the findings is in [`docs/results.md`](../docs/results.md).
