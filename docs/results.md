# Results

← [Back to README](../README.md)

## Experimental grid

| Dimension | Levels |
|-----------|--------|
| Datasets | German Credit, Taiwan Credit, UK, PAKDD |
| Resampling | 5 cross-validation folds + 30% hold-out test set |
| Base classifiers | Logistic Regression, Random Forest, XGBoost, Neural Network |
| Fairness processors | RW, DIR, PR, MA, AD, EOP, PS, ROC (+ unconstrained benchmark) |
| Metrics | AUC · EMP · profit per euro lent · IND · SP · SF |

## 1. Relationships between metrics

| Pair | Correlation | Interpretation |
|------|------------:|----------------|
| AUC ↔ Profit | **≈ 0.80** | Better-ranking models generally earn more, but the link is not perfect — profit must be measured directly. |
| Independence ↔ Separation | **≈ 0.95** | Reducing the gap in acceptance rates and the gap in error rates go almost hand in hand. |

Several fairness criteria can therefore be **approximately satisfied at the same time**, even though they are mathematically incompatible in their exact form.

## 2. Choice of criterion

**Separation** emerged as the most adequate criterion for lending:

- it accounts for the **asymmetric cost** of errors (a defaulted loan vs. a wrongly rejected good customer);
- it does not require ignoring real differences in creditworthiness, unlike Independence;
- unlike Sufficiency, it is not trivially met by any well-calibrated model, so it actually constrains historical bias.

## 3. The cost of fairness (Pareto analysis)

The Pareto frontier of *profit per euro lent* vs. *Separation* shows a **convex trade-off**:

| Fairness target (SP gap) | Average profit sacrifice |
|--------------------------|--------------------------|
| below **0.2** | **< 5%** (less than **€0.01 per euro lent**) |
| **0** (no discrimination) | **> 35%** |

The first large reductions in discrimination are cheap; the last ones are very expensive. A regulator or a bank can choose a point on the frontier that matches its tolerance.

## 4. Comparing processors

| Family | Processors | Practical assessment |
|--------|-----------|----------------------|
| Pre-processing | RW, DIR | Model-agnostic and simple, but act only indirectly on the final decisions. |
| **In-processing** | **PR**, MA, AD | **Prejudice Remover** offers one of the best profit–fairness balances; AD is powerful but harder to train and tune. |
| **Post-processing** | EOP, PS, **ROC** | **Reject Option Classification** achieves strong fairness gains at limited cost and can be applied to an existing scorecard without retraining. |

**Recommendation:** Prejudice Remover (when the model can be retrained) and Reject Option Classification (when it cannot) are the most practical options for a lender.

## 5. Limitations and future work

- A single binary protected attribute (AGE, cut at 25 years); intersectional groups were not analysed.
- Profit parameters (ROI, LGD distribution) are fixed at standard literature values.
- Static evaluation — no feedback loops from past lending decisions.
- Future directions discussed in the thesis: recent contributions by Hurlin et al. (2024) and Valdrighi et al. (2025), and the operational requirements of the EU AI Act for high-risk systems.
