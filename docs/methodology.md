# Methodology

← [Back to README](../README.md)

## 1. Problem setting

Credit scoring is framed as a **binary classification** task:

- $Y \in \{0, 1\}$ — true outcome (1 = *good* payer, 0 = *bad* / default)
- $\hat{Y} \in \{0, 1\}$ — model decision (1 = loan **accepted**, 0 = rejected)
- $S = \hat{P}(Y = 1 \mid X)$ — predicted score
- $A \in \{0, 1\}$ — protected attribute **AGE**: $A = 0$ unprivileged (`< 25` years), $A = 1$ privileged (`≥ 25` years)

A loan is accepted when the score exceeds a cut-off $t$: $\hat{Y} = \mathbb{1}[S \ge t]$.
In this study the cut-off is **not** fixed at 0.5: it is chosen to **maximise expected profit** (EMP-optimal fraction, `EMPfrac`).

## 2. Group fairness criteria

All metrics are expressed as **absolute differences between the two groups**: 0 = perfectly fair, higher = more discrimination.

### 2.1 Independence (IND) — demographic parity

$$
\text{IND} = \left| P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1) \right|
$$

Requires equal acceptance rates. US practice (Equal Credit Opportunity Act, EEOC guidelines) relaxes it through the **80% rule**:

$$
\text{DI} = \frac{P(\hat{Y}=1 \mid A=0)}{P(\hat{Y}=1 \mid A=1)} \ge 0.8
$$

*Limitation:* it ignores whether applicants can actually repay — if default rates genuinely differ between groups, enforcing IND forces the lender to accept riskier loans or reject good ones.

### 2.2 Separation (SP) — equalized odds

$$
\text{SP} = \tfrac{1}{2}\Big(\left|\text{FPR}_{0} - \text{FPR}_{1}\right| + \left|\text{FNR}_{0} - \text{FNR}_{1}\right|\Big)
$$

with, per group $a$: $\text{FPR}_a = P(\hat{Y}=1 \mid Y=0, A=a)$ (bad payers accepted) and $\text{FNR}_a = P(\hat{Y}=0 \mid Y=1, A=a)$ (good payers rejected).

**Why it is the preferred criterion for credit scoring:**

- A **false positive** (accepting a defaulter) costs the *bank* money; a **false negative** (rejecting a good payer) harms the *applicant*. Separation treats both errors explicitly.
- It conditions on the true outcome, so it is compatible with a model that is accurate: it does not ask the bank to ignore genuine differences in risk, only to make **the same kind of mistakes at the same rate** for young and older applicants.

### 2.3 Sufficiency (SF) — predictive parity

$$
\text{SF} = \left| \text{PPV}_{0} - \text{PPV}_{1} \right|, \qquad \text{PPV}_a = P(Y=1 \mid \hat{Y}=1, A=a)
$$

Requires that an accepted applicant has the same probability of repaying in both groups. Well-calibrated models tend to satisfy it automatically, which makes it a weak lever for correcting *historical* discrimination embedded in the training labels.

### 2.4 Incompatibility

Outside of degenerate cases (equal base rates or a perfect classifier), the three criteria **cannot all be satisfied exactly at the same time**. The empirical question is therefore how close one can get to each, and at what cost.

## 3. Fairness processors

| Stage | Processor | Mechanism | Tuning parameter |
|-------|-----------|-----------|------------------|
| **Pre** | Reweighing (RW) | Weight $w(a, y) = \dfrac{P(A=a)\,P(Y=y)}{P(A=a, Y=y)}$ — expected over observed frequency of each group–label cell | — |
| **Pre** | Disparate Impact Remover (DIR) | Maps each feature, within each group, onto a common quantile distribution so $A$ is not recoverable | Repair level $\lambda \in \{0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$ |
| **In** | Prejudice Remover (PR) | Logistic loss + $\eta \cdot \text{PI}$, where PI (prejudice index) is the mutual information between $\hat{Y}$ and $A$ | Regularisation weight $\eta$ |
| **In** | Meta-Fair Classifier (MA) | Solves a constrained optimisation that bounds a group metric (e.g. PPV ratio for Sufficiency) | $\tau \in [0.05, 0.30]$ |
| **In** | Adversarial Debiasing (AD) | Predictor network minimises classification loss while *maximising* the loss of an adversary that predicts $A$ from the prediction; trained for 50 epochs, monitoring classifier and adversarial batch losses | Adversary weight |
| **Post** | Equalized Odds Post-processing (EOP) | Group-specific ROC curves → group-specific thresholds that equalise TPR and FPR | — |
| **Post** | Platt Scaling (PS) | A separate logistic regression per group maps raw scores to calibrated default probabilities | — |
| **Post** | Reject Option Classification (ROC) | In a "critical region" of width $\theta$ around the threshold, assign the favourable outcome to the unprivileged group and the unfavourable one to the privileged group | $\theta \in \{0.1, 0.2, 0.3\}$ |

Pre- and in-processors were implemented in Python with **IBM AIF360**; base learners, tuning and post-processors (EOP, PS) in **R / caret**.

### Base classifiers

| Model | caret method |
|-------|--------------|
| Logistic Regression | `glm` |
| Random Forest | `rf` |
| Extreme Gradient Boosting | `xgbTree` |
| Feed-forward Neural Network | `nnet` |

Hyper-parameter grids are defined in `98_param_grids.R`; parallel cross-validation and probability extraction in `97_caret_settings.R`.

## 4. Profit-based evaluation

Accuracy-type metrics treat all errors equally, but a bank does not. Two profit measures were used.

### 4.1 Expected Maximum Profit (EMP)

EMP (Verbraken et al., 2014; R package `EMP`) is the expected profit of a scorecard **at its profit-maximising cut-off**, relative to a no-model baseline. The loss given default (LGD) is treated as random, with a bimodal distribution:

| Parameter | Meaning | Default value |
|-----------|---------|---------------|
| $p_0$ | probability of full recovery (LGD = 0) | 0.55 |
| $p_1$ | probability of total loss (LGD = 1) | 0.10 |
| ROI | constant return on a granted good loan | 0.2644 |

`empCreditScoring()` also returns **EMPfrac**, the share of applicants to reject at the optimum — used as the decision threshold for all models.

### 4.2 Realised profit (`compute_profit()`)

On the test set, every accepted loan contributes:

- **+ ROI × loan amount** if the borrower repays (benefit $C$)
- **− LGD × loan amount** if the borrower defaults (cost $B$)

Rejected applicants contribute 0. Three summaries are reported: **total profit**, **average profit per applicant** and **profit per euro lent** (total profit ÷ total amount granted) — the last one makes datasets of different size comparable.

## 5. Model selection and evaluation protocol

1. **Train/validation** — 5-fold cross-validation inside the 70% training set.
2. **Tuning** — for each processor, the meta-parameter ($\lambda$, $\eta$, $\tau$, $\theta$, …) and the classifier hyper-parameters are chosen to **maximise EMP on validation data**.
3. **Testing** — the selected configuration is evaluated on the 30% hold-out set: AUC, EMP, profit, IND, SP, SF (`94_evaluate.R`).
4. **Comparison** — all results are merged (datasets × folds × base models × processors) and compared against the **unconstrained benchmark**.
5. **Pareto analysis** — in the (profit per euro lent, SP) space, only non-dominated solutions are kept (`nondominated`); the frontier gives the minimum profit loss needed to reach a target fairness level.
