# Regulatory context

← [Back to README](../README.md)

| Framework | Jurisdiction | Relevance to this project |
|-----------|--------------|---------------------------|
| **Equal Credit Opportunity Act (ECOA)** | United States | Prohibits credit discrimination on protected characteristics, including age. Disparate impact is commonly screened with the **80% (four-fifths) rule**, the practical relaxation of the Independence criterion. |
| **General Data Protection Regulation (GDPR)** | European Union | Limits the processing of personal data, gives individuals rights around automated decision-making (Art. 22) and requires transparency — relevant to how protected attributes may be used to *measure* and *correct* bias. |
| **Artificial Intelligence Act — Regulation (EU) 2024/1689** | European Union | Classifies AI systems used to evaluate **creditworthiness of natural persons** as **high-risk**, requiring data governance, bias monitoring, documentation and human oversight — i.e. **fairness by design**. |

## What the results mean for compliance

- **Measurable fairness:** the IND / SP / SF metrics and the Pareto frontier provide the kind of quantitative evidence that bias-monitoring obligations call for.
- **Proportionate cost:** reducing the Separation gap below 0.2 costs on average less than 5% of profit, so meaningful compliance does not threaten a lender's business model.
- **Deployability:** post-processing methods (e.g. Reject Option Classification) can be added to an existing, already-validated scorecard — a pragmatic route for institutions adapting legacy models.

## Related literature

- Kozodoi, Jacob & Lessmann (2022), *European Journal of Operational Research* — reference framework replicated in this thesis.
- Hurlin et al. (2024) and Valdrighi et al. (2025) — recent work on fairness in credit scoring discussed in the thesis.
