# Data

Raw datasets are **not versioned** in this repository (licence / size). Download them into `data/raw/`; the partitioning notebook (`code_01`) writes the prepared splits into `data/prepared/`.

| Dataset | Observations | Availability | Source |
|---------|-------------:|--------------|--------|
| German Credit | 1,000 | Public | [UCI – Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Taiwan Credit | 23,531 (after cleaning) | Public | [UCI – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| GMSC | 150,000 | Public | [Kaggle – Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) |
| Home Credit | 307,511 | Public | [Kaggle – Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) |
| PAKDD | 50,000 | Competition data | PAKDD 2010 Data Mining Competition |
| UK | 30,000 | Not public | — |
| Bene | 3,123 | Not public | — |

## Expected layout

```
data/
├── raw/        # original files, one per dataset
└── prepared/   # produced by code_01_partitioning.ipynb
                # train / validation / test splits and 5 CV folds, as .csv (R) and .pkl (Python)
```

## Preparation applied in `code_01`

- Protected attribute `AGE` binarised: `< 25` → 0 (unprivileged), `≥ 25` → 1 (privileged)
- Target: 1 = good payer, 0 = default
- 70% training / 30% test split, 5-fold cross-validation inside training
- `MaxAbsScaler` feature scaling to [-1, 1]
