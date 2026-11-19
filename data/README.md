# Data

Raw datasets are **not versioned** in this repository (licence / size). The thesis experiments use **German Credit, Taiwan Credit, UK and PAKDD**; the other three belong to the reference framework of Kozodoi et al. (2022). Download them into `data/raw/`; the partitioning notebook (`code_01`) writes the prepared splits into `data/prepared/`.

| Dataset | Observations | Used in thesis | Source |
|---------|-------------:|:--------------:|--------|
| German Credit | 1,000 | ✅ | [UCI – Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Taiwan Credit | 23,531 (after cleaning) | ✅ | [UCI – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| GMSC | 150,000 | — | [Kaggle – Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) |
| Home Credit | 307,511 | — | [Kaggle – Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) |
| PAKDD | 50,000 | ✅ | PAKDD 2010 Data Mining Competition |
| UK | 30,000 | ✅ | Not redistributed here |
| Bene | 3,123 | — | Not redistributed here |

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
