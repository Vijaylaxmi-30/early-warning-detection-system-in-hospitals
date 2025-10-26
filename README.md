## ML Early Warning System (EWS)

An end-to-end, reproducible demonstration of an Early Warning System for in‑hospital deterioration risk using synthetic ICU-like vitals. The project:

- Generates a synthetic cohort with vitals time series
- Computes MEWS (Modified Early Warning Score) features
- Engineers statistical and random-projection temporal features
- Trains multiple models (XGBoost or fallback GBM, Logistic Regression with L2 "Ridge", Random Forest)
- Benchmarks against a MEWS-only baseline and prints a concise report

### Repository structure
- `early_warning_system.py`: Core library for data gen, feature engineering, training, and reporting
- `ML_Early_Warning_System_Demo.ipynb`: Walkthrough notebook (EDA + training demo)

### Requirements
Python 3.9+ (tested with CPython 3.12). Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost jupyter matplotlib seaborn
```

Notes:
- `xgboost` is optional; if it cannot be imported, the code falls back to `sklearn.ensemble.GradientBoostingClassifier` automatically.
- Jupyter, matplotlib, and seaborn are for the notebook; the core script runs without them.

### Quickstart (script usage)
Run a full synthetic experiment from Python:

```python
from early_warning_system import create_sample_data, EarlyWarningSystem

# 1) Create synthetic cohort and vitals (1000 patients by default)
df_cohort, df_vitals = create_sample_data(n_patients=1000, random_state=42)

# 2) Build features
ews = EarlyWarningSystem(random_state=42)
X, y = ews.prepare_training_data(df_cohort, df_vitals)

# 3) Train and evaluate models
results = ews.train_models(X, y)

# 4) Print a performance report (AUCs, deltas vs MEWS)
ews.generate_report()
```

Expected console output includes dataset/feature counts, AUC for each model, and improvement over the MEWS baseline when available.

### Running the notebook
```bash
jupyter notebook ML_Early_Warning_System_Demo.ipynb
```
Execute cells top‑down to reproduce the figures and results interactively.

### How it works (high level)
- **MEWS features**: Per‑timepoint MEWS computed from heart rate, SBP, respiratory rate, and temperature; aggregated features include mean, max, final, and std.
- **Statistical features**: For each vital (heart_rate, sbp, dbp, resp_rate, temperature, spo2) compute mean, std, min, max, range, and temporal slope (via linear fit).
- **Random‑projection temporal features**: The normalized statistical feature vector is projected using a fixed random matrix to create 900 "conv-like" features capturing multi‑scale interactions at low cost.
- **Models**: XGBoost (if available) or Gradient Boosting, Logistic Regression (L2), and Random Forest.
- **Metric**: ROC AUC on a stratified holdout split.

### Reproducibility
Set `random_state` (default 42) when constructing `EarlyWarningSystem` and when generating data for repeatable splits, features, and results.

### Extending
- Replace `create_sample_data` with your real cohort and multivariate vitals time series (must include `subject_id` and `hours_from_icu_admit`).
- Add domain features (e.g., lab trends, comorbidities) alongside the provided statistical and MEWS features.
- Swap/extend models or perform hyperparameter tuning (e.g., `sklearn.model_selection.GridSearchCV`).

### Troubleshooting
- If you see only MEWS deletion or `__pycache__` noise in Git, add ignores:
  - `.gitignore` entries: `__pycache__/` and `*.pyc`
- If XGBoost is unavailable, ensure `xgboost` is installed, or proceed with the GBM fallback.

