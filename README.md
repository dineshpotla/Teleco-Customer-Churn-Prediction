# Telco Customer Churn Prediction

End-to-end churn modeling on the public Telco Customer Churn dataset.

## Contents
- `notebooks/telco_churn.ipynb`: EDA, preprocessing, multiple models (logistic, RF, HistGB, CatBoost), feature engineering, threshold tuning, and saved artifacts.
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset copy.
- `models/`: Best model artifact (`best_model_catboost_fe.joblib`) and threshold metadata.
- `requirements.txt`: Python dependencies.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
# run notebook (or execute headless)
/Users/dinesh/Library/Python/3.9/bin/jupyter-nbconvert --to notebook --inplace --execute notebooks/telco_churn.ipynb
```

## Model
- Best model: CatBoost with feature engineering and tuned decision threshold (~0.46).
- Recent metrics (80/20 stratified split): accuracy ~0.796, ROC AUC ~0.842, PR AUC ~0.652.

## Notes
- Dataset is public (Kaggle Telco Customer Churn). Included here for convenience.
- Re-run the notebook to refresh artifacts or experiment with thresholds/feature tweaks.
