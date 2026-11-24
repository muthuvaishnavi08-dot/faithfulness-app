"""Train the small fusion model used by the app from the notebook's `dataset_with_scores.csv`.

Usage:  python train_fusion.py path/to/dataset_with_scores.csv
Creates models/fusion_full.joblib (rule + NLI + judge) and models/fusion_nojudge.joblib (rule + NLI).
Note: trained on ALL rows (English + Hindi); the AUROC printed here is on the training data (optimistic).
The honest, cross-validated numbers are the ones in the notebook.
"""
import os, sys
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

csv = sys.argv[1] if len(sys.argv) > 1 else "dataset_with_scores.csv"
df = pd.read_csv(csv)
need = {"s_rule", "s_nli", "s_judge", "label"}
assert need <= set(df.columns), f"missing columns: {need - set(df.columns)}"
os.makedirs("models", exist_ok=True)

for name, feats in {"fusion_full": ["s_rule", "s_nli", "s_judge"], "fusion_nojudge": ["s_rule", "s_nli"]}.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(df[feats].values, df["label"])
    joblib.dump(model, f"models/{name}.joblib")
    auc = roc_auc_score(df["label"], model.predict_proba(df[feats].values)[:, 1])
    coefs = {f: round(float(c), 3) for f, c in zip(feats, model[-1].coef_[0])}
    print(f"{name}: saved | train AUROC {auc:.3f} (optimistic) | standardised coefficients {coefs}")
