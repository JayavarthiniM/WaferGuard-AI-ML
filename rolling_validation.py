"""Expanding-window validation diagnostics for Wafer-GUARD.

This script deliberately excludes the final 20% holdout from model comparison.
All learned preprocessing is fit on each fold's training prefix only. Results are
diagnostic, not a production guarantee or a replacement for the required PyTorch
supervised-autoencoder notebook.

Run: python rolling_validation.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'secom_telemetry_modified.csv'
OUT = ROOT / 'artifacts' / 'rolling_validation_metrics.json'
SEED = 133

def features_for(df):
    cols = [c for c in df.columns if c.startswith('sensor_')]
    x = df[cols].copy()
    legacy = x['sensor_0'].eq(0.0)
    codes = x['sensor_1'].where(legacy)
    x.loc[legacy, 'sensor_0'] = np.nan
    x.loc[legacy, 'sensor_1'] = np.nan
    x['legacy_station'] = legacy.astype(float)
    for code in (100.0, 200.0, 300.0):
        x[f'legacy_sensor1_code_{int(code)}'] = codes.eq(code).astype(float)
    return x, (df['label'].to_numpy() == 1).astype(int)

def build_models():
    return {
      'logistic_balanced': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=.03, class_weight='balanced', max_iter=2000, random_state=SEED)),
      'extra_trees': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features=.5, class_weight='balanced', random_state=SEED, n_jobs=-1)),
      'random_forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=.5, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1)),
    }

def main():
    df = pd.read_csv(DATA)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    # Final 20% is embargoed from all model comparisons; three expanding-window folds in first 80%.
    n = len(df); holdout_start = int(n*.80); development = df.iloc[:holdout_start].reset_index(drop=True)
    X, y = features_for(development)
    # Train prefixes and immediately following validation windows, all before final holdout.
    folds = [(int(n*.40), int(n*.55)), (int(n*.50), int(n*.65)), (int(n*.60), holdout_start)]
    records=[]
    for fi,(train_end,val_end) in enumerate(folds,1):
        if val_end > len(development): val_end=len(development)
        tr=np.arange(0,train_end); va=np.arange(train_end,val_end)
        if len(np.unique(y[va])) < 2:
            print(f'SKIP fold {fi}: validation has one class'); continue
        # Remove features that are entirely missing in this training prefix.
        # This decision uses training rows only and avoids silently changing feature
        # dimensions inside SimpleImputer. The same selected columns are applied to
        # the later validation window.
        usable_cols = X.columns[X.iloc[tr].notna().any(axis=0)]
        dropped_all_missing = int(X.shape[1] - len(usable_cols))
        for name,model in build_models().items():
            model.fit(X.iloc[tr][usable_cols], y[tr])
            if hasattr(model, 'predict_proba'):
                score=model.predict_proba(X.iloc[va][usable_cols])[:,1]
            else:
                score=model.decision_function(X.iloc[va][usable_cols])
            # Ranking metrics only; do not select a threshold on these diagnostic folds.
            row={'fold':fi,'model':name,'train_rows':len(tr),'validation_rows':len(va),
                 'train_prevalence':float(y[tr].mean()),'validation_prevalence':float(y[va].mean()),
                 'all_missing_features_dropped_from_training_prefix':dropped_all_missing,
                 'average_precision':float(average_precision_score(y[va],score)),
                 'roc_auc':float(roc_auc_score(y[va],score))}
            records.append(row)
            print(row)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({'note':'Expanding-window diagnostics only; final 20% holdout excluded from model selection.','folds':records},indent=2))
    print(f'Wrote {OUT}')
if __name__=='__main__': main()
