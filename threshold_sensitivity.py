"""Fixed-threshold sensitivity diagnostics on pre-holdout chronological folds.

Thresholds are prespecified, not optimized on validation labels. The final 20%
test holdout is excluded. This is diagnostic only, not a production threshold.
Run: python threshold_sensitivity.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'secom_telemetry_modified.csv'
OUT=ROOT/'artifacts'/'threshold_sensitivity.json'
SEED=133

def make_x(df):
    cols=[c for c in df.columns if c.startswith('sensor_')]
    x=df[cols].copy(); legacy=x['sensor_0'].eq(0.0); codes=x['sensor_1'].where(legacy)
    x.loc[legacy,'sensor_0']=np.nan; x.loc[legacy,'sensor_1']=np.nan
    x['legacy_station']=legacy.astype(float)
    for code in (100.,200.,300.): x[f'legacy_sensor1_code_{int(code)}']=codes.eq(code).astype(float)
    return x,(df.label.to_numpy()==1).astype(int)

def main():
    df=pd.read_csv(DATA); df['timestamp']=pd.to_datetime(df['timestamp'],errors='raise')
    df=df.sort_values('timestamp',kind='stable').reset_index(drop=True)
    n=len(df); hold=int(n*.8); dev=df.iloc[:hold].reset_index(drop=True); X,y=make_x(dev)
    folds=[(int(n*.4),int(n*.55)),(int(n*.5),int(n*.65)),(int(n*.6),hold)]
    thresholds=[.10,.20,.30,.40,.50]
    rows=[]
    for fi,(te,ve) in enumerate(folds,1):
        tr=np.arange(te); va=np.arange(te, min(ve,len(dev)))
        usable=X.columns[X.iloc[tr].notna().any(axis=0)]
        imp=SimpleImputer(strategy='median')
        xt=imp.fit_transform(X.iloc[tr][usable]); xv=imp.transform(X.iloc[va][usable])
        model=RandomForestClassifier(n_estimators=300,min_samples_leaf=3,max_features=.5,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)
        model.fit(xt,y[tr]); score=model.predict_proba(xv)[:,1]
        for threshold in thresholds:
            pred=(score>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y[va],pred,labels=[0,1]).ravel()
            rows.append({'fold':fi,'threshold':threshold,'train_rows':len(tr),'validation_rows':len(va),'validation_prevalence':float(y[va].mean()),'average_precision':float(average_precision_score(y[va],score)),'precision':float(precision_score(y[va],pred,zero_division=0)),'recall':float(recall_score(y[va],pred,zero_division=0)),'f1':float(f1_score(y[va],pred,zero_division=0)),'predicted_positive_rate':float(pred.mean()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
    payload={'note':'Random Forest fixed-threshold sensitivity on pre-holdout chronological folds only. Thresholds were prespecified; final 20% holdout was not loaded for tuning. Results are diagnostic and do not establish a production operating point.','thresholds':thresholds,'results':rows}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2)); print('Wrote',OUT)
if __name__=='__main__': main()
