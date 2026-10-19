"""Training/validation-only seed stability diagnostic for Wafer-GUARD.

The final chronological 20% holdout is deliberately excluded. This script is
for measuring optimization/threshold instability, not selecting a model from
final-test outcomes.
"""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'secom_telemetry_modified.csv'
OUT = ROOT / 'artifacts' / 'seed_stability.json'
SEEDS = [17, 41, 73, 1337, 2026]
EPOCHS = 20
BATCH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(DATA)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
sensors = [c for c in df.columns if c.startswith('sensor_')]
y = (df['label'].to_numpy() == 1).astype(np.int64)
n = len(df); a = int(n * .60); b = int(n * .80)
# Validation only: no rows from the final 20% are accessed below.
raw = df[sensors].copy()
legacy = raw['sensor_0'].eq(0.0)
X = raw.copy()
X['legacy_station'] = legacy.astype(float)
for code in (100.0, 200.0, 300.0):
    X[f'sensor_1_legacy_code_{int(code)}'] = (legacy & raw['sensor_1'].eq(code)).astype(float)
X.loc[legacy, 'sensor_0'] = np.nan
X.loc[legacy, 'sensor_1'] = np.nan
indicators = ['legacy_station', 'sensor_1_legacy_code_100', 'sensor_1_legacy_code_200', 'sensor_1_legacy_code_300']
continuous = sensors
imp = SimpleImputer(strategy='median', keep_empty_features=True)
scaler = StandardScaler()
Xtr = scaler.fit_transform(imp.fit_transform(X.iloc[:a][continuous])).astype('float32')
Xva = scaler.transform(imp.transform(X.iloc[a:b][continuous])).astype('float32')
Xtr = np.concatenate([Xtr, X.iloc[:a][indicators].to_numpy(dtype='float32')], axis=1)
Xva = np.concatenate([Xva, X.iloc[a:b][indicators].to_numpy(dtype='float32')], axis=1)
ytr, yva = y[:a], y[a:b]

class Model(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.10),
                                     nn.Linear(256,128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128,32))
        self.decoder = nn.Sequential(nn.Linear(32,128), nn.GELU(), nn.Linear(128,256), nn.GELU(), nn.Linear(256,d))
        self.head = nn.Linear(32,1)
    def forward(self,x):
        z=self.encoder(x); return self.decoder(z), self.head(z).squeeze(-1)

rows=[]
for seed in SEEDS:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=Model(Xtr.shape[1]).to(DEVICE)
    pos=float(ytr.sum()); neg=float(len(ytr)-ytr.sum())
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/max(pos,1)],dtype=torch.float32,device=DEVICE))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr.astype('float32'))),
                      batch_size=BATCH,shuffle=True,generator=gen)
    for _ in range(EPOCHS):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            rec,logits=model(xb)
            assert rec.shape == xb.shape and logits.shape == yb.shape
            loss=((rec-xb)**2).mean()+bce(logits,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    with torch.no_grad():
        _, logits=model(torch.from_numpy(Xva).to(DEVICE))
        p=torch.sigmoid(logits).cpu().numpy()
    ts=np.unique(np.concatenate(([0.0,0.5,1.0],p)))
    vals=[(float(t),float(f1_score(yva,p>=t,zero_division=0))) for t in ts]
    threshold,valf1=max(vals,key=lambda z:(z[1],-abs(z[0]-.5)))
    rows.append({'seed':seed,'validation_rows':int(len(yva),'__oops__') if False else int(len(yva)),
                 'validation_defects':int(yva.sum()),'validation_prevalence':float(yva.mean()),
                 'average_precision':float(average_precision_score(yva,p)),
                 'roc_auc':float(roc_auc_score(yva,p)) if len(np.unique(yva))==2 else None,
                 'best_validation_f1':valf1,'validation_threshold':threshold,
                 'precision_at_selected_threshold':float(precision_score(yva,p>=threshold,zero_division=0)),
                 'recall_at_selected_threshold':float(recall_score(yva,p>=threshold,zero_division=0)),
                 'predicted_positives_at_selected_threshold':int((p>=threshold).sum())})

aps=[r['average_precision'] for r in rows]
summary={'purpose':'optimization and threshold stability only; final test holdout not accessed',
         'device':str(DEVICE),'seeds':SEEDS,'epochs_per_seed':EPOCHS,
         'train_rows':int(a),'validation_rows':int(b-a),'validation_defects':int(yva.sum()),
         'validation_prevalence':float(yva.mean()),'runs':rows,
         'summary':{'average_precision_min':float(min(aps)),'average_precision_median':float(np.median(aps)),
                    'average_precision_max':float(max(aps)),
                    'best_validation_f1_min':float(min(r['best_validation_f1'] for r in rows)),
                    'best_validation_f1_max':float(max(r['best_validation_f1'] for r in rows)),
                    'threshold_min':float(min(r['validation_threshold'] for r in rows)),
                    'threshold_max':float(max(r['validation_threshold'] for r in rows))}}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
