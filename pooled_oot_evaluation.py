"""Pooled out-of-time evaluation for the repaired supervised autoencoder.

Uses expanding training prefixes and the immediately following validation/test windows.
The final 20% chronological holdout is never used here. Thresholds are selected on each
fold's validation window only; pooled ranking metrics and alert-budget metrics are reported
without selecting anything from pooled test labels.

Run: python pooled_oot_evaluation.py --data secom_telemetry_modified.csv
"""
from pathlib import Path
import argparse, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

HP = dict(batch_size=64, epochs=30, lr=1e-3, weight_decay=1e-5, hidden=256, latent=32, lambda_clf=1.0)
ROOT = Path(__file__).resolve().parent

def make_features(df):
    cols = [c for c in df.columns if c.startswith('sensor_')]
    raw = df[cols].copy()
    legacy = raw['sensor_0'].eq(0.0).to_numpy()
    X = raw.copy()
    X.loc[legacy, 'sensor_0'] = np.nan
    X.loc[legacy, 'sensor_1'] = np.nan
    indicators = np.column_stack([legacy] + [(legacy & raw['sensor_1'].eq(c).to_numpy()) for c in (100.0,200.0,300.0)]).astype(np.float32)
    return X, indicators, legacy, (df['label'].to_numpy() == 1).astype(np.int64), cols

def train_predict(X, indicators, legacy, y, cols, train_idx, val_idx, test_idx, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    A = scaler.fit_transform(imp.fit_transform(X.iloc[train_idx][cols])).astype(np.float32)
    V = scaler.transform(imp.transform(X.iloc[val_idx][cols])).astype(np.float32)
    T = scaler.transform(imp.transform(X.iloc[test_idx][cols])).astype(np.float32)
    # Strict PS-L: neutral slots and mask reconstruction targets on legacy rows.
    parts=[]; masks=[]
    for arr, idx in ((A,train_idx),(V,val_idx),(T,test_idx)):
        leg=legacy[idx]
        arr[leg,0]=0.; arr[leg,1]=0.
        ind=indicators[idx]
        parts.append(np.concatenate([arr,ind],axis=1).astype(np.float32))
        m=np.ones_like(parts[-1],dtype=np.float32); m[leg,0]=0.; m[leg,1]=0.
        masks.append(m)
    A,V,T=parts; MA,MV,MT=masks
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class SAE(nn.Module):
        def __init__(self, f):
            super().__init__(); h=HP['hidden']; z=HP['latent']
            self.enc=nn.Sequential(nn.Linear(f,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(.10),nn.Linear(h,h//2),nn.LayerNorm(h//2),nn.GELU(),nn.Linear(h//2,z))
            self.dec=nn.Sequential(nn.Linear(z,h//2),nn.GELU(),nn.Linear(h//2,h),nn.GELU(),nn.Linear(h,f))
            self.head=nn.Linear(z,1)
        def forward(self,x):
            z=self.enc(x); return self.dec(z),self.head(z).squeeze(-1)
    model=SAE(A.shape[1]).to(device)
    yt=torch.tensor(y[train_idx],dtype=torch.float32,device=device)
    pos=float(yt.sum()); pw=torch.tensor([(len(yt)-pos)/max(pos,1.)],dtype=torch.float32,device=device)
    bce=nn.BCEWithLogitsLoss(pos_weight=pw)
    opt=torch.optim.AdamW(model.parameters(),lr=HP['lr'],weight_decay=HP['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs'])
    g=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(A),yt.cpu(),torch.from_numpy(MA)),batch_size=HP['batch_size'],shuffle=True,generator=g)
    for _ in range(HP['epochs']):
        model.train()
        for xb,yb,mb in loader:
            xb=xb.to(device); yb=yb.to(device); mb=mb.to(device)
            rec,logit=model(xb)
            assert rec.shape==xb.shape and logit.shape==yb.shape
            recon=(((rec-xb)**2)*mb).sum()/mb.sum().clamp(min=1.)
            loss=recon+HP['lambda_clf']*bce(logit,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        sched.step()
    def prob(arr):
        model.eval()
        with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(arr).to(device))[1]).cpu().numpy()
    pv=prob(V); pt=prob(T); yv=y[val_idx]; ytst=y[test_idx]
    candidates=np.unique(np.r_[0.,.5,1.,pv])
    threshold=max(candidates,key=lambda t:(f1_score(yv,(pv>=t).astype(int),zero_division=0),-abs(t-.5)))
    return pt,float(threshold),float(f1_score(yv,(pv>=threshold).astype(int),zero_division=0))

def block_bootstrap(y, p, reps=2000, block=25, seed=20261018):
    rng=np.random.default_rng(seed); n=len(y); starts=np.arange(0,n,block); chunks=[np.arange(s,min(s+block,n)) for s in starts]
    vals={'average_precision':[],'roc_auc':[]}
    for _ in range(reps):
        ids=np.concatenate([chunks[i] for i in rng.integers(0,len(chunks),size=len(chunks))])[:n]
        if len(np.unique(y[ids]))<2: continue
        vals['average_precision'].append(float(average_precision_score(y[ids],p[ids])))
        vals['roc_auc'].append(float(roc_auc_score(y[ids],p[ids])))
    return {k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in vals.items() if v}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='secom_telemetry_modified.csv'); ap.add_argument('--out',default='artifacts/pooled_oot_metrics.json'); ap.add_argument('--epochs',type=int,default=HP['epochs']); a=ap.parse_args()
    HP['epochs']=a.epochs
    df=pd.read_csv(a.data); df['timestamp']=pd.to_datetime(df['timestamp'],errors='raise'); df=df.sort_values('timestamp',kind='stable').reset_index(drop=True)
    X,ind,legacy,y,cols=make_features(df); n=len(df)
    # Three disjoint test windows wholly inside the first 80%; final 20% stays untouched.
    folds=[(int(.40*n),int(.50*n),int(.60*n)),(int(.50*n),int(.60*n),int(.70*n)),(int(.60*n),int(.70*n),int(.80*n))]
    all_y=[]; all_p=[]; fold_rows=[]
    for i,(tr_end,va_end,te_end) in enumerate(folds,1):
        tr=np.arange(0,tr_end); va=np.arange(tr_end,va_end); te=np.arange(va_end,te_end)
        p,thr,vf1=train_predict(X,ind,legacy,y,cols,tr,va,te,seed=1337+17*i)
        yy=y[te]; all_y.extend(yy.tolist()); all_p.extend(p.tolist())
        row={'fold':i,'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'train_prevalence':float(y[tr].mean()),'validation_prevalence':float(y[va].mean()),'test_prevalence':float(yy.mean()),'validation_selected_threshold':thr,'validation_f1':vf1,'test_average_precision':float(average_precision_score(yy,p)),'test_roc_auc':float(roc_auc_score(yy,p)) if len(np.unique(yy))>1 else None}
        fold_rows.append(row); print(json.dumps(row))
    yy=np.asarray(all_y,dtype=int); pp=np.asarray(all_p,dtype=float)
    order=np.argsort(-pp,kind='stable'); topk={}
    for frac in (.05,.10):
        k=max(1,int(np.ceil(frac*len(yy)))); sel=order[:k]; tp=int(yy[sel].sum())
        topk[f'top_{int(frac*100)}pct']={'k':k,'precision_at_k':tp/k,'recall_at_k':tp/max(1,int(yy.sum())),'true_positives':tp}
    out={'protocol':'Pooled out-of-time rolling-origin evaluation; folds are 40/50/60%, 50/60/70%, 60/70/80% train/validation/test endpoints; final chronological 20% holdout excluded. Preprocessing fitted on each training prefix only; thresholds selected on each validation window only.','rows':len(yy),'defects':int(yy.sum()),'prevalence':float(yy.mean()),'average_precision':float(average_precision_score(yy,pp)),'roc_auc':float(roc_auc_score(yy,pp)),'top_k_alert_metrics':topk,'contiguous_block_bootstrap_95pct_ci_block25':block_bootstrap(yy,pp),'bootstrap_note':'Resamples contiguous chunks of 25 ordered observations; descriptive interval, approximate block bootstrap.','folds':fold_rows}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print('\nPOOLED:',json.dumps(out,indent=2)); print('Wrote',a.out)
if __name__=='__main__': main()
