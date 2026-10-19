from pathlib import Path
import pandas as pd, numpy as np, json
p=Path(__file__).resolve().parent / 'secom_telemetry_modified.csv'
df=pd.read_csv(p); df['timestamp']=pd.to_datetime(df['timestamp']); df=df.sort_values('timestamp',kind='stable').reset_index(drop=True)
cols=[c for c in df.columns if c.startswith('sensor_')]
n=len(df); splits={'train':df.iloc[:int(.6*n)],'validation':df.iloc[int(.6*n):int(.8*n)],'test':df.iloc[int(.8*n):]}
# Derive normalization parameters from training only, then use those fixed parameters for drift diagnostics.
tr=splits['train'][cols].copy(); leg=tr.sensor_0.eq(0); tr.loc[leg,'sensor_0']=np.nan; tr.loc[leg,'sensor_1']=np.nan
med=tr.median(numeric_only=True); q1=tr.quantile(.25); q3=tr.quantile(.75); iqr=(q3-q1).replace(0,np.nan)
out={'rows':n,'splits':{},'largest_median_shift_features':[],'largest_missingness_change_features':[]}
shifts=[]; miss=[]
for name,part in splits.items():
 x=part[cols].copy(); legacy=x.sensor_0.eq(0); x.loc[legacy,'sensor_0']=np.nan; x.loc[legacy,'sensor_1']=np.nan
 out['splits'][name]={'rows':len(part),'defects':int((part.label==1).sum()),'prevalence':float((part.label==1).mean()),'legacy_rows':int(legacy.sum()),'raw_missing_fraction':float(part[cols].isna().mean().mean())}
 for c in cols:
  miss.append({'feature':c,'split':name,'missing_rate':float(x[c].isna().mean())})
  if name!='train':
   valid=x[c].dropna(); scale=iqr[c]
   if len(valid) and pd.notna(scale): shifts.append({'feature':c,'split':name,'abs_median_shift_in_train_IQR':float(abs(valid.median()-med[c])/scale)})
out['largest_median_shift_features']=sorted(shifts,key=lambda z:z['abs_median_shift_in_train_IQR'],reverse=True)[:15]
# missingness difference compared with train
md=pd.DataFrame(miss).pivot(index='feature',columns='split',values='missing_rate').fillna(0)
for split in ['validation','test']:
 for c,v in (md[split]-md['train']).abs().items(): miss.append({'feature':c,'split':split,'abs_missing_rate_change_vs_train':float(v)})
mm=[x for x in miss if 'abs_missing_rate_change_vs_train' in x]
out['largest_missingness_change_features']=sorted(mm,key=lambda z:z['abs_missing_rate_change_vs_train'],reverse=True)[:15]
out_path=Path(__file__).resolve().parent / 'artifacts' / 'temporal_drift_report.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
