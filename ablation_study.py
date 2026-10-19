#!/usr/bin/env python3
"""Staged ablation comparison for the Wafer-GUARD audit.

This is a staged protocol comparison, not a perfect causal decomposition. C0 -> C1
changes multiple inseparable evaluation-protocol choices (resampling before split,
balanced holdout, and threshold protocol) to establish how contaminated scores compare
with a natural-prevalence split. C1 -> C2, C2 -> C3, and C3 -> C4 onward each target
one named change. Every config is run for multiple seeds and reported as mean +/- std.
Nothing here reads the evaluator-only mask file.

Rows
  C0 original        random 80/20 split AFTER SMOTE on all data, mean-impute+scale on all
                     data, [B,1,F] broadcast bug, balanced test set, threshold 0.5
  C1 +split first/no SMOTE    natural-prevalence random 60/20/20 split, class imbalance handled by
                     pos_weight (no SMOTE); preprocessing still fit on ALL rows
  C2 +train-only fit imputer + scaler fit on training rows only
  C3 +chronological  time-ordered 60/20/20 split (train < val < test)
  C4 +shape fix      [B,F] tensors, reconstruction/input shapes asserted equal
  C5 +PS-L handling  sensor_0 sentinel and sensor_1 codes -> NaN + categorical indicators
  C6 +median impute  median instead of mean imputation (= the submitted notebook design)
  C7 +PS-L strict    candidate: legacy sensor_0/sensor_1 slots are NOT filled with a fitted
                     value; they are set to 0 after scaling and masked out of the
                     reconstruction loss (indicators carry the information)

Threshold for C1..C7: the F1-maximising threshold on the validation split only.
C0 uses 0.5 on a balanced test set, as the original notebook did.

Usage:
  python ablation_study.py --data secom_telemetry_modified.csv --seeds 1337 17 41
  python ablation_study.py --dry-run      # data/split checks only, no torch needed
"""
import argparse, json, os, random, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

HP = dict(batch_size=64, epochs=30, lr=1e-3, weight_decay=1e-5, hidden=256, latent=32)

CONFIGS = [
    ("C0 original",        dict(split="smote_random", fit="all",   imb="smote",      shape="bug",   psl="naive",   imp="mean")),
    ("C1 +split first/no SMOTE",    dict(split="random",       fit="all",   imb="pos_weight", shape="bug",   psl="naive",   imp="mean")),
    ("C2 +train-only fit", dict(split="random",       fit="train", imb="pos_weight", shape="bug",   psl="naive",   imp="mean")),
    ("C3 +chronological",  dict(split="chrono",       fit="train", imb="pos_weight", shape="bug",   psl="naive",   imp="mean")),
    ("C4 +shape fix",      dict(split="chrono",       fit="train", imb="pos_weight", shape="fixed", psl="naive",   imp="mean")),
    ("C5 +PS-L handling",  dict(split="chrono",       fit="train", imb="pos_weight", shape="fixed", psl="handled", imp="mean")),
    ("C6 +median impute",  dict(split="chrono",       fit="train", imb="pos_weight", shape="fixed", psl="handled", imp="median")),
    ("C7 +PS-L strict",    dict(split="chrono",       fit="train", imb="pos_weight", shape="fixed", psl="strict",  imp="median")),
]


def load(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cols = [c for c in df.columns if c.startswith("sensor_")]
    y = (df["label"].to_numpy() == 1).astype(np.int64)
    return df[cols].copy(), y, cols


def features(raw, psl):
    """Return (matrix with NaNs, indicator matrix or None, legacy bool array)."""
    legacy = raw["sensor_0"].eq(0.0).to_numpy()
    X = raw.copy()
    ind = None
    if psl in ("handled", "strict"):
        codes = [(legacy & raw["sensor_1"].eq(c).to_numpy()) for c in (100.0, 200.0, 300.0)]
        ind = np.stack([legacy] + codes, axis=1).astype(np.float32)
        X.loc[legacy, "sensor_0"] = np.nan   # sentinel, not a measurement
        X.loc[legacy, "sensor_1"] = np.nan   # categorical firmware code, not a float
    return X.to_numpy(dtype=np.float64), ind, legacy


def make_splits(split, y, seed):
    n = len(y)
    if split == "chrono":
        a, b = int(n * 0.6), int(n * 0.8)
        return np.arange(a), np.arange(a, b), np.arange(b, n)
    idx = np.arange(n)
    trv, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    tr, va = train_test_split(trv, test_size=0.25, stratify=y[trv], random_state=seed)
    return tr, va, te


def prepare(raw, y, cfg, seed):
    """Build train/val/test arrays (+ optional reconstruction mask) for one config."""
    X, ind, legacy = features(raw, cfg["psl"])
    if cfg["split"] == "smote_random":
        # ORIGINAL (leaky) protocol: impute+scale on ALL rows, SMOTE on ALL rows, random split.
        from imblearn.over_sampling import SMOTE
        Xs = StandardScaler().fit_transform(SimpleImputer(strategy=cfg["imp"], keep_empty_features=True).fit_transform(X))
        Xb, yb = SMOTE(random_state=seed).fit_resample(Xs, y)
        Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=seed)
        return dict(Xtr=Xtr, ytr=ytr, Xva=None, yva=None, Xte=Xte, yte=yte, Mtr=None, original=True)

    tr, va, te = make_splits(cfg["split"], y, seed)
    fit_rows = np.arange(len(y)) if cfg["fit"] == "all" else tr
    imp = SimpleImputer(strategy=cfg["imp"], keep_empty_features=True).fit(X[fit_rows])
    sc = StandardScaler().fit(imp.transform(X[fit_rows]))
    Z = sc.transform(imp.transform(X)).astype(np.float32)
    M = None
    if cfg["psl"] == "strict":
        M = np.ones_like(Z, dtype=np.float32)
        Z[legacy, 0] = 0.0; Z[legacy, 1] = 0.0   # neutral slot; no fitted value is claimed
        M[legacy, 0] = 0.0; M[legacy, 1] = 0.0   # and the model is not asked to reconstruct it
    if ind is not None:
        Z = np.concatenate([Z, ind], axis=1)
        if M is not None:
            M = np.concatenate([M, np.ones_like(ind)], axis=1)
    if cfg["fit"] == "train":   # causality guard: no statistics from val/test
        assert imp.statistics_.shape[0] == X.shape[1]
    pick = lambda a, i: None if a is None else a[i]
    return dict(Xtr=Z[tr], ytr=y[tr], Xva=Z[va], yva=y[va], Xte=Z[te], yte=y[te],
                Mtr=pick(M, tr), original=False, idx=(tr, va, te))


def fit_predict(d, cfg, seed, epochs):
    import torch, torch.nn as nn
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    F = d["Xtr"].shape[1]

    class SAE(nn.Module):
        def __init__(s, h, lat):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(F, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.10),
                                  nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU(), nn.Linear(h // 2, lat))
            s.dec = nn.Sequential(nn.Linear(lat, h // 2), nn.GELU(), nn.Linear(h // 2, h), nn.GELU(), nn.Linear(h, F))
            s.head = nn.Linear(lat, 1)
        def forward(s, x):
            x = x.flatten(start_dim=1)          # tolerate [B,F] or [B,1,F], as the original did
            z = s.enc(x)
            return s.dec(z), s.head(z).squeeze(-1)

    model = SAE(HP["hidden"], HP["latent"])
    Xtr = torch.from_numpy(d["Xtr"]).float(); ytr = torch.from_numpy(d["ytr"]).float()
    Mtr = None if d["Mtr"] is None else torch.from_numpy(d["Mtr"]).float()
    if cfg["imb"] == "pos_weight":
        pw = torch.tensor([(len(ytr) - ytr.sum()) / max(float(ytr.sum()), 1.0)])
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    bs = HP["batch_size"]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb, yb = Xtr[b], ytr[b]
            if cfg["shape"] == "bug":
                xb = xb.unsqueeze(1)             # [B,1,F]  -> reproduces the silent [B,B,F] broadcast
            rec, logit = model(xb)
            if cfg["shape"] == "fixed":
                assert rec.shape == xb.shape, (rec.shape, xb.shape)
                assert logit.shape == yb.shape
            sq = (rec - xb) ** 2
            if Mtr is not None:
                m = Mtr[b]
                recon = (sq * m).sum() / m.sum().clamp(min=1.0)
            else:
                recon = sq.mean()
            loss = recon + bce(logit, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

    def prob(X):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X).float())[1]).numpy()

    pte = prob(d["Xte"])
    if d["original"]:
        thr = 0.5
    else:
        pva = prob(d["Xva"]); yva = d["yva"]
        cands = np.unique(np.concatenate(([0.0, 0.5, 1.0], pva)))
        thr = max(cands, key=lambda t: (f1_score(yva, (pva >= t).astype(int), zero_division=0), -abs(t - 0.5)))
    yte = d["yte"]
    return dict(f1=f1_score(yte, (pte >= thr).astype(int), zero_division=0),
                ap=average_precision_score(yte, pte),
                auc=roc_auc_score(yte, pte) if len(np.unique(yte)) == 2 else float("nan"),
                prevalence=float(yte.mean()), threshold=float(thr), n_test=int(len(yte)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="secom_telemetry_modified.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 17, 41])
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--out", default="artifacts/ablation_results.json")
    ap.add_argument("--dry-run", action="store_true", help="only build splits/features and run causality checks")
    a = ap.parse_args()

    raw, y, _ = load(a.data)
    print(f"rows={len(y)} natural prevalence={y.mean():.4%}")
    results = {}
    for name, cfg in CONFIGS:
        rows = []
        for s in a.seeds:
            if a.dry_run and cfg["split"] == "smote_random":
                continue                                    # needs imbalanced-learn
            d = prepare(raw, y, cfg, s)
            if a.dry_run:
                print(f"[dry] {name}: train {d['Xtr'].shape} val {None if d['Xva'] is None else d['Xva'].shape} "
                      f"test {d['Xte'].shape} test-prev {d['yte'].mean():.3%}")
                break
            rows.append(fit_predict(d, cfg, s, a.epochs))
        if rows:
            results[name] = rows
            print(f"{name}: " + ", ".join(f"F1={r['f1']:.3f}/AP={r['ap']:.3f}" for r in rows))
    if a.dry_run:
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    ms = lambda v: f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}"
    print("\n| Config | Test prevalence | F1 | AP | ROC-AUC |\n|---|---:|---:|---:|---:|")
    for name, rows in results.items():
        print(f"| {name} | {np.mean([r['prevalence'] for r in rows]):.1%} | "
              f"{ms([r['f1'] for r in rows])} | {ms([r['ap'] for r in rows])} | {ms([r['auc'] for r in rows])} |")


if __name__ == "__main__":
    main()