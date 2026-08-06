#!/usr/bin/env python
# ============================================================
# 01_train_tcga.py
# Train a compact 50-gene immune classifier on TCGA to predict
# the established (Thorsson 2018) immune subtypes, show it keeps
# PROGNOSTIC value (survival) in TCGA, and FREEZE it for transfer
# to the immunotherapy cohorts (02_validate_ici.py).
#
#   Angle 2 (minimal prognostic panel)  <-- this script
#   Angle 1 (predicts ICI response)     <-- 02_validate_ici.py
# ============================================================
# !pip install pandas numpy scikit-learn scipy statsmodels lifelines matplotlib seaborn xenaPython joblib -q

import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from immune_panel import (IMMUNE_GENES, resolve_genes, zscore_frame,
                          immune_hot_score, HOT_SUBTYPES, COLD_SUBTYPES, XGBWrapper)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
OUT = "outputs"; MODEL = "model"
os.makedirs(OUT, exist_ok=True); os.makedirs(MODEL, exist_ok=True)

CANCER_TYPES = None
# set CANCER_TYPES = None to use all 33 TCGA types

# ------------------------------------------------------------
# 1. LOAD TCGA  (expression panel + Thorsson subtype + survival)
# ------------------------------------------------------------
print("[1] Loading TCGA from UCSC Xena ...")
import xenaPython as xena
HOST = xena.PUBLIC_HUBS["pancanAtlasHub"]
EXPR_DS   = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
IMMUNE_DS = "Subtype_Immune_Model_Based.txt"
SURV_DS   = "Survival_SupplementalTable_S1_20171025_xena_sp"

def _find_vec(x, n, depth=0):
    if isinstance(x, list):
        if len(x) == n and (len(x) == 0 or not isinstance(x[0], list)): return x
        if depth < 4:
            for el in x:
                r = _find_vec(el, n, depth + 1)
                if r is not None: return r
    return None

def _parse_codes(fc):
    def split(s): return s.split("\t") if isinstance(s, str) else list(s)
    if isinstance(fc, list) and fc:
        f0 = fc[0]
        if isinstance(f0, dict):
            c = f0.get("code") or f0.get("codes"); return split(c) if c is not None else None
        if isinstance(f0, str):  return split(f0)
        if isinstance(f0, list): return [str(v) for v in f0]
    return None

def xget1(ds, field, samples):
    """one field -> Series; decodes coded fields via field_codes, passes numeric through."""
    raw = xena.dataset_probe_values(HOST, ds, samples, [field])
    vec = _find_vec(raw, len(samples))
    assert vec is not None, f"could not locate values for '{field}'. RAW head: {str(raw)[:250]}"
    try: codes = _parse_codes(xena.field_codes(HOST, ds, [field]))
    except Exception: codes = None
    if codes:
        out = []
        for v in vec:
            if v is None or (isinstance(v, float) and np.isnan(v)): out.append(np.nan); continue
            try: i = int(round(float(v)))
            except Exception: out.append(np.nan); continue
            out.append(codes[i] if 0 <= i < len(codes) else np.nan)
        return pd.Series(out, index=samples)
    return pd.Series([np.nan if v is None else v for v in vec], index=samples)

def parse_expr(raw, samples, genes):
    def is_mat(x): return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)
    mat = None
    if is_mat(raw) and len(raw) == len(genes): mat = raw
    elif isinstance(raw, list):
        for el in raw:
            if is_mat(el) and len(el) == len(genes): mat = el; break
    if mat is None and is_mat(raw): mat = raw
    assert mat is not None, "unexpected expression return shape"
    arr = np.array([[np.nan if v is None else float(v) for v in row] for row in mat], dtype=float)
    return pd.DataFrame(arr.T, index=samples, columns=genes[:arr.shape[0]])

samples = xena.dataset_samples(HOST, IMMUNE_DS, None)
assert samples, "no samples from immune dataset"
imm_field = xena.dataset_field(HOST, IMMUNE_DS)
field = [f for f in imm_field if f.lower() not in ("sampleid","_patient","sample")][0]
def _norm(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().replace(".0", "")
    import re
    m = re.search(r"(C[1-6])", s)
    if m: return m.group(1)
    return s if s.startswith("C") else (f"C{s}" if s in list("123456") else np.nan)
subtype = xget1(IMMUNE_DS, field, samples).map(_norm).dropna()
assert len(subtype) > 100, "no C1..C6 labels parsed"
samples = subtype.index.tolist()
print(f"    labelled samples: {len(samples)}")

cancer  = xget1(SURV_DS, "cancer type abbreviation", samples)
os_evt  = pd.to_numeric(xget1(SURV_DS, "OS", samples), errors="coerce")
os_time = pd.to_numeric(xget1(SURV_DS, "OS.time", samples), errors="coerce")
meta = pd.DataFrame({"cancer_type": cancer, "OS": os_evt, "OS_time": os_time})

raw = xena.dataset_probe_values(HOST, EXPR_DS, samples, IMMUNE_GENES)
expr = parse_expr(raw, samples, IMMUNE_GENES).dropna(axis=1, how="all")
assert expr.shape[1] > 10, "too few genes returned"

df = expr.join(meta).join(subtype.rename("subtype")).dropna(subset=["subtype"])
if CANCER_TYPES:
    df = df[df["cancer_type"].isin(CANCER_TYPES)]

mapping = resolve_genes(expr.columns)
panel = list(mapping.keys())
print(f"    panel genes present: {len(panel)}/{len(IMMUNE_GENES)}")

# training set = samples WITH a Thorsson label and full expression
train = df.dropna(subset=[mapping[g] for g in panel] + ["subtype"]).copy()
train = train[train["subtype"].astype(str).str.match(r"C[1-6]")]
X_raw = train[[mapping[g] for g in panel]]
X = zscore_frame(X_raw)                       # per-gene z within TCGA
y = train["subtype"].astype(str).values
print(f"    training samples: {len(train)}   subtypes: {sorted(set(y))}")

# ------------------------------------------------------------
# 2. TRAIN + CROSS-VALIDATE the compact classifier  (ANGLE 2)
# ------------------------------------------------------------
print("[2] Training compact immune-subtype classifier ...")
xgb_clf = XGBWrapper(n_estimators=500, max_depth=6, learning_rate=0.05, 
                     subsample=0.8, eval_metric='mlogloss', 
                     random_state=RANDOM_STATE, n_jobs=-1)

clf = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
    ('xgb', xgb_clf)
])

cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
y_cv = cross_val_predict(clf, X, y, cv=cv)
acc = accuracy_score(y, y_cv); bacc = balanced_accuracy_score(y, y_cv)

# bootstrap CI on CV accuracy (report this, not just the point estimate)
rng = np.random.RandomState(RANDOM_STATE)
n = len(y); boot_accs = []
y_arr, ycv_arr = np.asarray(y), np.asarray(y_cv)
for _ in range(2000):
    bi = rng.randint(0, n, n)
    boot_accs.append(accuracy_score(y_arr[bi], ycv_arr[bi]))
acc_lo, acc_hi = np.percentile(boot_accs, [2.5, 97.5])
print(f"    5-fold accuracy = {acc*100:.1f}% [95% CI {acc_lo*100:.1f}-{acc_hi*100:.1f}%]"
      f"   balanced = {bacc*100:.1f}%")
clf.fit(X, y)                                 # final fit on all training data

# confusion matrix figure
cm = confusion_matrix(y, y_cv, labels=sorted(set(y)))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
plt.xlabel("Predicted"); plt.ylabel("True (Thorsson)")
plt.title(f"TCGA subtype prediction (acc {acc*100:.0f}%)")
plt.tight_layout(); plt.savefig(f"{OUT}/Fig_confusion_tcga.png", dpi=180); plt.close()

# ------------------------------------------------------------
# 2b. Unsupervised recovery check (supporting; no leakage)
# ------------------------------------------------------------
pca = PCA(0.95, random_state=RANDOM_STATE).fit_transform(X)
km = KMeans(6, n_init=20, random_state=RANDOM_STATE).fit_predict(pca)
ari_unsup = adjusted_rand_score(y, km)
print(f"    unsupervised k=6 vs Thorsson ARI = {ari_unsup:.3f} (supporting)")

# ------------------------------------------------------------
# 3. TCGA SURVIVAL by subtype   (ANGLE 2: prognostic value)
# ------------------------------------------------------------
print("[3] TCGA survival stratification by immune subtype ...")
surv = train.dropna(subset=["OS","OS_time"]).copy()
surv["OS"] = surv["OS"].astype(float); surv["OS_time"] = surv["OS_time"].astype(float)
surv = surv[surv["OS_time"] > 0]
surv["hot"] = surv["subtype"].map(lambda s: "Hot(C2/C3)" if s in HOT_SUBTYPES
                                  else ("Cold(C4/C5)" if s in COLD_SUBTYPES else "Interm."))

# KM by 6 subtypes
plt.figure(figsize=(7,5)); kmf = KaplanMeierFitter()
for s in sorted(surv["subtype"].unique()):
    m = surv["subtype"] == s
    kmf.fit(surv.loc[m,"OS_time"], surv.loc[m,"OS"], label=s); kmf.plot_survival_function(ci_show=False)
lr = multivariate_logrank_test(surv["OS_time"], surv["subtype"], surv["OS"])
plt.title(f"TCGA OS by immune subtype (log-rank p={lr.p_value:.1e})")
plt.xlabel("Days"); plt.ylabel("Overall survival"); plt.tight_layout()
plt.savefig(f"{OUT}/Fig_tcga_survival_subtype.png", dpi=180); plt.close()
print(f"    log-rank across subtypes: p = {lr.p_value:.2e}")

# Cox: hot vs cold as covariate (+ cancer type strata)
cox_df = surv[["OS_time","OS"]].copy()
cox_df["hot_vs_cold"] = (surv["subtype"].isin(HOT_SUBTYPES)).astype(int)
try:
    cph = CoxPHFitter().fit(cox_df, "OS_time", "OS")
    hr = np.exp(cph.params_["hot_vs_cold"]); p_cox = cph.summary.loc["hot_vs_cold","p"]
    print(f"    Cox hot-vs-cold: HR={hr:.2f}, p={p_cox:.2e}")
except Exception as e:
    hr = p_cox = np.nan; print("    Cox failed:", e)

# ------------------------------------------------------------
# 4. FREEZE the model for transfer to ICI cohorts
# ------------------------------------------------------------
joblib.dump({"clf": clf, "panel": panel, "mapping_train": mapping,
             "train_mean": X_raw.mean(0), "train_std": X_raw.std(0)},
            f"{MODEL}/immune_clf.joblib")
summary = dict(n_train=int(len(train)), n_genes=len(panel),
               cv_accuracy=float(acc), cv_accuracy_lo=float(acc_lo), cv_accuracy_hi=float(acc_hi),
               balanced_accuracy=float(bacc),
               unsup_ari=float(ari_unsup), logrank_p=float(lr.p_value),
               cox_hr=float(hr), cox_p=float(p_cox),
               cancer_types=CANCER_TYPES)
json.dump(summary, open(f"{OUT}/tcga_summary.json","w"), indent=2)
train.assign(cv_pred=y_cv).to_csv(f"{OUT}/tcga_predictions.csv")

print("\n=== NUMBERS FOR THE PAPER (TCGA) ===")
print(json.dumps(summary, indent=2))
print("Model frozen -> model/immune_clf.joblib  | now run 02_validate_ici.py")
