# immune_panel.py  --  shared config & helpers
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# 50-gene immune signature panel
IMMUNE_GENES = [
    "CD8A","CD8B","CD3D","CD3E","GZMB","PRF1","IFNG","TBX21","EOMES","CXCL9",
    "PDCD1","CD274","HAVCR2","LAG3","TIGIT","CTLA4","VSIR","SIGLEC7",
    "FOXP3","IL2RA","IKZF2","TNFRSF18",
    "CD68","CD163","MRC1","ARG1","IL10","TGFB1","CCL2","CSF1R",
    "NCAM1","KLRD1","NKG7","GNLY",
    "CD19","MS4A1","CD79A","IGHG1",
    "S100A8","S100A9","FCGR3B","MPO",
    "TNF","IL6","IL1B","CXCL8",
    "HLA-A","HLA-DRA","B2M","TAP1",
]

# Common gene-symbol aliases so cohorts with older symbols still match
ALIASES = {
    "CXCL8": ["IL8"], "IL8": ["CXCL8"],
    "HAVCR2": ["TIM3","TIMD3"], "PDCD1": ["PD1","CD279"],
    "CD274": ["PDL1","B7H1"], "VSIR": ["VISTA","C10orf54", "GI24"],
    "FCGR3B": ["CD16B"], "MS4A1": ["CD20"], "NCAM1": ["CD56"],
    "TNFRSF18": ["GITR", "AITR"], "IL2RA": ["CD25"], "SIGLEC7": ["CD328", "SIGLEC-7"],
}

# "Immune-hot vs cold" collapse of the six subtypes (for small-cohort power)
HOT_SUBTYPES  = {"C2","C3"}          # IFN-g dominant, inflammatory
COLD_SUBTYPES = {"C4","C5"}          # lymphocyte depleted, immunologically quiet
# C1 (wound healing) & C6 (TGF-b) treated as intermediate/immunosuppressed

# Marker sets used ONLY for interpreting unsupervised clusters (no leakage)
MARKERS = {
    "IFNg_Cytotoxic": ["CD8A","GZMB","PRF1","IFNG","CXCL9","HLA-DRA","TAP1"],
    "WoundHealing_M2": ["TGFB1","IL10","ARG1","MRC1","CD163"],
    "TGFb_Treg": ["FOXP3","IL2RA","TNFRSF18","TGFB1","IL10"],
    "Inflammatory_Bcell": ["CD19","MS4A1","CD79A","IL1B","CXCL8","IGHG1"],
    "Myeloid": ["S100A8","S100A9","FCGR3B","MPO"],
}


def resolve_genes(available_symbols):
    """Map panel genes onto whatever symbols a cohort actually has."""
    avail = set(available_symbols)
    mapping = {}          # panel_gene -> symbol present in cohort
    for g in IMMUNE_GENES:
        if g in avail:
            mapping[g] = g
        else:
            for alt in ALIASES.get(g, []):
                if alt in avail:
                    mapping[g] = alt
                    break
    return mapping        # missing genes simply absent from dict


def zscore_frame(df):
    """Per-gene z-score WITHIN a cohort (makes classifiers transferable
    across normalisation schemes: TCGA log2-norm vs cohort TPM)."""
    return (df - df.mean(0)) / (df.std(0).replace(0, 1.0))


def immune_hot_score(z_df, mapping):
    """Continuous hot-vs-cold score = mean z of cytotoxic/IFNg markers
    minus mean z of myeloid/suppressive markers. Robust, cohort-agnostic."""
    def m(keys):
        cols = [mapping[g] for g in keys if g in mapping]
        return z_df[cols].mean(1) if cols else pd.Series(0, index=z_df.index)
    hot = m(MARKERS["IFNg_Cytotoxic"])
    cold = (m(MARKERS["Myeloid"]) + m(MARKERS["WoundHealing_M2"])) / 2
    return (hot - cold).rename("immune_hot_score")


# ------------------------------------------------------------
# Baseline comparators (answers "why not just use one gene?")
# ------------------------------------------------------------
BASELINE_GENES = ["CD274", "IFNG", "CD8A"]   # PD-L1, IFN-gamma, CD8 -- classic single-gene comparators


def baseline_gene_scores(z_df, mapping, genes=BASELINE_GENES):
    """Return a dict {gene: z-scored expression Series} for single-gene baselines,
    restricted to genes actually present in this cohort. Use these as comparators
    against the panel's hot_score so a reviewer can see the panel isn't just
    reinventing a well-known single marker."""
    out = {}
    for g in genes:
        if g in mapping and mapping[g] in z_df.columns:
            out[g] = z_df[mapping[g]]
    return out


# ------------------------------------------------------------
# Bootstrap confidence intervals (report these, not bare p-values)
# ------------------------------------------------------------
def bootstrap_auc_ci(y_true, score, n_boot=2000, alpha=0.05, random_state=42):
    """Percentile bootstrap CI for ROC-AUC. Returns (point_auc, lo, hi, n_boot_used)."""
    y_true = np.asarray(y_true); score = np.asarray(score)
    mask = ~pd.isna(y_true) & ~pd.isna(score)
    y_true, score = y_true[mask], score[mask]
    point = roc_auc_score(y_true, score)
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(y_true))
    boots = []
    for _ in range(n_boot):
        bi = resample(idx, random_state=rng.randint(1e9))
        yb, sb = y_true[bi], score[bi]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, sb))
    if len(boots) < 50:
        return point, np.nan, np.nan, len(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, len(boots)


def bootstrap_cox_hr_ci(df, duration_col, event_col, covariate_col,
                        n_boot=1000, alpha=0.05, random_state=42):
    """Percentile bootstrap CI for a Cox HR on one continuous covariate.
    df must already be restricted to complete cases. Returns (hr, lo, hi, n_boot_used)."""
    from lifelines import CoxPHFitter
    d = df[[duration_col, event_col, covariate_col]].dropna()
    cph = CoxPHFitter().fit(d, duration_col, event_col)
    point = float(np.exp(cph.params_[covariate_col]))
    rng = np.random.RandomState(random_state)
    n = len(d)
    boots = []
    for _ in range(n_boot):
        bi = resample(np.arange(n), random_state=rng.randint(1e9))
        db = d.iloc[bi]
        if db[event_col].sum() < 5:            # need enough events to fit
            continue
        try:
            cph_b = CoxPHFitter().fit(db, duration_col, event_col)
            boots.append(np.exp(cph_b.params_[covariate_col]))
        except Exception:
            continue
    if len(boots) < 50:
        return point, np.nan, np.nan, len(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, len(boots)


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.xgb = XGBClassifier(**kwargs)
        self.le = LabelEncoder()
    def fit(self, X, y):
        self.le.fit(y)
        self.xgb.fit(X, self.le.transform(y))
        self.classes_ = self.le.classes_
        return self
    def predict(self, X):
        return self.le.inverse_transform(self.xgb.predict(X))
    def predict_proba(self, X):
        return self.xgb.predict_proba(X)

