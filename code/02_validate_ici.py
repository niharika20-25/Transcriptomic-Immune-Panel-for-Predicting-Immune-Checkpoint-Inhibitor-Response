#!/usr/bin/env python
# ============================================================
# 02_validate_ici.py
# Apply the frozen TCGA-trained immune classifier to independent
# immune-checkpoint-inhibitor (ICI) cohorts and test whether the
# predicted subtype / immune-hot score predicts:
#     (a) RESPONSE  (responder vs non-responder)      <-- ANGLE 1
#     (b) on-treatment SURVIVAL (OS)                   <-- ANGLE 1+2
#
# PRIMARY cohort: IMvigor210 (atezolizumab, urothelial, Mariathasan
#   2018 Nature). Export it once with export_imvigor210.R -> 2 CSVs.

# (see COHORTS below): a melanoma anti-PD-1 set (Hugo/Riaz/Liu) etc.
# ============================================================
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from immune_panel import (resolve_genes, zscore_frame, immune_hot_score,
                          HOT_SUBTYPES, COLD_SUBTYPES, BASELINE_GENES,
                          baseline_gene_scores, bootstrap_auc_ci,
                          bootstrap_cox_hr_ci, XGBWrapper)

# ------------------------------------------------------------
# Pre-registered primary endpoint (fixes: "many tests on one cohort").
# Everything else reported here is SECONDARY / exploratory and gets a
# Benjamini-Hochberg correction across cohorts at the end.
# ------------------------------------------------------------
PRIMARY_ENDPOINT = "hotscore_auc"   # AUC of hot_score for response, in the PRIMARY cohort (IMvigor210)
PRIMARY_COHORT = "IMvigor210"

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
print("[2] Loading frozen model ...")
bundle = joblib.load("model/immune_clf.joblib")
CLF, PANEL = bundle["clf"], bundle["panel"]

# ------------------------------------------------------------
# Register cohorts here. Each needs:
#   expr_csv : genes x samples  (rows=gene symbol, cols=sample id)
#   clin_csv : samples x columns, must contain a response col and/or OS
#   resp_col : column with responder labels
#   resp_map : dict mapping raw labels -> 1 (responder) / 0 (non-resp)
#   os_col, event_col : survival time + event (1=death)
# ------------------------------------------------------------
COHORTS = {
    "IMvigor210": dict(
        expr_csv="imvigor210_expr.csv", clin_csv="imvigor210_clin.csv",
        resp_col="binaryResponse",
        resp_map={"CR/PR":1, "SD/PD":0},            # IMvigor210 provides this field
        # fallback if only 'Best Confirmed Overall Response' present:
        resp_col_alt="Best Confirmed Overall Response",
        resp_map_alt={"CR":1,"PR":1,"SD":0,"PD":0,"NE":np.nan},
        os_col="os", event_col="censOS",            # censOS: 1=event in this package
    ),
    # ------------------------------------------------------------
    # SECOND / THIRD cohorts. A single ICI cohort is a weak claim for a
    # conference reviewer -- add at least one of these before presenting.
    # Both are free on GEO. Use export_geo_cohort.py to pull + format them
    # into the same 2-CSV shape as IMvigor210 (expr: genes x samples,
    # clin: samples x columns). GEO metadata field NAMES vary by series and
    # sometimes by re-upload, so export_geo_cohort.py prints clin.columns
    # after download -- copy the exact response/OS column names you see
    # into resp_col / os_col / event_col below before uncommenting.
    #
    "Hugo2016": dict(               # GSE78220, melanoma, anti-PD-1
        expr_csv="hugo2016_expr.csv", clin_csv="hugo2016_clin.csv",
        resp_col="anti-pd-1 response", 
        resp_map={"Complete Response":1, "Partial Response":1,
                                        "Progressive Disease":0},
        os_col=None, event_col=None,   # this series has no OS field -- response only
    ),
    # "Riaz2017": dict(                # GSE91061, melanoma, nivolumab, pre-treatment samples
    #     expr_csv="riaz2017_expr.csv", clin_csv="riaz2017_clin.csv",
    #     resp_col="response", resp_map={"PR":1,"CR":1,"SD":0,"PD":0},
    #     os_col="os_days", event_col="dead",
    # ),
}


def load_cohort(cfg):
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)          # genes x samples
    clin = pd.read_csv(cfg["clin_csv"], index_col=0)
    mapping = resolve_genes(expr.index)
    present = [g for g in PANEL if g in mapping]
    Xc = expr.loc[[mapping[g] for g in present]].T           # samples x genes
    Xc.columns = present
    Xc = Xc.apply(pd.to_numeric, errors="coerce")
    # log if the data look like linear TPM/counts (heuristic)
    if np.nanmax(Xc.values) > 50:
        Xc = np.log2(Xc + 1)
    Xc = Xc.dropna()
    clin = clin.reindex(Xc.index)
    return Xc, clin, mapping, present


RESPONSE_KEYWORDS = ["response", "resp", "rec ist", "recist", "benefit"]


def get_response(clin, cfg, cohort_name=""):
    col = cfg["resp_col"] if cfg["resp_col"] in clin else cfg.get("resp_col_alt")
    rmap = cfg["resp_map"] if cfg["resp_col"] in clin else cfg.get("resp_map_alt", {})
    if col is None or col not in clin:
        # don't silently return all-NaN -- surface the mismatch so a
        # renamed/misspelled GEO column doesn't quietly kill the cohort.
        guesses = [c for c in clin.columns
                  if any(k in str(c).lower() for k in RESPONSE_KEYWORDS)]
        print(f"  [{cohort_name}] WARNING: response column "
              f"'{cfg.get('resp_col')}' not found. "
              f"Columns that look response-like: {guesses or 'none found'}. "
              f"Update resp_col in COHORTS and re-run.")
        return pd.Series(np.nan, index=clin.index)
    return clin[col].map(rmap).astype("float")


def predict_subtypes(Xc, present):
    """z-score within cohort, align to trained panel (fill absent genes with 0),
    then predict subtype + probability of a hot phenotype."""
    z = zscore_frame(Xc)
    full = pd.DataFrame(0.0, index=z.index, columns=PANEL)
    for g in present:
        full[g] = z[g].values
    pred = pd.Series(CLF.predict(full.values), index=z.index, name="pred_subtype")
    proba = pd.DataFrame(CLF.predict_proba(full.values),
                         index=z.index, columns=CLF.classes_)
    p_hot = proba[[c for c in CLF.classes_ if c in HOT_SUBTYPES]].sum(1)
    return pred, p_hot.rename("p_hot"), z


all_rows = []
for name, cfg in COHORTS.items():
    print(f"\n=== {name} ===")
    Xc, clin, mapping, present = load_cohort(cfg)
    print(f"  samples={len(Xc)}  panel genes present={len(present)}/{len(PANEL)}")
    pred, p_hot, z = predict_subtypes(Xc, present)
    hot_score = immune_hot_score(z, {g: g for g in present})
    resp = get_response(clin, cfg, cohort_name=name)

    d = pd.DataFrame({"pred_subtype": pred, "p_hot": p_hot,
                      "hot_score": hot_score, "response": resp})
    d["hot_group"] = d["pred_subtype"].map(
        lambda s: "Hot" if s in HOT_SUBTYPES else ("Cold" if s in COLD_SUBTYPES else "Interm"))
    d.to_csv(f"{OUT}/{name}_assignments.csv")

    # ---- (a) RESPONSE association ----
    dd = d.dropna(subset=["response"])
    p_chi = auc = p_u = rr_hot = rr_cold = np.nan
    auc_lo = auc_hi = np.nan
    baseline_aucs = {}
    if len(dd) > 10:
        # chi-square: hot/cold/interm x responder
        tab = pd.crosstab(dd["hot_group"], dd["response"])
        chi2, p_chi, *_ = chi2_contingency(tab)
        # AUC of continuous hot_score for response, with bootstrap CI
        try:
            auc, auc_lo, auc_hi, nb = bootstrap_auc_ci(dd["response"], dd["hot_score"])
        except Exception:
            auc = np.nan
        # Mann-Whitney hot_score responders vs non
        u, p_u = mannwhitneyu(dd.loc[dd.response==1,"hot_score"],
                              dd.loc[dd.response==0,"hot_score"])
        rr_hot = dd.loc[dd.hot_group=="Hot","response"].mean()
        rr_cold = dd.loc[dd.hot_group=="Cold","response"].mean()
        print(f"  RESPONSE  chi2 p={p_chi:.3f} | hot_score AUC={auc:.3f} "
              f"[95% CI {auc_lo:.3f}-{auc_hi:.3f}] (MWU p={p_u:.3f}) | "
              f"RR hot={rr_hot:.2f} cold={rr_cold:.2f}")

        # ---- baseline comparators: is the 50-gene panel doing anything a
        # single well-known marker (PD-L1 / IFNG / CD8A) wouldn't already do?
        # This is the question every reviewer asks first -- answer it up front.
        base_scores = baseline_gene_scores(z, mapping)
        for gname, gscore in base_scores.items():
            gscore = gscore.reindex(dd.index)
            try:
                b_auc, b_lo, b_hi, _ = bootstrap_auc_ci(dd["response"], gscore.loc[dd.index])
            except Exception:
                b_auc = b_lo = b_hi = np.nan
            baseline_aucs[gname] = b_auc
            print(f"    baseline [{gname} alone] AUC={b_auc:.3f} [{b_lo:.3f}-{b_hi:.3f}]")

        plt.figure(figsize=(5,4))
        sns.boxplot(data=dd, x="response", y="hot_score",
                    palette=["#457B9D","#E63946"])
        sns.stripplot(data=dd, x="response", y="hot_score", color="black", size=3, alpha=.5)
        plt.xticks([0,1], ["Non-responder","Responder"])
        plt.title(f"{name}: immune-hot score vs response\nAUC={auc:.2f}, p={p_u:.3f}")
        plt.tight_layout(); plt.savefig(f"{OUT}/{name}_response.png", dpi=180); plt.close()

        # ---- ROC Curve Plot ----
        plt.figure(figsize=(6,5))
        fpr, tpr, _ = roc_curve(dd["response"], dd["hot_score"])
        plt.plot(fpr, tpr, label=f"50-Gene Panel (AUC = {auc:.2f})", linewidth=2, color="#E63946")
        
        colors = {"CD274": "#457B9D", "IFNG": "#1D3557", "CD8A": "#F4A261"}
        for gname, gscore in base_scores.items():
            gscore = gscore.reindex(dd.index)
            if not gscore.isna().all():
                fpr_g, tpr_g, _ = roc_curve(dd["response"], gscore.loc[dd.index])
                b_auc = baseline_aucs.get(gname, np.nan)
                plt.plot(fpr_g, tpr_g, label=f"{gname} (AUC = {b_auc:.2f})", linestyle="--", color=colors.get(gname, "gray"))
        
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name}: ROC Curve for Response Prediction")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{OUT}/{name}_roc.png", dpi=180)
        plt.close()
    else:
        print("  (insufficient response labels)")

    # ---- (b) SURVIVAL association ----
    p_lr = hr = p_cox = np.nan
    hr_lo = hr_hi = np.nan
    if cfg.get("os_col") in clin and cfg.get("event_col") in clin:
        s = d.join(clin[[cfg["os_col"], cfg["event_col"]]]).rename(
            columns={cfg["os_col"]:"T", cfg["event_col"]:"E"})
        s = s.dropna(subset=["T","E"]); s["T"]=s["T"].astype(float); s["E"]=s["E"].astype(float)
        s = s[s["T"] > 0]
        if len(s) > 10 and s["hot_group"].isin(["Hot","Cold"]).sum() > 6:
            hc = s[s["hot_group"].isin(["Hot","Cold"])]
            lr = logrank_test(hc.loc[hc.hot_group=="Hot","T"], hc.loc[hc.hot_group=="Cold","T"],
                              hc.loc[hc.hot_group=="Hot","E"], hc.loc[hc.hot_group=="Cold","E"])
            p_lr = lr.p_value
            cox = s[["T","E"]].copy(); cox["hot_score"]=s["hot_score"]
            try:
                cph = CoxPHFitter().fit(cox.dropna(), "T", "E")
                hr = np.exp(cph.params_["hot_score"]); p_cox = cph.summary.loc["hot_score","p"]
                hr, hr_lo, hr_hi, _ = bootstrap_cox_hr_ci(cox, "T", "E", "hot_score")
            except Exception: pass
            plt.figure(figsize=(6,5)); kmf=KaplanMeierFitter()
            for grp,c in [("Hot","#E63946"),("Cold","#457B9D")]:
                m = s["hot_group"]==grp
                if m.sum()>3:
                    kmf.fit(s.loc[m,"T"], s.loc[m,"E"], label=f"{grp} (n={m.sum()})")
                    kmf.plot_survival_function(ci_show=False, color=c)
            plt.title(f"{name}: OS by immune phenotype (log-rank p={p_lr:.3f})")
            plt.xlabel("Time"); plt.ylabel("OS"); plt.tight_layout()
            plt.savefig(f"{OUT}/{name}_survival.png", dpi=180); plt.close()
            print(f"  SURVIVAL  log-rank p={p_lr:.3f} | Cox(hot_score) "
                  f"HR={hr:.2f} [{hr_lo:.2f}-{hr_hi:.2f}] p={p_cox:.3f}")

    row = dict(cohort=name, n=len(Xc), n_resp=int(len(dd)),
              resp_chi2_p=p_chi, hotscore_auc=auc,
              hotscore_auc_lo=auc_lo, hotscore_auc_hi=auc_hi, resp_mwu_p=p_u,
              RR_hot=rr_hot, RR_cold=rr_cold,
              surv_logrank_p=p_lr, cox_hr=hr, cox_hr_lo=hr_lo, cox_hr_hi=hr_hi,
              cox_p=p_cox, is_primary_endpoint=(name == PRIMARY_COHORT))
    for gname, gauc in baseline_aucs.items():
        row[f"baseline_{gname}_auc"] = gauc
    all_rows.append(row)

summary_df = pd.DataFrame(all_rows)

# ------------------------------------------------------------
# Multiple-testing correction across SECONDARY p-values (everything
# except the pre-registered primary endpoint). Report both raw and
# BH-adjusted p so a reviewer sees you didn't cherry-pick.
# ------------------------------------------------------------
secondary_pcols = ["resp_chi2_p", "resp_mwu_p", "surv_logrank_p", "cox_p"]
raw_p = []
for col in secondary_pcols:
    if col in summary_df:
        raw_p.extend([(col, i, v) for i, v in summary_df[col].items() if pd.notna(v)])
if raw_p:
    _, adj, _, _ = multipletests([v for *_, v in raw_p], method="fdr_bh")
    for (col, i, _v), a in zip(raw_p, adj):
        summary_df.loc[i, f"{col}_fdr"] = a

summary_df.to_csv(f"{OUT}/ici_validation_summary.csv", index=False)
print(f"\n=== ICI VALIDATION SUMMARY (numbers for the paper) ===")
print(f"Pre-registered primary endpoint: {PRIMARY_ENDPOINT} in {PRIMARY_COHORT}. "
      f"All other p-values below are secondary/exploratory (BH-FDR corrected columns "
      f"end in _fdr).")
print(summary_df.to_string(index=False))

# ============================================================
# NEW ANALYSES: IMvigor210 specific incremental value and subgroups
# ============================================================
print("\n=== IMvigor210 NOVELTY ANALYSES ===")
import statsmodels.api as sm
from scipy.stats import kruskal, chi2

clin_path = "imvigor210_clin.csv"
assign_path = "outputs/IMvigor210_assignments.csv"
if os.path.exists(clin_path) and os.path.exists(assign_path):
    clin_imv = pd.read_csv(clin_path, index_col=0)
    assign_imv = pd.read_csv(assign_path, index_col=0)
    
    # Merge
    imv = clin_imv.join(assign_imv[["hot_score", "response"]], how="inner")
    imv = imv.dropna(subset=["hot_score"])

    # 1. Construct Validation (Kruskal-Wallis vs Immune phenotype)
    pheno_col = "Immune phenotype"
    if pheno_col in imv.columns:
        valid_imv = imv.dropna(subset=[pheno_col])
        groups = [group["hot_score"].values for name, group in valid_imv.groupby(pheno_col)]
        kw_stat, kw_p = kruskal(*groups)
        print(f"\n1. CONSTRUCT VALIDATION: Kruskal-Wallis p={kw_p:.2e} across {pheno_col}")
        
        plt.figure(figsize=(5, 4))
        sns.boxplot(data=valid_imv, x=pheno_col, y="hot_score", palette="Set2")
        sns.stripplot(data=valid_imv, x=pheno_col, y="hot_score", color="black", size=3, alpha=0.5)
        plt.title(f"Construct Validation: Immune phenotype\nKruskal-Wallis p={kw_p:.2e}")
        plt.tight_layout()
        plt.savefig(f"{OUT}/IMvigor210_construct_validation.png", dpi=180)
        plt.close()
        
    # 2. Incremental Value over TMB and Neoantigen
    def run_incremental(df, cov_col, name):
        df_sub = df.dropna(subset=["response", "hot_score", cov_col]).copy()
        if len(df_sub) < 20: return None
        
        X_base = sm.add_constant(df_sub[cov_col].astype(float))
        y = df_sub["response"].astype(float)
        try:
            model_base = sm.Logit(y, X_base).fit(disp=0)
            pred_base = model_base.predict(X_base)
            auc_base = roc_auc_score(y, pred_base)
        except Exception: return None
        
        X_full = X_base.copy()
        X_full["hot_score"] = df_sub["hot_score"].astype(float)
        try:
            model_full = sm.Logit(y, X_full).fit(disp=0)
            pred_full = model_full.predict(X_full)
            auc_full = roc_auc_score(y, pred_full)
        except Exception: return None
        
        lr_stat = 2 * (model_full.llf - model_base.llf)
        lr_p = chi2.sf(lr_stat, 1)
        
        rng = np.random.RandomState(42)
        n_boot = 2000
        idx = np.arange(len(y))
        y_arr = y.values
        xb_arr = X_base.values
        xf_arr = X_full.values
        deltas = []
        for _ in range(n_boot):
            b_idx = rng.choice(idx, size=len(idx), replace=True)
            if len(np.unique(y_arr[b_idx])) < 2: continue
            try:
                mb = sm.Logit(y_arr[b_idx], xb_arr[b_idx]).fit(disp=0)
                mf = sm.Logit(y_arr[b_idx], xf_arr[b_idx]).fit(disp=0)
                pb = mb.predict(xb_arr[b_idx])
                pf = mf.predict(xf_arr[b_idx])
                deltas.append(roc_auc_score(y_arr[b_idx], pf) - roc_auc_score(y_arr[b_idx], pb))
            except Exception:
                continue
                
        if deltas:
            d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
        else:
            d_lo = d_hi = np.nan
            
        return {
            "covariate": name,
            "n": len(df_sub),
            "auc_base": auc_base,
            "auc_full": auc_full,
            "delta_auc": auc_full - auc_base,
            "delta_lo": d_lo,
            "delta_hi": d_hi,
            "lrt_p": lr_p
        }

    res_tmb = run_incremental(imv, "FMOne mutation burden per MB", "TMB")
    res_neo = run_incremental(imv, "Neoantigen burden per MB", "Neoantigen")
    
    print("\n2. INCREMENTAL VALUE:")
    inc_results = []
    for r in [res_tmb, res_neo]:
        if r:
            inc_results.append(r)
            sig = "SIGNIFICANTLY" if r['lrt_p'] < 0.05 else "DOES NOT significantly"
            print(f"  Beyond {r['covariate']} (n={r['n']}):")
            print(f"    AUC {r['covariate']} alone: {r['auc_base']:.3f}")
            print(f"    AUC + hot_score: {r['auc_full']:.3f}")
            print(f"    Delta AUC: {r['delta_auc']:.3f} [95% CI: {r['delta_lo']:.3f} to {r['delta_hi']:.3f}]")
            print(f"    LRT p-value: {r['lrt_p']:.2e} -> hot_score {sig} improve fit.")
            
    # 3. Subgroup Robustness
    print("\n3. SUBGROUP ROBUSTNESS (AUC of hot_score alone):")
    sub_results = []
    
    def run_subgroup(df, col):
        if col not in df.columns: return
        for val, grp in df.groupby(col):
            d_sub = grp.dropna(subset=["response", "hot_score"])
            n = len(d_sub)
            if n < 15 or d_sub["response"].nunique() < 2:
                print(f"  {col}={val}: n={n}, too small to test.")
                continue
            
            try:
                auc, lo, hi, _ = bootstrap_auc_ci(d_sub["response"], d_sub["hot_score"], random_state=42)
                sub_results.append({"subgroup_var": col, "subgroup_val": val, "n": n, "auc": auc, "ci_lo": lo, "ci_hi": hi})
                print(f"  {col}={val}: n={n} | AUC={auc:.3f} [95% CI: {lo:.3f} to {hi:.3f}]")
            except Exception as e:
                print(f"  {col}={val}: n={n} | Failed: {e}")

    run_subgroup(imv, "Received platinum")
    run_subgroup(imv, "TCGA Subtype")
    
    if inc_results: pd.DataFrame(inc_results).to_csv(f"{OUT}/imvigor210_incremental_value.csv", index=False)
    if sub_results: pd.DataFrame(sub_results).to_csv(f"{OUT}/imvigor210_subgroups.csv", index=False)
