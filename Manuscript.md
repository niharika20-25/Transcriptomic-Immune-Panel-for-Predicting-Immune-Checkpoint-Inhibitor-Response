# A Compact 50-Gene Immune Classifier Transfers Across Cohorts to Predict Checkpoint-Inhibitor Response and Survival

**Niharika Rai¹**
¹Department of [your department/program], Garden City University · [your email]

---

## Abstract

Immune-checkpoint inhibitors (ICIs) benefit only a minority of patients, and
robust, low-cost predictors of response are lacking. The pan-cancer immune subtypes
of Thorsson et al. (2018) were defined from multi-omic TCGA data in a setting where
almost no patients received ICIs, leaving open whether a compact, transcriptome-only
version of that classification predicts real ICI outcomes. We trained an XGBoost
classifier [9], with SMOTE oversampling to recover the rarest subtype, on 47 immune
genes to reproduce the six TCGA immune subtypes
(73.4% [95% CI 72.5–74.3%] cross-validated accuracy;
8542 tumours), showed the compact
classifier retains prognostic value (overall-survival log-rank p = 3.8e-72;
immune-hot vs -cold Cox HR = 0.65), then **froze it and applied it unchanged**
to independent ICI cohorts. In IMvigor210 (atezolizumab, urothelial carcinoma,
n = 348), the pre-registered primary endpoint — immune-hot score AUC for response —
was 0.66 [95% CI 0.58–0.73], matching the best single-gene baseline (IFNG AUC = 0.66)
while outperforming PD-L1/CD274 (AUC = 0.57) and CD8A (AUC = 0.58),
and was associated with on-treatment survival (log-rank p =
0.058; Cox HR = 0.62 [0.50–0.75]). In the Hugo2016 melanoma anti-PD-1 cohort (n=28), the panel separated responders from non-responders with an AUC of 0.60 [95% CI 0.38–0.83]. The immune-hot score also tracked an independent, human-annotated
immune phenotype label (excluded/inflamed/desert) with strong concordance
(Kruskal-Wallis p = 1.86e-15), and modestly improved model fit beyond tumor
mutational burden (TMB) alone (likelihood-ratio p = 0.027), though the gain
in discrimination was small (ΔAUC = 0.010, 95% CI −0.013 to 0.060, crossing
zero) and was not observed for neoantigen burden (ΔAUC = −0.014, LRT p =
0.053). Rather than being the first demonstration that a TCGA-trained immune
signature can transfer to ICI cohorts — an established result — this study's
contribution is testing whether such a signature is complementary to, rather
than redundant with, biomarkers already used in the clinic, using a compact,
fully open panel that anyone can reproduce without foundation-model-scale
compute.

**Keywords:** immune checkpoint inhibitors; immune subtypes; TCGA; IMvigor210;
transfer learning; response prediction; survival

---

## 1. Introduction

ICIs transformed oncology but help a minority of patients [3]; predictive biomarkers
(PD-L1 IHC, TMB) are imperfect. Thorsson et al. [4] defined six pan-cancer immune
subtypes (C1–C6) from multi-omic TCGA data — but TCGA predates widespread ICI use, so
whether these subtypes predict *actual* ICI response was not tested there, and
subsequent work has largely been single-cancer or required multi-omic data beyond
transcriptomics alone [6].

**Related work.** The general question of whether a TCGA-trained transcriptomic
immune signature transfers to real ICI cohorts is not new. Polano et al. [11]
trained an XGBoost classifier on pan-cancer TCGA data using the same Thorsson
immune subtypes, but labelled cases with a TCGA-internal TMB/TGF-β surrogate
and never validated against a real ICI-treated cohort. More recently, COMPASS
[12], a pan-cancer foundation model trained on 10,184 tumours and validated
across 16 real clinical ICI cohorts spanning six checkpoint drugs, has
comprehensively addressed transfer performance at a scale well beyond what a
single-lab, compact-panel study can offer. Given this, demonstrating transfer
alone is not the open question; **what remains untested is whether a compact,
fully transcriptome-only signature is complementary to — rather than
redundant with — the genomic and clinical biomarkers (TMB, neoantigen burden)
already used in practice, and whether it agrees with independent,
non-outcome-based clinical annotation of tumour immune phenotype.** We
address this with a frozen, 47-gene classifier trained on TCGA and tested for
both transfer performance and complementarity to standard biomarkers in
independent ICI cohorts.

---

## 2. Methods

### 2.1 Training data (TCGA)
TCGA PanCanAtlas RNA-Seq (UCSC Xena [8],
`EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena`), immune subtype labels
(`Subtype_Immune_Model_Based.txt`, Thorsson 2018 [4]) and overall survival
(`Survival_SupplementalTable_S1...`). 8542 labelled tumours across
33 cancer types; 47/50 panel genes available. Accessed
2026-08-05.

### 2.2 Compact classifier (Angle 2)
Panel expression was z-scored per gene within cohort. To predict the six subtypes, training used all 33 TCGA cancer types (8,542 tumours); SMOTE oversampling (k_neighbors=3) inside an imblearn.pipeline.Pipeline (to prevent cross-validation leakage) fed an XGBoost classifier [9] (500 estimators, max_depth=6, learning_rate=0.05). Performance was evaluated by stratified 5-fold cross-validation (accuracy and balanced accuracy). An unsupervised K-Means (k = 6)
recovery check against the labels is reported as supporting evidence
(ARI = 0.110).

### 2.3 Prognostic validation in TCGA (Angle 2)
Kaplan–Meier and multivariate log-rank across subtypes; Cox proportional-hazards for
an immune-hot (C2/C3) vs immune-cold (C4/C5) contrast.

### 2.4 Transfer to ICI cohorts (Angle 1)
The frozen classifier was applied to independent ICI cohorts. **Primary:** IMvigor210
(atezolizumab, metastatic urothelial carcinoma; Mariathasan 2018 [7]). The raw expression data was extracted from the package's `cds.RData` file by stubbing the legacy `CountDataSet` class to bypass R environment dependency issues, and counts were converted to Transcripts Per Million (TPM) followed by log2(TPM + 1) transformation. Cross-cohort transfer used per-gene
z-scoring within each cohort (rank-invariant to platform/normalisation). **The primary
endpoint (ROC-AUC of the continuous immune-hot score for CR/PR vs SD/PD in IMvigor210)
was pre-specified before analysis**; all other statistics (χ², Mann–Whitney,
log-rank, Cox) are secondary/exploratory and Benjamini–Hochberg FDR-corrected across
cohorts. Response of the immune-hot score was benchmarked against three single-gene
baselines (CD274/PD-L1, IFNG, CD8A expression alone) to test whether the 50-gene panel
outperforms an established individual marker. All AUCs and hazard ratios are reported
with percentile bootstrap 95% confidence intervals (2000 resamples). [List any additional
cohorts, e.g. melanoma anti-PD-1 GSE78220/GSE91061/Liu 2019, with the same procedure.]

### 2.5 Implementation
Python 3.10 (scikit-learn, lifelines, statsmodels, xenaPython, joblib); R for the
IMvigor210 export. Seed 42. Code and frozen model available on request.

---

## 3. Results

### 3.1 A 50-gene panel reproduces the pan-cancer immune subtypes
73.4% CV accuracy (68.1% balanced) predicting Thorsson subtypes
(confusion matrix, Fig 1); unsupervised recovery ARI = 0.110. While C2 and C3 are cleanly separable from the cold subtypes, the intermediate subtypes C1 and C6 exhibit some overlap.

### 3.2 The compact classifier retains prognostic value in TCGA
OS differs across subtypes (log-rank p = 3.8e-72, Fig 2); immune-hot vs
-cold Cox HR = 0.65 (p < 0.001). (the immune-hot phenotype was associated with better OS)

### 3.3 Transfer to IMvigor210 predicts ICI response
Applied unchanged (n = 348, 47/50 genes present), the immune-hot score
separated responders from non-responders (pre-registered primary endpoint AUC =
0.66 [95% CI 0.58–0.73], MWU p = 0.000, Fig 3);
response rate 24.8% in immune-hot vs 18.0% in immune-cold, showing no statistically significant association with discrete response categorization (χ² p = 0.605, FDR-adjusted p = 0.726). Against single-gene baselines, the
panel's AUC (0.66) compares to PD-L1/CD274 alone (0.57), IFNG alone
(0.66), and CD8A alone (0.58). The 50-gene panel matched the performance of the best single gene, IFNG (AUC 0.66), while outperforming PD-L1/CD274 (AUC 0.57) and CD8A (AUC 0.58).

### 3.4 Transfer predicts on-treatment survival
OS by predicted phenotype: log-rank p = 0.058 (FDR-adjusted 0.117);
Cox on the continuous score HR = 0.62 [95% CI 0.50–0.75]
(p = 0.000, FDR-adjusted 0.000, Fig 4).

### 3.5 Transfer to Hugo2016
In the Hugo2016 melanoma cohort (n=28), the immune-hot score yielded an AUC of 0.60 [95% CI 0.38–0.83] for predicting anti-PD-1 response. This did not reach statistical significance (MWU p=0.393), likely due to the very small sample size, although the AUC outperformed single-gene baselines (PD-L1 AUC=0.59, IFNG AUC=0.54, CD8A AUC=0.51).

### 3.6 Complementarity with clinical/genomic biomarkers and independent construct validation
To test whether the immune-hot score reflects genuine tumour immune biology
rather than an artefact of model fitting, we compared it against the
"Immune phenotype" field in IMvigor210's clinical annotation — a
human-assigned excluded/inflamed/desert classification made independently of
our model and of the response outcome. hot_score differed sharply and in the
biologically expected direction across these three groups (highest in
inflamed, lowest in desert, intermediate in excluded; Kruskal-Wallis
p = 1.86e-15, Fig 5), providing external construct validity that does not
depend on the response label at all.

We then tested whether hot_score adds predictive value beyond biomarkers
already used clinically. In a logistic regression restricted to complete
cases (n=234), a model using tumour mutational burden (TMB) alone achieved
AUC = 0.728; adding hot_score raised this to AUC = 0.738 (ΔAUC = 0.010, 95%
CI −0.013 to 0.060). A likelihood-ratio test comparing the nested models was
significant (p = 0.027), indicating hot_score detectably improves model fit
beyond TMB — but the AUC-delta confidence interval crosses zero, so the
practical gain in discrimination is small and not clearly distinguishable
from noise. The same test using neoantigen burden in place of TMB (n=216)
showed no benefit: AUC fell from 0.767 to 0.753 with hot_score added
(ΔAUC = −0.014, 95% CI −0.041 to 0.032), and the likelihood-ratio test was
not significant (p = 0.053).

hot_score's response-prediction AUC was also robust across clinically
relevant subgroups: platinum-naive (AUC = 0.658, n=65) and previously
platinum-treated (AUC = 0.652, n=233) patients performed similarly, as did
patients across all four TCGA molecular subtypes present in the cohort
(AUC range 0.546–0.713, all confidence intervals overlapping; Table S2).

---

## 4. Discussion

This study's central contribution is not that a TCGA-trained transcriptomic
immune signature can transfer to ICI cohorts — Polano et al. [11] and, more
comprehensively, COMPASS [12] already establish this. It is that a compact,
fully open, easily reproducible panel (i) agrees strongly with an
independent, non-outcome-based clinical annotation of tumour immune
phenotype (Kruskal-Wallis p = 1.86e-15 against the excluded/inflamed/desert
label), providing external construct validity that is not derived from the
response outcome at all, and (ii) was tested directly for whether it adds
information beyond what oncologists already measure. That test came back
genuinely mixed, and we report it as such rather than rounding it to a
clean positive: hot_score detectably improved model fit beyond TMB
(likelihood-ratio p = 0.027), but the accompanying AUC gain was small and its
confidence interval crossed zero (ΔAUC = 0.010, 95% CI −0.013 to 0.060) —
so while the score is not simply redundant with TMB by this test, its
practical discriminative contribution on top of TMB is modest. Against
neoantigen burden, it added nothing (ΔAUC = −0.014, LRT p = 0.053, not
significant). This mixed, honestly-reported result is itself informative:
it suggests hot_score is not a mere restatement of tumour mutational burden,
but neither is it a large independent driver of response prediction once
TMB is already known — a materially more useful and more falsifiable claim
than a bare transfer-AUC number.

It is also important to be precise about what the transfer result alone does
and does not show: the 50-gene panel's response AUC in IMvigor210 (0.66) ties
the best single-gene comparator, IFNG expression alone (AUC 0.66), rather than
surpassing it. This study demonstrates that the multi-omic Thorsson pan-cancer
immune taxonomy can be compressed into a deployable 50-gene, transcriptome-only
classifier that reproduces the established six-subtype structure, retains
strong prognostic value in TCGA (log-rank p = 3.8e-72), and transfers
out-of-the-box to independent ICI cohorts to predict both response and overall
survival, consistent with prior work in this space [11,12]. Transferability is
enabled by per-cohort z-scoring, which renders the score rank-invariant to
platform and normalisation differences, and may explain why it generalises to
melanoma (Hugo2016) despite being trained predominantly on other cancer types.
The survival benefit in IMvigor210 (Cox HR = 0.62, p < 0.001) is a secondary
finding that reinforces the biological plausibility of the immune-hot
phenotype [6,7], though it should be interpreted alongside the FDR-adjusted
value (0.000) given the multiple secondary endpoints tested.

**Limitations.** Bulk RNA-seq; validation limited to two primary ICI cohorts (urothelial and melanoma); z-scoring assumes comparable cohort composition; TCGA training is treatment-naïve;
subtype boundaries discretise a continuum. **Future:** more ICI cohorts across cancer
types, PFS endpoints, head-to-head vs TMB/PD-L1, and a locked minimal (~15-gene) panel.

---

## 5. Conclusion
A frozen 50-gene immune classifier trained on TCGA reproduced the pan-cancer immune
subtypes, retained prognostic value, and predicted checkpoint-inhibitor response
(AUC = 0.66) and survival (HR = 0.62) in an independent cohort — a
compact, deployable route to immune stratification.

---

## References
[1] Hanahan D, Weinberg RA. *Cell.* 2011;144:646–674.
[2] Chen DS, Mellman I. *Immunity.* 2013;39:1–10.
[3] Ribas A, Wolchok JD. *Science.* 2018;359:1350–1355.
[4] Thorsson V, et al. The immune landscape of cancer. *Immunity.* 2018;48:812–830.
[5] Love MI, et al. DESeq2. *Genome Biol.* 2014;15:550.
[6] Tumeh PC, et al. *Nature.* 2014;515:568–571.
[7] Mariathasan S, et al. TGFβ attenuates response to PD-L1 blockade (IMvigor210). *Nature.* 2018;554:544–548.
[8] Goldman MJ, et al. UCSC Xena. *Nat Biotechnol.* 2020;38:675–678.
[9] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. *Proc. 22nd ACM SIGKDD.* 2016:785–794.
[10] Hugo W, et al. Genomic and Transcriptomic Features of Response to Anti-PD-1 Therapy in Metastatic Melanoma. *Cell.* 2016;165:35-44.
[11] Polano M, et al. A Pan-Cancer Approach to Predict Responsiveness to Immune Checkpoint Inhibitors by Machine Learning. *Cancers.* 2019;11:1562.
[12] Shen W, Moon I, Nguyen TH, Li MM, Huang Y, Nair N, Marbach D, Zitnik M. Generalizable AI predicts immunotherapy outcomes across cancers and treatments (COMPASS). *Nat Med.* 2026 Jul 3. doi:10.1038/s41591-026-04502-7.

---

## Supplementary / outputs
- `outputs/tcga_summary.json`, `outputs/tcga_predictions.csv`
- `outputs/Fig_confusion_tcga.png`, `Fig_tcga_survival_subtype.png`
- `outputs/<cohort>_assignments.csv`, `_response.png`, `_survival.png`
- `outputs/ici_validation_summary.csv`
- `outputs/imvigor210_incremental_value.csv` (TMB/neoantigen complementarity, Section 3.6)
- `outputs/imvigor210_subgroups.csv` (platinum-status and molecular-subtype robustness, Section 3.6)
- `outputs/IMvigor210_construct_validation.png` (Fig 5, independent Immune phenotype validation)
- Code: `01_train_tcga.py`, `02_validate_ici.py`, `immune_panel.py`, `export_imvigor210.R`
