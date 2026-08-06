# Pan-Cancer Immune Classifier → ICI Response & Survival

A compact 50-gene immune classifier trained on TCGA, frozen, and transferred to
immune-checkpoint-inhibitor (ICI) cohorts to predict **response** (Angle 1) and
**survival** (Angle 1+2).

## Files
| File | Role |
|---|---|
| `immune_panel.py` | Shared 50-gene panel, aliases, z-score/hot-score helpers, **baseline single-gene comparators**, **bootstrap CI helpers** |
| `01_train_tcga.py` | Load TCGA, train+CV classifier (with accuracy CI), TCGA survival, **freeze model** |
| `02_validate_ici.py` | Apply frozen model to ICI cohorts; response + survival tests, baseline comparison, bootstrap CIs, FDR correction across secondary endpoints |
| `export_imvigor210.R` | One-time R export of IMvigor210 → 2 CSVs |
| `export_geo_cohort.py` | Generic GEO exporter for adding a **2nd/3rd ICI cohort** (e.g. GSE78220, GSE91061) in the same 2-CSV format |
| `Manuscript_skeleton.md` | Honest manuscript with `{{placeholders}}` to fill |

> **Note:** delete the old `Pan_Cancer_Immune_ICI.ipynb` notebook before submitting/presenting —
> keep only `Pan_Cancer_Immune_ICI_fixed.ipynb`. Having two near-identical notebooks in the
> submission looks like an unresolved draft, not a finished project.

## How to run

### Step 0 — deps
```bash
pip install pandas numpy scikit-learn scipy statsmodels lifelines matplotlib seaborn xenaPython joblib -q
```

### Step 1 — train on TCGA (writes model/immune_clf.joblib + outputs/tcga_*.json/csv/png)
```bash
python 01_train_tcga.py
```
Sanity: panel genes present ≈ 46–50/50; CV accuracy prints; a frozen model appears in `model/`.
If a Xena fetch returns empty (dataset IDs occasionally change), open the datapages at
https://xenabrowser.net/datapages/ (PanCanAtlas hub) and update the 3 dataset IDs at the top.

### Step 2 — export IMvigor210 once (in R)
```r
source("export_imvigor210.R")   # writes imvigor210_expr.csv, imvigor210_clin.csv
```
Package + data: http://research-pub.gene.com/IMvigor210CoreBiologies/ (CC-BY, Mariathasan 2018).

### Step 3 — validate on ICI data (writes outputs/*_response.png, *_survival.png, summary csv)
```bash
python 02_validate_ici.py
```
This now also prints, per cohort: a bootstrap 95% CI on the response AUC and on the
survival HR, and AUC for three single-gene baselines (CD274/PD-L1, IFNG, CD8A) so you
can show the panel adds something beyond a well-known single marker. Secondary p-values
(everything except the pre-registered primary endpoint) get a Benjamini–Hochberg FDR
correction in `outputs/ici_validation_summary.csv` (columns ending `_fdr`).

### Step 3b — add a second ICI cohort (do this before presenting)
One ICI cohort is a weak claim on its own. Add a melanoma anti-PD-1 cohort:
```bash
pip install GEOparse -q
python export_geo_cohort.py GSE78220 hugo2016     # Hugo 2016
python export_geo_cohort.py GSE91061 riaz2017     # Riaz 2017
```
The script prints the real column names in the downloaded metadata — copy the exact
response/OS column names into the `Hugo2016` / `Riaz2017` entries already scaffolded
in `COHORTS` inside `02_validate_ici.py`, then uncomment that entry and re-run Step 3.
GEO series sometimes ship RNA-seq expression as a separate supplementary counts file
rather than per-sample tables; if `export_geo_cohort.py` warns about this, download that
file manually from the GEO accession page and load it with `pandas.read_csv` in the same
genes-x-samples shape as `imvigor210_expr.csv`.

### Step 4 — fill the manuscript
Copy every `{{...}}` from `outputs/tcga_summary.json` and `outputs/ici_validation_summary.csv`.

## Adding more ICI cohorts (strengthens the paper)
Melanoma anti-PD-1 sets make good second cohorts: **GSE78220** (Hugo 2016),
**GSE91061** (Riaz 2017), **Liu 2019** (supplementary). Export each to two CSVs
(`genes × samples` expression, `samples × clinical`) and register them in the
`COHORTS` dict in `02_validate_ici.py` with the right response/OS column names.

---

## How to read your results (and keep it honest)

The headline is the **ICI response AUC / survival HR**, not a perfect number.
`hotscore_auc` in **IMvigor210** is the pre-registered primary endpoint (set at the top
of `02_validate_ici.py`) — lead with that number specifically; treat every other p-value
in the summary as secondary/exploratory and quote its `_fdr`-corrected value alongside
the raw one.

- **AUC ≳ 0.65 and/or survival p < 0.05** → clean positive; strong story.
- **AUC ~0.55–0.65, trends but n.s.** → still reportable as "modest, consistent
  signal from a compact panel"; add a second cohort to strengthen.
- **Null** → reframe as "a transcriptome-only compression of the pan-cancer immune
  subtypes is prognostic in TCGA but does **not** by itself predict ICI response,
  suggesting response needs features beyond bulk immune-gene levels." That is a
  legitimate, publishable negative finding — do not fabricate a positive one.

## Novelty statement (for your cover letter)
The six immune subtypes are known (Thorsson 2018); re-deriving them is not novel.
What is novel here: (1) compressing that multi-omic taxonomy into a 50-gene,
transcriptome-only classifier, and (2) demonstrating **out-of-the-box transfer** of a
treatment-naïve-trained classifier to independent ICI cohorts to predict response and
survival — a test the original study could not do because TCGA predates ICI use.

## Realistic venues
- **Conference:** ISCB regional / IEEE-Springer bioinformatics workshop / institutional
  symposium — a poster or short paper is very appropriate for this scope.
- **Preprint:** post to **bioRxiv** first (free, citable, immediate).
- **Journal:** open/mid-tier bioinformatics or immuno-oncology methods venue. Adding a
  second ICI cohort materially improves acceptance odds.

## Before submitting — delete the old synthetic project
The earlier synthetic-data files (`Pan_Cancer_Immune_Subtype_Analysis.*`, the old
`Manuscript_*.md`, old `Fig1–8`, old CSVs) must NOT travel with this submission.
