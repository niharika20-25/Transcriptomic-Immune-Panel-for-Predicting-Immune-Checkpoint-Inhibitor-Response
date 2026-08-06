#!/usr/bin/env Rscript
# export_imvigor210_stub.R
# Load IMvigor210 RData directly without the full package.
# Strategy: stub out the CountDataSet class from old DESeq so R can deserialize it.

library(Biobase)

# ---- 1. Stub the old DESeq CountDataSet class ----
# This tricks R into loading cds.RData even without the DESeq package.
setClass("CountDataSet",
  contains = "eSet",   # eSet is in Biobase; CountDataSet extends it
  representation(
    fitInfo       = "environment",
    dispTable     = "character",
    multivariateConditions = "logical"
  )
)

# Provide a generic counts() method for our stub
setGeneric("counts", function(object, ...) standardGeneric("counts"))
setMethod("counts", "CountDataSet", function(object, ...) {
  assayData(object)[["counts"]]
})

# ---- 2. Load the data ----
message("Loading cds.RData ...")
load("IMvigor210CoreBiologies/data/cds.RData")
message("Loaded. Class: ", class(cds))

# ---- 3. Extract counts and clinical data ----
cnt <- cds@assayData$counts
fd  <- cds@featureData@data
pd  <- cds@phenoData@data

message("Counts dim: ", nrow(cnt), " x ", ncol(cnt))
message("featureData cols: ", paste(colnames(fd), collapse=", "))
message("phenoData cols:   ", paste(colnames(pd), collapse=", "))

# ---- 4. Convert raw counts -> TPM ----
# Some gene lengths may be NA; guard against that
len_kb <- fd$length / 1000
len_kb[is.na(len_kb) | len_kb <= 0] <- 1   # fallback

rpk  <- cnt / len_kb
tpm  <- sweep(rpk, 2, colSums(rpk), "/") * 1e6
rownames(tpm) <- fd$Symbol

# Drop NA / blank symbols and deduplicate
ok  <- !is.na(rownames(tpm)) & rownames(tpm) != ""
tpm <- tpm[ok, ]
tpm <- tpm[!duplicated(rownames(tpm)), ]

message("TPM matrix after symbol cleanup: ", nrow(tpm), " genes x ", ncol(tpm), " samples")

# ---- 5. Write CSVs ----
write.csv(as.data.frame(tpm), "imvigor210_expr.csv")
write.csv(pd,                 "imvigor210_clin.csv")

cat("Wrote imvigor210_expr.csv (", nrow(tpm), "genes x", ncol(tpm), "samples)\n")
cat("Wrote imvigor210_clin.csv (", nrow(pd), "samples)\n")
cat("Clinical columns:\n"); print(colnames(pd))
