#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(4, 5))) {
  stop(paste(
    "Usage: enrich_go.R",
    "<assignments.csv> <expression.csv> <output_dir> <library_path>",
    "[top2000|all_detected]"
  ))
}
assignments_path <- normalizePath(args[[1]], mustWork = TRUE)
expression_path <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- args[[3]]
library_path <- args[[4]]
universe_mode <- if (length(args) == 5) args[[5]] else "top2000"
if (!(universe_mode %in% c("top2000", "all_detected"))) {
  stop("universe_mode must be top2000 or all_detected")
}
if (nzchar(library_path)) .libPaths(c(normalizePath(library_path, mustWork = TRUE), .libPaths()))
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Mm.eg.db)
  library(AnnotationDbi)
})
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

assignments <- read.csv(assignments_path, stringsAsFactors = FALSE, check.names = FALSE)
expression <- read.csv(expression_path, stringsAsFactors = FALSE, check.names = FALSE, row.names = 1)
assignments$gene_symbol <- toupper(trimws(assignments$gene_symbol))
assignments <- assignments[!is.na(assignments$gene_symbol) & assignments$gene_symbol != "", , drop = FALSE]

all_symbols <- unique(assignments$gene_symbol)
all_mouse_symbols <- AnnotationDbi::keys(org.Mm.eg.db, keytype = "SYMBOL")
mapping <- AnnotationDbi::select(
  org.Mm.eg.db,
  keys = all_mouse_symbols,
  keytype = "SYMBOL",
  columns = c("SYMBOL", "ENTREZID")
)
mapping <- mapping[!is.na(mapping$ENTREZID), c("SYMBOL", "ENTREZID")]
mapping <- unique(mapping)
mapping$input_symbol <- toupper(mapping$SYMBOL)
write.csv(mapping, file.path(output_dir, "mouse_symbol_to_entrez_mapping.csv"), row.names = FALSE)

assigned <- merge(assignments, mapping, by.x = "gene_symbol", by.y = "input_symbol")
if (universe_mode == "top2000") {
  universe <- unique(assigned$ENTREZID)
} else {
  expression_symbols <- toupper(trimws(rownames(expression)))
  expression_symbols <- unique(expression_symbols[!is.na(expression_symbols) & expression_symbols != ""])
  expression_mapping <- mapping[mapping$input_symbol %in% expression_symbols, , drop = FALSE]
  universe <- unique(expression_mapping$ENTREZID)
}
if (length(universe) < 500) stop("Mapped top-2000 universe is unexpectedly small")

run_one <- function(pattern_id) {
  query <- unique(assigned$ENTREZID[assigned$pattern == pattern_id])
  if (length(query) < 20) stop(paste("Pattern", pattern_id, "query is unexpectedly small"))
  result <- enrichGO(
    gene = query,
    universe = universe,
    OrgDb = org.Mm.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 5,
    maxGSSize = 500,
    readable = TRUE
  )
  raw <- as.data.frame(result)
  if (nrow(raw) == 0) stop(paste("Pattern", pattern_id, "returned no GO BP terms"))
  raw$pattern <- pattern_id
  raw$significant_fdr_0p05 <- raw$p.adjust < 0.05
  significant <- result
  significant@result <- significant@result[significant@result$p.adjust < 0.05, , drop = FALSE]
  simplified <- if (nrow(significant@result)) {
    tryCatch(
      as.data.frame(simplify(
        significant,
        cutoff = 0.7,
        by = "p.adjust",
        select_fun = min,
        measure = "Wang"
      )),
      error = function(error) as.data.frame(significant)
    )
  } else {
    raw[0, , drop = FALSE]
  }
  simplified$pattern <- rep(pattern_id, nrow(simplified))
  simplified$significant_fdr_0p05 <- if (nrow(simplified)) {
    simplified$p.adjust < 0.05
  } else {
    logical(0)
  }
  write.csv(raw, file.path(output_dir, paste0("pattern_", pattern_id, "_enrichGO_all.csv")), row.names = FALSE)
  write.csv(simplified, file.path(output_dir, paste0("pattern_", pattern_id, "_enrichGO_simplified.csv")), row.names = FALSE)
  data.frame(
    pattern = pattern_id,
    assigned_raw_genes = length(unique(assignments$raw_gene[assignments$pattern == pattern_id])),
    mapped_gene_symbols = length(unique(assignments$gene_symbol[assignments$pattern == pattern_id])),
    mapped_entrez_query = length(query),
    tested_terms = nrow(raw),
    significant_terms_fdr_0p05 = sum(raw$p.adjust < 0.05),
    simplified_significant_terms_fdr_0p05 = sum(simplified$p.adjust < 0.05)
  )
}

summary <- do.call(rbind, lapply(sort(unique(assignments$pattern)), run_one))
summary$universe_mode <- universe_mode
summary$mapped_universe_entrez <- length(universe)
write.csv(summary, file.path(output_dir, "analysis_summary.csv"), row.names = FALSE)
writeLines(c(
  paste0("clusterProfiler_version=", packageVersion("clusterProfiler")),
  paste0("org.Mm.eg.db_version=", packageVersion("org.Mm.eg.db")),
  paste0("universe_mode=", universe_mode),
  paste0("mapped_universe_entrez_n=", length(universe)),
  "organism_mapping=axolotl annotated ortholog symbols mapped case-insensitively to mouse SYMBOL/ENTREZID",
  paste0(
    "primary_analysis=clusterProfiler::enrichGO, BP, BH correction, ",
    if (universe_mode == "top2000") {
      "top-2000 clustered genes as conditional eligible universe"
    } else {
      "all detected expression genes with mouse annotation as assay background"
    }
  ),
  "redundancy_reduction=clusterProfiler::simplify, semantic cutoff 0.7",
  "inference_note=orthology-based exploratory functional annotation"
), file.path(output_dir, "analysis_contract.txt"))
capture.output(sessionInfo(), file = file.path(output_dir, "sessionInfo.txt"))
