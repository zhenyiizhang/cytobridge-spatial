#!/usr/bin/env Rscript
# GO enrichment for the gene groups calculated in the MOSTA notebook.
# Usage: Rscript enrich_go.R BACKGROUND_CSV GROUPS_CSV GROUP_COLUMN PREFIX OUTPUT_DIR

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(clusterProfiler)
  library(org.Mm.eg.db)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("Usage: Rscript enrich_go.R BACKGROUND_CSV GROUPS_CSV GROUP_COLUMN PREFIX OUTPUT_DIR")
}
background <- read.csv(args[[1]], stringsAsFactors = FALSE)
groups <- read.csv(args[[2]], stringsAsFactors = FALSE)
column <- args[[3]]
prefix <- args[[4]]
output <- args[[5]]
stopifnot("profile" %in% names(background), all(c("profile", column) %in% names(groups)))
stopifnot(!anyDuplicated(background$profile), !anyDuplicated(groups$profile))
stopifnot(all(groups$profile %in% background$profile))
dir.create(file.path(output, "tables"), recursive = TRUE, showWarnings = FALSE)

mapping <- suppressMessages(AnnotationDbi::select(
  org.Mm.eg.db, keys = sort(unique(background$profile)),
  keytype = "SYMBOL", columns = c("SYMBOL", "ENTREZID")
))
mapping <- unique(mapping[!is.na(mapping$ENTREZID), c("SYMBOL", "ENTREZID")])
mapping$ENTREZID <- as.character(mapping$ENTREZID)
universe <- sort(unique(mapping$ENTREZID))
write.csv(mapping, file.path(output, "tables", "background_symbol_to_entrez.csv"), row.names = FALSE)

for (group in sort(unique(groups[[column]]))) {
  symbols <- groups$profile[groups[[column]] == group]
  query <- sort(unique(mapping$ENTREZID[mapping$SYMBOL %in% symbols]))
  result <- enrichGO(
    gene = query, universe = universe, OrgDb = org.Mm.eg.db,
    keyType = "ENTREZID", ont = "ALL", pool = TRUE,
    pAdjustMethod = "BH", pvalueCutoff = 1, qvalueCutoff = 1,
    minGSSize = 5, maxGSSize = 500, readable = TRUE
  )
  table <- as.data.frame(result)
  if (!nrow(table)) stop(sprintf("No GO terms for %s %s", column, group))
  table$query_id <- paste0(prefix, group)
  table <- table[order(table$p.adjust, table$pvalue, -table$Count,
                       table$Description, method = "radix"), , drop = FALSE]
  significant <- table[is.finite(table$p.adjust) & table$p.adjust < 0.05, , drop = FALSE]
  stem <- file.path(output, "tables", paste0(prefix, group, "_enrichGO"))
  write.csv(table, paste0(stem, "_all.csv"), row.names = FALSE)
  write.csv(significant, paste0(stem, "_fdr_lt_0p05.csv"), row.names = FALSE)
  write.csv(head(significant, 20), paste0(stem, "_display_top20.csv"), row.names = FALSE)
  message(sprintf("%s%s: %d tested terms, %d with adjusted p < 0.05",
                  prefix, group, nrow(table), nrow(significant)))
}
writeLines(capture.output(sessionInfo()), file.path(output, "sessionInfo.txt"))
