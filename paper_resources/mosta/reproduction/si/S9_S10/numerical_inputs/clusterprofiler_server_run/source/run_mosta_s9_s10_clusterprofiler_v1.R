#!/usr/bin/env Rscript

# Recompute MOSTA S9/S10 GO enrichment with clusterProfiler on the corrected
# latest-package Brain programs/phases.  The server output is immutable and
# records all symbol mappings, tested terms, package versions, input hashes,
# and the exact R session.  Plotting is deliberately excluded.

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(clusterProfiler)
  library(digest)
  library(jsonlite)
  library(org.Mm.eg.db)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: run_mosta_s9_s10_clusterprofiler.R SHARED_RUN OUTPUT_DIR")
}

shared_run <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(dirname(args[[2]]), mustWork = TRUE)
output_dir <- file.path(output_dir, basename(args[[2]]))
if (file.exists(output_dir)) {
  stop(sprintf("Refusing to overwrite existing output: %s", output_dir))
}

stage_dir <- sprintf("%s.stage-%d", output_dir, Sys.getpid())
if (file.exists(stage_dir)) {
  stop(sprintf("Unexpected existing staging directory: %s", stage_dir))
}
dir.create(stage_dir, recursive = TRUE, showWarnings = FALSE)
on.exit({
  if (dir.exists(stage_dir)) unlink(stage_dir, recursive = TRUE, force = TRUE)
}, add = TRUE)
for (name in c("source", "inputs", "tables", "provenance")) {
  dir.create(file.path(stage_dir, name), recursive = TRUE, showWarnings = FALSE)
}

script_args <- commandArgs(trailingOnly = FALSE)
script_token <- script_args[grepl("^--file=", script_args)]
if (length(script_token) != 1L) stop("Could not resolve the executing R script")
script_path <- normalizePath(sub("^--file=", "", script_token[[1]]), mustWork = TRUE)
file.copy(script_path, file.path(stage_dir, "source", basename(script_path)), overwrite = FALSE)

assignment_path <- file.path(
  shared_run, "s8_gene_programs", "brain_hvg_ward_k2_assignments.csv"
)
phase_path <- file.path(
  shared_run, "s10_developmental_wave", "s10_top1000_dp3_assignments.csv"
)
summary_path <- file.path(shared_run, "summary.json")
settings_path <- file.path(
  shared_run, "s8_gene_programs", "s8_gene_program_settings.json"
)
required_paths <- c(assignment_path, phase_path, summary_path, settings_path)
if (!all(file.exists(required_paths))) {
  stop(sprintf("Missing required inputs: %s", paste(required_paths[!file.exists(required_paths)], collapse = "; ")))
}

assignments <- read.csv(assignment_path, stringsAsFactors = FALSE, check.names = FALSE)
phases <- read.csv(phase_path, stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(identical(names(assignments), c("profile", "cluster")))
stopifnot(all(c("profile", "phase") %in% names(phases)))
stopifnot(nrow(assignments) == 2000L)
stopifnot(nrow(phases) == 1000L)
stopifnot(identical(sort(unique(assignments$cluster)), c(1L, 2L)))
stopifnot(identical(sort(unique(phases$phase)), c(1L, 2L, 3L)))
stopifnot(!anyDuplicated(assignments$profile))
stopifnot(!anyDuplicated(phases$profile))
stopifnot(all(phases$profile %in% assignments$profile))

file.copy(assignment_path, file.path(stage_dir, "inputs", basename(assignment_path)))
file.copy(phase_path, file.path(stage_dir, "inputs", basename(phase_path)))

background_symbols <- sort(unique(trimws(as.character(assignments$profile))))
if (length(background_symbols) != 2000L || any(!nzchar(background_symbols))) {
  stop("The corrected Brain original-HVG universe must contain 2,000 non-empty symbols")
}

mapping_raw <- suppressMessages(
  AnnotationDbi::select(
    org.Mm.eg.db,
    keys = background_symbols,
    keytype = "SYMBOL",
    columns = c("SYMBOL", "ENTREZID")
  )
)
mapping_raw$SYMBOL <- as.character(mapping_raw$SYMBOL)
mapping_raw$ENTREZID <- as.character(mapping_raw$ENTREZID)
mapping_valid <- mapping_raw[
  !is.na(mapping_raw$ENTREZID) & nzchar(mapping_raw$ENTREZID),
  c("SYMBOL", "ENTREZID"),
  drop = FALSE
]
mapping_valid <- unique(mapping_valid)
mapping_valid <- mapping_valid[order(mapping_valid$SYMBOL, mapping_valid$ENTREZID), , drop = FALSE]
write.csv(mapping_raw, file.path(stage_dir, "tables", "background_symbol_to_entrez_all_rows.csv"), row.names = FALSE)
write.csv(mapping_valid, file.path(stage_dir, "tables", "background_symbol_to_entrez_valid.csv"), row.names = FALSE)

mapped_symbols <- unique(mapping_valid$SYMBOL)
background_entrez <- sort(unique(mapping_valid$ENTREZID))
ambiguous_counts <- table(mapping_valid$SYMBOL)
n_ambiguous_symbols <- sum(ambiguous_counts > 1L)
if (length(background_entrez) < 1000L) {
  stop(sprintf("Unexpectedly low mapped Brain background: %d unique Entrez IDs", length(background_entrez)))
}

map_symbols <- function(symbols) {
  symbols <- sort(unique(trimws(as.character(symbols))))
  mapped <- mapping_valid[mapping_valid$SYMBOL %in% symbols, , drop = FALSE]
  list(
    input_symbols = symbols,
    mapped_symbols = sort(unique(mapped$SYMBOL)),
    entrez = sort(unique(mapped$ENTREZID))
  )
}

run_enrich_go <- function(query_symbols, query_id) {
  mapped <- map_symbols(query_symbols)
  if (length(mapped$entrez) < 5L) {
    stop(sprintf("Query %s has fewer than five mapped Entrez IDs", query_id))
  }
  result <- suppressMessages(
    clusterProfiler::enrichGO(
      gene = mapped$entrez,
      universe = background_entrez,
      OrgDb = org.Mm.eg.db,
      keyType = "ENTREZID",
      ont = "ALL",
      pAdjustMethod = "BH",
      pvalueCutoff = 1.0,
      qvalueCutoff = 1.0,
      minGSSize = 5L,
      maxGSSize = 500L,
      readable = TRUE,
      pool = TRUE
    )
  )
  table <- as.data.frame(result)
  if (!nrow(table)) stop(sprintf("clusterProfiler returned no tested terms for %s", query_id))
  required <- c("ID", "Description", "GeneRatio", "BgRatio", "pvalue", "p.adjust", "qvalue", "geneID", "Count", "ONTOLOGY")
  if (!all(required %in% names(table))) {
    stop(sprintf("clusterProfiler result for %s is missing columns: %s", query_id, paste(setdiff(required, names(table)), collapse = ", ")))
  }
  table$query_id <- query_id
  table$query_symbol_count <- length(mapped$input_symbols)
  table$query_mapped_symbol_count <- length(mapped$mapped_symbols)
  table$query_entrez_count <- length(mapped$entrez)
  table$background_symbol_count <- length(background_symbols)
  table$background_mapped_symbol_count <- length(mapped_symbols)
  table$background_entrez_count <- length(background_entrez)
  table$clusterprofiler_pool <- TRUE
  table$multiple_testing_scope <- "one BH family across pooled BP+MF+CC terms per query"
  table <- table[order(table$p.adjust, table$pvalue, -table$Count, table$Description, method = "radix"), , drop = FALSE]

  all_path <- file.path(stage_dir, "tables", sprintf("%s_enrichGO_all.csv", query_id))
  sig_path <- file.path(stage_dir, "tables", sprintf("%s_enrichGO_fdr_lt_0p05.csv", query_id))
  top_path <- file.path(stage_dir, "tables", sprintf("%s_enrichGO_display_top20.csv", query_id))
  write.csv(table, all_path, row.names = FALSE)
  significant <- table[is.finite(table$p.adjust) & table$p.adjust < 0.05, , drop = FALSE]
  write.csv(significant, sig_path, row.names = FALSE)
  display <- head(significant, 20L)
  write.csv(display, top_path, row.names = FALSE)

  input_table <- data.frame(
    SYMBOL = mapped$input_symbols,
    mapped = mapped$input_symbols %in% mapped$mapped_symbols,
    stringsAsFactors = FALSE
  )
  write.csv(input_table, file.path(stage_dir, "inputs", sprintf("%s_query_symbols.csv", query_id)), row.names = FALSE)

  list(
    query_id = query_id,
    input_symbol_count = length(mapped$input_symbols),
    mapped_symbol_count = length(mapped$mapped_symbols),
    unique_entrez_count = length(mapped$entrez),
    tested_term_count = nrow(table),
    significant_term_count_fdr_lt_0p05 = nrow(significant),
    displayed_term_count = nrow(display),
    minimum_p_adjust = min(table$p.adjust, na.rm = TRUE),
    top_term = as.character(table$Description[[1]]),
    top_term_id = as.character(table$ID[[1]]),
    top_term_ontology = as.character(table$ONTOLOGY[[1]])
  )
}

summaries <- list()
for (cluster_id in c(1L, 2L)) {
  query_id <- sprintf("s9_pattern_%d", cluster_id)
  query <- assignments$profile[assignments$cluster == cluster_id]
  summaries[[query_id]] <- run_enrich_go(query, query_id)
}
for (phase_id in c(1L, 2L, 3L)) {
  query_id <- sprintf("s10_phase_%d", phase_id)
  query <- phases$profile[phases$phase == phase_id]
  summaries[[query_id]] <- run_enrich_go(query, query_id)
}

summary_table <- do.call(
  rbind,
  lapply(summaries, function(value) as.data.frame(value, stringsAsFactors = FALSE))
)
rownames(summary_table) <- NULL
write.csv(summary_table, file.path(stage_dir, "tables", "clusterprofiler_query_summary.csv"), row.names = FALSE)

orgdb_metadata <- as.data.frame(AnnotationDbi::metadata(org.Mm.eg.db), stringsAsFactors = FALSE)
write.csv(orgdb_metadata, file.path(stage_dir, "provenance", "org.Mm.eg.db_metadata.csv"), row.names = FALSE)
orgdb_sqlite <- AnnotationDbi::dbfile(org.Mm.eg.db)

sha256 <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
file_identity <- function(path) {
  info <- file.info(path)
  list(
    path = normalizePath(path, mustWork = TRUE),
    size_bytes = unname(as.numeric(info$size)),
    sha256 = sha256(path)
  )
}

manifest <- list(
  schema_version = 1L,
  dataset = "MOSTA",
  panels = c("S9a-b", "S10b-d"),
  status = "COMPLETE",
  computation_only = TRUE,
  style_source = "not used in computation; submitted SI and historical MOSTA plotters are render-only authority",
  calculation_contract = list(
    query_source = "corrected latest-package S8 Ward-k2 programs and S10 DP3 phases",
    background = "same 2,000 eligible Brain original-HVG symbols used by S8",
    organism = "Mus musculus",
    orgdb = "org.Mm.eg.db",
    key_type = "ENTREZID",
    ontology = "ALL",
    ontologies_pooled = TRUE,
    multiple_testing = "Benjamini-Hochberg over one pooled BP+MF+CC family per query",
    pvalue_cutoff_compute = 1.0,
    qvalue_cutoff_compute = 1.0,
    display_significance_cutoff = "p.adjust < 0.05",
    min_gene_set_size = 5L,
    max_gene_set_size = 500L,
    display_order = "p.adjust asc, pvalue asc, Count desc, Description asc",
    display_top_n = 20L,
    no_term_cherry_picking = TRUE
  ),
  mapping = list(
    background_input_symbols = length(background_symbols),
    background_mapped_symbols = length(mapped_symbols),
    background_unique_entrez = length(background_entrez),
    ambiguous_symbol_count = unname(n_ambiguous_symbols),
    policy = "retain all valid SYMBOL-to-ENTREZID mappings; deduplicate Entrez IDs per query and universe"
  ),
  inputs = list(
    s8_assignments = file_identity(assignment_path),
    s10_phase_assignments = file_identity(phase_path),
    shared_summary = file_identity(summary_path),
    s8_settings = file_identity(settings_path)
  ),
  queries = summaries,
  software = list(
    R = R.version.string,
    clusterProfiler = as.character(packageVersion("clusterProfiler")),
    org.Mm.eg.db = as.character(packageVersion("org.Mm.eg.db")),
    AnnotationDbi = as.character(packageVersion("AnnotationDbi")),
    DOSE = as.character(packageVersion("DOSE")),
    enrichplot = as.character(packageVersion("enrichplot")),
    orgdb_sqlite = file_identity(orgdb_sqlite)
  )
)
write_json(manifest, file.path(stage_dir, "manifest.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
capture.output(sessionInfo(), file = file.path(stage_dir, "provenance", "sessionInfo.txt"))

all_files_before_complete <- list.files(stage_dir, recursive = TRUE, full.names = TRUE, all.files = FALSE)
identities <- lapply(all_files_before_complete[file.info(all_files_before_complete)$isdir == FALSE], file_identity)
names(identities) <- sub(paste0("^", stage_dir, "/"), "", names(identities))
write_json(identities, file.path(stage_dir, "provenance", "file_identities_before_complete.json"), pretty = TRUE, auto_unbox = TRUE)
writeLines("COMPLETE", file.path(stage_dir, "COMPLETE"), useBytes = TRUE)

if (!file.rename(stage_dir, output_dir)) {
  stop(sprintf("Failed atomic rename from %s to %s", stage_dir, output_dir))
}
on.exit(NULL, add = FALSE)

files <- list.files(output_dir, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE)
dirs <- files[file.info(files)$isdir]
regular <- files[!file.info(files)$isdir]
if (length(regular)) Sys.chmod(regular, mode = "0444")
if (length(dirs)) Sys.chmod(dirs, mode = "0555")
Sys.chmod(output_dir, mode = "0555")

cat(sprintf("COMPLETE %s\n", output_dir))
print(summary_table)
