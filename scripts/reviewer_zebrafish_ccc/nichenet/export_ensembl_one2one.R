#!/usr/bin/env Rscript

# Export a provenance-complete zebrafish -> mouse Ensembl Compara mapping.
#
# No packages are installed by this script.  For an online export, biomaRt,
# readr, dplyr, and jsonlite must already be present in the isolated R library.
# Passing --raw-input makes the filtering step fully offline and is useful for
# replaying a previously frozen BioMart response.

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (is.na(index)) return(default)
  if (index == length(args)) stop("Missing value after ", flag)
  args[[index + 1]]
}

out_dir <- arg_value("--out-dir")
raw_input <- arg_value("--raw-input")
ensembl_version <- as.integer(arg_value("--ensembl-version", "116"))
if (is.null(out_dir)) stop("--out-dir is required")
if (dir.exists(out_dir) && length(list.files(out_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop("Output directory must be absent or empty: ", out_dir)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

attributes_requested <- c(
  "ensembl_gene_id",
  "external_gene_name",
  "mmusculus_homolog_ensembl_gene",
  "mmusculus_homolog_associated_gene_name",
  "mmusculus_homolog_orthology_type",
  "mmusculus_homolog_orthology_confidence",
  "mmusculus_homolog_perc_id"
)

if (is.null(raw_input)) {
  if (!requireNamespace("biomaRt", quietly = TRUE)) {
    stop("biomaRt is required for an online export; install it in the isolated environment first")
  }
  mart <- biomaRt::useEnsembl(
    biomart = "genes",
    dataset = "drerio_gene_ensembl",
    version = ensembl_version
  )
  available <- biomaRt::listAttributes(mart)$name
  missing_attributes <- setdiff(attributes_requested, available)
  if (length(missing_attributes) > 0) {
    stop("Ensembl release lacks requested attributes: ", paste(missing_attributes, collapse = ", "))
  }
  raw <- biomaRt::getBM(attributes = attributes_requested, mart = mart)
  source_mode <- "biomaRt_query"
  source_path <- NA_character_
} else {
  if (!file.exists(raw_input)) stop("--raw-input does not exist: ", raw_input)
  raw <- read_csv(raw_input, show_col_types = FALSE)
  missing_attributes <- setdiff(attributes_requested, colnames(raw))
  if (length(missing_attributes) > 0) {
    stop("Frozen raw input lacks columns: ", paste(missing_attributes, collapse = ", "))
  }
  raw <- raw[, attributes_requested]
  source_mode <- "frozen_raw_input"
  source_path <- normalizePath(raw_input)
}

raw_path <- file.path(out_dir, "ensembl_compara_drerio_to_mouse_raw.csv.gz")
write_csv(raw, raw_path)

standardized <- raw %>%
  transmute(
    zebrafish_ensembl_gene = as.character(ensembl_gene_id),
    zebrafish_symbol = trimws(as.character(external_gene_name)),
    mouse_ensembl_gene = as.character(mmusculus_homolog_ensembl_gene),
    mouse_symbol = trimws(as.character(mmusculus_homolog_associated_gene_name)),
    orthology_type = as.character(mmusculus_homolog_orthology_type),
    orthology_confidence = suppressWarnings(as.numeric(mmusculus_homolog_orthology_confidence)),
    mouse_percent_identity = suppressWarnings(as.numeric(mmusculus_homolog_perc_id))
  )

confident <- standardized %>%
  filter(
    nzchar(zebrafish_symbol),
    nzchar(mouse_symbol),
    orthology_type == "ortholog_one2one",
    orthology_confidence == 1
  ) %>%
  mutate(
    zebrafish_symbol_key = tolower(zebrafish_symbol),
    mouse_symbol_key = tolower(mouse_symbol)
  ) %>%
  distinct(zebrafish_symbol_key, mouse_symbol_key, .keep_all = TRUE)

z_degree <- confident %>% count(zebrafish_symbol_key, name = "z_degree")
m_degree <- confident %>% count(mouse_symbol_key, name = "m_degree")
strict <- confident %>%
  left_join(z_degree, by = "zebrafish_symbol_key") %>%
  left_join(m_degree, by = "mouse_symbol_key") %>%
  filter(z_degree == 1, m_degree == 1) %>%
  arrange(zebrafish_symbol_key, mouse_symbol_key) %>%
  distinct(zebrafish_symbol_key, .keep_all = TRUE) %>%
  select(
    zebrafish_ensembl_gene,
    zebrafish_symbol,
    mouse_ensembl_gene,
    mouse_symbol,
    orthology_type,
    orthology_confidence,
    mouse_percent_identity
  )

strict_path <- file.path(out_dir, "ensembl_compara_drerio_to_mouse_strict_one2one.csv")
write_csv(strict, strict_path)

manifest <- list(
  schema_version = 1,
  workflow = "ensembl_compara_zebrafish_mouse_strict_one2one_export",
  status = "complete",
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  ensembl_release = ensembl_version,
  biomart = "genes",
  dataset = "drerio_gene_ensembl",
  source_mode = source_mode,
  source_path = source_path,
  attributes = attributes_requested,
  filter = list(
    orthology_type = "ortholog_one2one",
    orthology_confidence = 1,
    nonempty_symbols = TRUE,
    symbol_level_bijection_after_casefold = TRUE
  ),
  counts = list(
    raw_rows = nrow(raw),
    high_confidence_one2one_symbol_pairs = nrow(confident),
    strict_bijective_symbol_pairs = nrow(strict)
  ),
  output_md5 = list(
    raw = unname(tools::md5sum(raw_path)),
    strict = unname(tools::md5sum(strict_path))
  ),
  packages = list(
    R = R.version.string,
    biomaRt = if (requireNamespace("biomaRt", quietly = TRUE)) as.character(packageVersion("biomaRt")) else NA_character_,
    dplyr = as.character(packageVersion("dplyr")),
    readr = as.character(packageVersion("readr"))
  )
)
write_json(manifest, file.path(out_dir, "orthology_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
print(manifest$counts)
