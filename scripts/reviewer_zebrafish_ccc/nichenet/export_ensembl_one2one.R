#!/usr/bin/env Rscript

# Export a provenance-complete zebrafish -> mouse Ensembl Compara mapping.
#
# No packages are installed by this script.  For an online export, biomaRt,
# dplyr, and jsonlite must already be present in the isolated R library.
# readr is optional; base R handles CSV/gzip when it is unavailable.
# Passing --raw-input makes the filtering step fully offline and is useful for
# replaying a previously frozen BioMart response.

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

csv_backend <- if (requireNamespace("readr", quietly = TRUE)) "readr" else "base_r"
read_csv_table <- function(path) {
  if (csv_backend == "readr") {
    return(readr::read_csv(path, show_col_types = FALSE))
  }
  if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    connection <- gzfile(path, open = "rt")
    on.exit(close(connection), add = TRUE)
    frame <- utils::read.csv(
      connection,
      check.names = FALSE,
      stringsAsFactors = FALSE,
      comment.char = ""
    )
  } else {
    frame <- utils::read.csv(
      path,
      check.names = FALSE,
      stringsAsFactors = FALSE,
      comment.char = ""
    )
  }
  frame
}
write_csv_table <- function(frame, path) {
  if (csv_backend == "readr") {
    readr::write_csv(frame, path)
    return(invisible(NULL))
  }
  if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    connection <- gzfile(path, open = "wt")
    on.exit(close(connection), add = TRUE)
    utils::write.csv(
      as.data.frame(frame), connection, row.names = FALSE, na = ""
    )
  } else {
    utils::write.csv(
      as.data.frame(frame), path, row.names = FALSE, na = ""
    )
  }
  invisible(NULL)
}

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
mapping_policy <- arg_value("--mapping-policy", "strict_confidence1")
allowed_mapping_policies <- c(
  "strict_confidence1",
  "one2one_bijective_all_confidence"
)
if (is.null(out_dir)) stop("--out-dir is required")
if (!mapping_policy %in% allowed_mapping_policies) {
  stop(
    "--mapping-policy must be one of: ",
    paste(allowed_mapping_policies, collapse = ", ")
  )
}
analysis_tier <- if (mapping_policy == "strict_confidence1") {
  "primary"
} else {
  "sensitivity"
}
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
attribute_display_headers <- c(
  ensembl_gene_id = "Gene stable ID",
  external_gene_name = "Gene name",
  mmusculus_homolog_ensembl_gene = "Mouse gene stable ID",
  mmusculus_homolog_associated_gene_name = "Mouse gene name",
  mmusculus_homolog_orthology_type = "Mouse homology type",
  mmusculus_homolog_orthology_confidence = "Mouse orthology confidence [0 low, 1 high]",
  mmusculus_homolog_perc_id = "%id. target Mouse gene identical to query gene"
)

standardize_attribute_headers <- function(frame) {
  resolved <- vapply(attributes_requested, function(attribute) {
    candidates <- c(attribute, unname(attribute_display_headers[[attribute]]))
    found <- candidates[candidates %in% colnames(frame)]
    if (length(found) == 0) return(NA_character_)
    found[[1]]
  }, character(1))
  missing <- names(resolved)[is.na(resolved)]
  if (length(missing) > 0) {
    accepted <- vapply(missing, function(attribute) {
      paste0(
        attribute,
        " or ",
        dQuote(unname(attribute_display_headers[[attribute]]))
      )
    }, character(1))
    stop(
      "BioMart input lacks required attributes: ",
      paste(accepted, collapse = "; ")
    )
  }
  standardized <- frame[, unname(resolved), drop = FALSE]
  colnames(standardized) <- attributes_requested
  list(
    data = standardized,
    resolved_header_map = as.list(resolved),
    original_headers = as.list(colnames(frame))
  )
}

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
  source_input_record <- NULL
} else {
  if (!file.exists(raw_input)) stop("--raw-input does not exist: ", raw_input)
  raw <- read_csv_table(raw_input)
  source_mode <- "frozen_raw_input"
  source_path <- normalizePath(raw_input)
  source_input_record <- list(
    path = source_path,
    size_bytes = unname(file.info(raw_input)$size),
    md5 = unname(tools::md5sum(raw_input))
  )
}

header_audit <- standardize_attribute_headers(raw)
raw <- header_audit$data

raw_path <- file.path(out_dir, "ensembl_compara_drerio_to_mouse_raw.csv.gz")
write_csv_table(raw, raw_path)

standardized <- raw %>%
  transmute(
    zebrafish_ensembl_gene = coalesce(as.character(ensembl_gene_id), ""),
    zebrafish_symbol = coalesce(trimws(as.character(external_gene_name)), ""),
    mouse_ensembl_gene = coalesce(as.character(mmusculus_homolog_ensembl_gene), ""),
    mouse_symbol = coalesce(
      trimws(as.character(mmusculus_homolog_associated_gene_name)), ""
    ),
    orthology_type = coalesce(as.character(mmusculus_homolog_orthology_type), ""),
    orthology_confidence = suppressWarnings(as.numeric(mmusculus_homolog_orthology_confidence)),
    mouse_percent_identity = suppressWarnings(as.numeric(mmusculus_homolog_perc_id))
  )

one_to_one <- standardized %>%
  filter(
    nzchar(zebrafish_symbol),
    nzchar(mouse_symbol),
    orthology_type == "ortholog_one2one"
  )
policy_candidates <- if (mapping_policy == "strict_confidence1") {
  one_to_one %>% filter(orthology_confidence == 1)
} else {
  one_to_one
}
policy_candidates <- policy_candidates %>%
  mutate(
    zebrafish_symbol_key = tolower(zebrafish_symbol),
    mouse_symbol_key = tolower(mouse_symbol)
  ) %>%
  distinct(zebrafish_symbol_key, mouse_symbol_key, .keep_all = TRUE)

z_degree <- policy_candidates %>% count(zebrafish_symbol_key, name = "z_degree")
m_degree <- policy_candidates %>% count(mouse_symbol_key, name = "m_degree")
mapping <- policy_candidates %>%
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
if (nrow(mapping) == 0) {
  stop("No mapping pairs remain under --mapping-policy ", mapping_policy)
}

mapping_filename <- if (mapping_policy == "strict_confidence1") {
  "ensembl_compara_drerio_to_mouse_strict_one2one.csv"
} else {
  "ensembl_compara_drerio_to_mouse_one2one_bijective_all_confidence.csv"
}
mapping_path <- file.path(out_dir, mapping_filename)
write_csv_table(mapping, mapping_path)

selected_confidence_counts <- mapping %>%
  mutate(confidence_label = ifelse(
    is.na(orthology_confidence), "NA", as.character(orthology_confidence)
  )) %>%
  count(confidence_label, name = "n")

manifest <- list(
  schema_version = 2,
  workflow = "ensembl_compara_zebrafish_mouse_one2one_bijective_export",
  status = "complete",
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  ensembl_release = ensembl_version,
  biomart = "genes",
  dataset = "drerio_gene_ensembl",
  mapping_policy = mapping_policy,
  analysis_tier = analysis_tier,
  primary_claim_allowed = mapping_policy == "strict_confidence1",
  mapping_label = if (mapping_policy == "strict_confidence1") {
    "primary: Ensembl116 ortholog_one2one confidence=1 casefold-symbol-bijective"
  } else {
    "sensitivity only: Ensembl116 ortholog_one2one confidence unfiltered casefold-symbol-bijective"
  },
  source_mode = source_mode,
  source_path = source_path,
  source_input = source_input_record,
  attributes = attributes_requested,
  accepted_display_headers = as.list(attribute_display_headers),
  resolved_header_map = header_audit$resolved_header_map,
  original_input_headers = header_audit$original_headers,
  filter = list(
    orthology_type = "ortholog_one2one",
    orthology_confidence_policy = if (mapping_policy == "strict_confidence1") {
      "require_equal_1"
    } else {
      "not_filtered"
    },
    nonempty_symbols = TRUE,
    symbol_level_bijection_after_casefold = TRUE
  ),
  counts = list(
    raw_rows = nrow(raw),
    nonempty_ortholog_one2one_rows = nrow(one_to_one),
    policy_candidate_symbol_pairs = nrow(policy_candidates),
    selected_bijective_symbol_pairs = nrow(mapping),
    selected_confidence_counts = setNames(
      as.list(selected_confidence_counts$n),
      selected_confidence_counts$confidence_label
    )
  ),
  mapping_file = mapping_filename,
  output_md5 = list(
    raw = unname(tools::md5sum(raw_path)),
    mapping = unname(tools::md5sum(mapping_path))
  ),
  packages = list(
    R = R.version.string,
    csv_backend = csv_backend,
    biomaRt = if (requireNamespace("biomaRt", quietly = TRUE)) as.character(packageVersion("biomaRt")) else NA_character_,
    dplyr = as.character(packageVersion("dplyr")),
    readr = if (requireNamespace("readr", quietly = TRUE)) as.character(packageVersion("readr")) else NA_character_
  )
)
write_json(manifest, file.path(out_dir, "orthology_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
print(manifest$counts)
