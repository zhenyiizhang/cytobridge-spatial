#!/usr/bin/env Rscript

# Run one of two cross-species NicheNet-v2 conditions on frozen shared inputs.
#
#   default: official NicheNet-v2 mouse LR network
#   custom:  current zebrafish LR database mapped to strict mouse one-to-one
#            orthologues and used only as a candidate gate
#
# Both modes always use the same official mouse ligand-target matrix.  The
# custom mode therefore does not claim to construct a zebrafish signaling/GRN
# prior from a flat LR table.

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
  library(tibble)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (is.na(index)) return(default)
  if (index == length(args)) stop("Missing value after ", flag)
  args[[index + 1]]
}
as_flag <- function(value) tolower(as.character(value)) %in% c("1", "true", "yes", "y")

mode <- tolower(arg_value("--mode", "default"))
shared_dir <- arg_value("--shared-dir")
prior_dir <- arg_value("--prior-dir")
out_dir <- arg_value("--out-dir")
custom_lr_path <- arg_value(
  "--custom-lr",
  if (!is.null(shared_dir)) file.path(shared_dir, "custom_lr_strict_one2one_mapped.csv") else NULL
)
nichenetr_source <- arg_value("--nichenetr-source")
expected_version <- arg_value("--expected-nichenetr-version", "2.2.1.1")
allow_version_mismatch <- as_flag(arg_value("--allow-version-mismatch", "false"))
allow_installed_nichenetr <- as_flag(
  arg_value("--allow-installed-nichenetr", "false")
)
verify_official_md5 <- as_flag(arg_value("--verify-official-md5", "true"))
allow_nonformal_shared <- as_flag(arg_value("--allow-nonformal-shared-input", "false"))
min_expression_fraction <- as.numeric(arg_value("--min-expression-fraction", "0.05"))
min_target_genes <- as.integer(arg_value("--min-target-genes", "20"))
min_background_genes <- as.integer(arg_value("--min-background-genes", "500"))
top_ligands <- as.integer(arg_value("--top-ligands", "30"))
top_targets_per_ligand <- as.integer(arg_value("--top-targets-per-ligand", "100"))

if (!mode %in% c("default", "custom")) stop("--mode must be default or custom")
if (is.null(shared_dir) || is.null(prior_dir) || is.null(out_dir)) {
  stop("--shared-dir, --prior-dir, and --out-dir are required")
}
if (!dir.exists(shared_dir)) stop("Shared input directory does not exist: ", shared_dir)
if (!dir.exists(prior_dir)) stop("Prior directory does not exist: ", prior_dir)
if (dir.exists(out_dir) && length(list.files(out_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop("Output directory must be absent or empty: ", out_dir)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

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
  as_tibble(frame)
}
write_csv_table <- function(frame, path) {
  if (csv_backend == "readr") {
    readr::write_csv(frame, path)
  } else {
    utils::write.csv(
      as.data.frame(frame),
      path,
      row.names = FALSE,
      na = ""
    )
  }
}

frozen_source_commit <- "66f90d5eeafef280b2b2f339b3fd70ffec1781dd"
frozen_core_md5 <- c(
  "supporting_functions.R" = "7c74d88d20545d568cea038c35ab393c",
  "evaluate_model_target_prediction.R" = "59dc49100d2a0f9ccb7f74acb1459ae9",
  "evaluate_model_ligand_prediction.R" = "e4b42986954dc2c73ef5adc356e4a6ee",
  "application_prediction.R" = "2cf8b23dc3535c1626fa7d590aab6d86"
)

if (is.null(nichenetr_source)) {
  if (!allow_installed_nichenetr) {
    stop(
      "Formal execution requires --nichenetr-source so commit and core-file hashes ",
      "can be verified. Use --allow-installed-nichenetr true only for an explicitly ",
      "documented non-formal installed-package run."
    )
  }
  if (!requireNamespace("nichenetr", quietly = TRUE)) {
    stop(
      "nichenetr is not installed; provide --nichenetr-source pointing to the ",
      "frozen official checkout"
    )
  }
  package_version <- as.character(packageVersion("nichenetr"))
  if (!allow_version_mismatch && package_version != expected_version) {
    stop(
      "nichenetr version is ", package_version,
      "; expected exactly ", expected_version,
      ". Use the pinned source/package or explicitly pass --allow-version-mismatch true."
    )
  }
  predict_ligand_activities_impl <- getExportedValue(
    "nichenetr", "predict_ligand_activities"
  )
  get_weighted_ligand_target_links_impl <- getExportedValue(
    "nichenetr", "get_weighted_ligand_target_links"
  )
  nichenetr_engine <- list(
    mode = "installed_package",
    version = package_version,
    expected_version = expected_version,
    version_verified = package_version == expected_version,
    installed_package_explicitly_allowed = allow_installed_nichenetr,
    package_path = normalizePath(find.package("nichenetr"))
  )
} else {
  if (!dir.exists(nichenetr_source)) {
    stop("--nichenetr-source does not exist: ", nichenetr_source)
  }
  source_root <- normalizePath(nichenetr_source, mustWork = TRUE)
  description_path <- file.path(source_root, "DESCRIPTION")
  if (!file.exists(description_path)) {
    stop("NicheNet source checkout lacks DESCRIPTION: ", source_root)
  }
  description <- read.dcf(description_path)
  description_md5 <- unname(tools::md5sum(description_path))
  package_version <- unname(description[1, "Version"])
  if (!allow_version_mismatch && package_version != expected_version) {
    stop(
      "NicheNet source version is ", package_version,
      "; expected exactly ", expected_version
    )
  }
  source_commit <- tryCatch(
    system2(
      "git",
      c("-C", shQuote(source_root), "rev-parse", "HEAD"),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(error) character()
  )
  source_status <- attr(source_commit, "status")
  if (
    length(source_commit) != 1 ||
      (!is.null(source_status) && source_status != 0) ||
      tolower(source_commit) != frozen_source_commit
  ) {
    stop(
      "NicheNet source commit mismatch: expected ", frozen_source_commit,
      "; observed ", paste(source_commit, collapse = " ")
    )
  }
  source_dirty <- tryCatch(
    system2(
      "git",
      c(
        "-C", shQuote(source_root), "status", "--porcelain",
        "--untracked-files=no"
      ),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(error) "git-status-error"
  )
  core_paths <- file.path(source_root, "R", names(frozen_core_md5))
  missing_core <- names(frozen_core_md5)[!file.exists(core_paths)]
  if (length(missing_core) > 0) {
    stop("NicheNet source lacks core files: ", paste(missing_core, collapse = ", "))
  }
  observed_core_md5 <- unname(tools::md5sum(core_paths))
  names(observed_core_md5) <- names(frozen_core_md5)
  if (any(tolower(observed_core_md5) != frozen_core_md5)) {
    mismatch <- names(frozen_core_md5)[
      tolower(observed_core_md5) != frozen_core_md5
    ]
    stop("NicheNet core source MD5 mismatch: ", paste(mismatch, collapse = ", "))
  }

  source_dependencies <- c(
    "dplyr", "tibble", "tidyr", "ROCR", "caTools", "data.table"
  )
  missing_dependencies <- source_dependencies[
    !vapply(source_dependencies, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_dependencies) > 0) {
    stop(
      "Pinned NicheNet core source lacks runtime dependencies: ",
      paste(missing_dependencies, collapse = ", ")
    )
  }
  source_environment <- new.env(parent = globalenv())
  for (package in c("dplyr", "tibble", "tidyr")) {
    for (symbol in getNamespaceExports(package)) {
      assign(
        symbol,
        getExportedValue(package, symbol),
        envir = source_environment
      )
    }
  }
  source_environment$prediction <- ROCR::prediction
  source_environment$performance <- ROCR::performance
  source_environment$trapz <- caTools::trapz
  source_environment$data.table <- data.table::data.table
  for (path in core_paths) sys.source(path, envir = source_environment)
  required_core_functions <- c(
    "predict_ligand_activities", "get_weighted_ligand_target_links"
  )
  missing_functions <- required_core_functions[
    !vapply(
      required_core_functions,
      exists,
      logical(1),
      envir = source_environment,
      mode = "function",
      inherits = FALSE
    )
  ]
  if (length(missing_functions) > 0) {
    stop(
      "Pinned NicheNet source did not define: ",
      paste(missing_functions, collapse = ", ")
    )
  }
  predict_ligand_activities_impl <- source_environment$predict_ligand_activities
  get_weighted_ligand_target_links_impl <-
    source_environment$get_weighted_ligand_target_links
  nichenetr_engine <- list(
    mode = "pinned_core_source",
    version = package_version,
    expected_version = expected_version,
    version_verified = package_version == expected_version,
    source_root = source_root,
    git_commit = unname(source_commit),
    expected_git_commit = frozen_source_commit,
    commit_verified = TRUE,
    tracked_worktree_dirty = length(source_dirty) > 0,
    tracked_worktree_status = as.list(unname(source_dirty)),
    description_md5 = description_md5,
    core_files = as.list(setNames(core_paths, names(frozen_core_md5))),
    expected_core_md5 = as.list(frozen_core_md5),
    observed_core_md5 = as.list(observed_core_md5),
    core_md5_verified = TRUE,
    runtime_dependencies = as.list(setNames(
      vapply(
        source_dependencies,
        function(package) as.character(packageVersion(package)),
        character(1)
      ),
      source_dependencies
    ))
  )
}

ltm_path <- file.path(prior_dir, "ligand_target_matrix_nsga2r_final_mouse.rds")
lr_path <- file.path(prior_dir, "lr_network_mouse_21122021.rds")
if (!file.exists(ltm_path)) stop("Missing official ligand-target matrix: ", ltm_path)
if (!file.exists(lr_path)) stop("Missing official LR network: ", lr_path)

expected_md5 <- c(
  ligand_target_matrix_nsga2r_final_mouse.rds = "ac80d846fe0bfc4879a5b52ca85ffeb9",
  lr_network_mouse_21122021.rds = "cf33ee8b6bf84bdf2d11cab9c8f94b9e"
)
observed_md5 <- c(
  ligand_target_matrix_nsga2r_final_mouse.rds = unname(tools::md5sum(ltm_path)),
  lr_network_mouse_21122021.rds = unname(tools::md5sum(lr_path))
)
if (verify_official_md5 && any(observed_md5 != expected_md5)) {
  stop(
    "Official NicheNet-v2 asset MD5 mismatch. expected=",
    paste(names(expected_md5), expected_md5, collapse = ";"),
    " observed=", paste(names(observed_md5), observed_md5, collapse = ";")
  )
}

ligand_target_matrix <- readRDS(ltm_path)
lr_network <- readRDS(lr_path)
if (is.null(rownames(ligand_target_matrix)) || is.null(colnames(ligand_target_matrix))) {
  stop("The ligand-target matrix must have target genes in rows and ligands in columns")
}
if (!all(c("from", "to") %in% colnames(lr_network))) {
  stop("The official LR network must contain from/to columns")
}
default_lr <- lr_network %>%
  transmute(ligand = as.character(from), receptor = as.character(to)) %>%
  distinct()

units_path <- file.path(shared_dir, "units_manifest.csv")
expression_path <- file.path(shared_dir, "expression_by_stage_celltype.csv.gz")
prepare_manifest_path <- file.path(shared_dir, "prepare_manifest.json")
for (path in c(units_path, expression_path, prepare_manifest_path)) {
  if (!file.exists(path)) stop("Missing shared input: ", path)
}

prepare_manifest <- fromJSON(prepare_manifest_path, simplifyVector = TRUE)
if (!identical(as.character(prepare_manifest$status), "complete")) {
  stop("Shared-input preparation manifest is not complete")
}
if (!allow_nonformal_shared) {
  if (!isTRUE(prepare_manifest$formal_mode)) {
    stop("Formal run requires a shared-input manifest with formal_mode=true")
  }
  if (
    length(prepare_manifest$normalization$frozen_target_sum) != 1 ||
      as.numeric(prepare_manifest$normalization$frozen_target_sum) != 1105
  ) {
    stop("Formal run requires frozen_target_sum exactly 1105")
  }
  if (!isTRUE(prepare_manifest$normalization$x_validation$passed)) {
    stop("Formal run requires successful X reconstruction validation")
  }
  orthology_release <- suppressWarnings(
    as.integer(prepare_manifest$orthology_source$ensembl_release)
  )
  if (
    !isTRUE(prepare_manifest$orthology_source$verified) ||
      !isTRUE(prepare_manifest$orthology_source$strict_map_md5_verified) ||
      length(orthology_release) != 1 || is.na(orthology_release) || orthology_release != 116
  ) {
    stop("Formal run requires a verified Ensembl release 116 strict orthology manifest")
  }
  frozen_input_sha256 <- c(
    h5ad = "433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd",
    custom_lr_db = "27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37"
  )
  valid_sha256 <- function(value) {
    length(value) == 1 && !is.na(value) && grepl("^[0-9a-fA-F]{64}$", value)
  }
  h5ad_sha256 <- as.character(prepare_manifest$inputs$h5ad$sha256)
  custom_lr_sha256 <- as.character(prepare_manifest$inputs$custom_lr_db$sha256)
  if (!valid_sha256(h5ad_sha256) || !valid_sha256(custom_lr_sha256)) {
    stop("Formal shared-input manifest lacks valid scalar H5AD/custom-LR SHA256 values")
  }
  observed_input_sha256 <- c(
    h5ad = h5ad_sha256,
    custom_lr_db = custom_lr_sha256
  )
  if (any(tolower(observed_input_sha256) != frozen_input_sha256)) {
    stop("Formal shared-input manifest does not reference the frozen H5AD/custom-LR hashes")
  }
}

shared_file_records <- as_tibble(prepare_manifest$output_files)
required_integrity_columns <- c("path", "size_bytes", "sha256", "md5")
if (!all(required_integrity_columns %in% colnames(shared_file_records))) {
  stop(
    "Shared-input manifest output_files lacks: ",
    paste(setdiff(required_integrity_columns, colnames(shared_file_records)), collapse = ", ")
  )
}
required_shared_files <- c(
  "units_manifest.csv",
  "expression_by_stage_celltype.csv.gz",
  "custom_lr_strict_one2one_mapped.csv"
)
if (!all(required_shared_files %in% shared_file_records$path)) {
  stop(
    "Shared-input manifest does not inventory: ",
    paste(setdiff(required_shared_files, shared_file_records$path), collapse = ", ")
  )
}
if (any(grepl("^(/|[A-Za-z]:)", shared_file_records$path))) {
  stop("Shared-input manifest output paths must be relative")
}
shared_root <- normalizePath(shared_dir, mustWork = TRUE)
shared_file_paths <- normalizePath(
  file.path(shared_root, shared_file_records$path),
  mustWork = TRUE
)
shared_prefix <- paste0(shared_root, .Platform$file.sep)
if (any(!startsWith(shared_file_paths, shared_prefix))) {
  stop("Shared-input manifest contains a path outside shared-dir")
}
observed_shared_md5 <- unname(tools::md5sum(shared_file_paths))
if (any(tolower(observed_shared_md5) != tolower(shared_file_records$md5))) {
  mismatch <- shared_file_records$path[
    tolower(observed_shared_md5) != tolower(shared_file_records$md5)
  ]
  stop("Shared-input file MD5 mismatch: ", paste(mismatch, collapse = ", "))
}
observed_shared_size <- as.numeric(file.info(shared_file_paths)$size)
if (any(observed_shared_size != as.numeric(shared_file_records$size_bytes))) {
  stop("Shared-input file size mismatch against prepare_manifest.json")
}
shared_file_integrity <- list(
  verified = TRUE,
  algorithm = "MD5 plus file size; SHA256 remains recorded in the preparation manifest",
  files_checked = nrow(shared_file_records)
)

units <- read_csv_table(units_path)
expression <- read_csv_table(expression_path)
units$source_stage_id <- as.character(units$source_stage_id)
units$target_stage_id <- as.character(units$target_stage_id)
expression$stage_id <- as.character(expression$stage_id)
expression$cell_type <- as.character(expression$cell_type)
expression$gene_mouse <- as.character(expression$gene_mouse)
required_expression <- c(
  "stage_id", "stage_label", "cell_type", "n_cells", "gene_mouse",
  "pct_detected", "mean_normalized_linear", "mean_log1p"
)
if (!all(required_expression %in% colnames(expression))) {
  stop("Expression summary lacks: ", paste(setdiff(required_expression, colnames(expression)), collapse = ", "))
}

if (mode == "custom") {
  if (is.null(custom_lr_path) || !file.exists(custom_lr_path)) {
    stop("Custom mode requires --custom-lr or the mapped custom LR table in shared-dir")
  }
  if (
    !allow_nonformal_shared &&
      normalizePath(custom_lr_path, mustWork = TRUE) !=
        normalizePath(file.path(shared_dir, "custom_lr_strict_one2one_mapped.csv"), mustWork = TRUE)
  ) {
    stop("Formal custom mode must use the integrity-checked mapped LR table from shared-dir")
  }
  custom_lr <- read_csv_table(custom_lr_path)
  required_custom <- c(
    "ligand_mouse", "receptor_mouse_components", "ligand_zebrafish",
    "receptor_zebrafish", "pathways", "categories"
  )
  if (!all(required_custom %in% colnames(custom_lr))) {
    stop("Mapped custom LR table lacks: ", paste(setdiff(required_custom, colnames(custom_lr)), collapse = ", "))
  }
}

component_values <- function(receiver_expression, component_string) {
  components <- unique(strsplit(as.character(component_string), ";", fixed = TRUE)[[1]])
  components <- components[nzchar(components)]
  matched <- receiver_expression %>% filter(gene_mouse %in% components)
  present <- length(components) > 0 && n_distinct(matched$gene_mouse) == length(components)
  gate <- present && all(matched$pct_detected >= min_expression_fraction)
  list(
    components = paste(components, collapse = ";"),
    n_components = length(components),
    all_components_present = present,
    receptor_gate_pass = gate,
    receptor_pct_detected_min = if (present) min(matched$pct_detected) else NA_real_,
    receptor_mean_normalized_linear_min = if (present) min(matched$mean_normalized_linear) else NA_real_,
    receptor_mean_log1p_min = if (present) min(matched$mean_log1p) else NA_real_
  )
}

empty_activity <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  source_stage_label = character(), target_stage_label = character(), receiver = character(),
  mode = character(), test_ligand = character(), auroc = double(), aupr = double(),
  aupr_corrected = double(), pearson = double(), rank = double()
)
empty_sender <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  receiver = character(), mode = character(), sender = character(), ligand = character(),
  sender_ligand_pct_detected = double(), sender_ligand_mean_normalized_linear = double(),
  aupr_corrected = double(), ligand_activity_rank = double(), activity_scope = character()
)
empty_lr <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  receiver = character(), mode = character(), sender = character(), ligand = character(),
  receptor = character(), pathways = character(), categories = character(),
  aupr_corrected = double(), ligand_activity_rank = double(), activity_scope = character()
)
empty_targets <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  receiver = character(), mode = character(), ligand = character(), target = character(),
  ligand_target_score = double()
)
empty_target_errors <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  receiver = character(), mode = character(), ligand = character(), error = character()
)
empty_coverage <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  source_stage_label = character(), target_stage_label = character(), receiver = character(),
  mode = character(), input_status = character(), n_response_genes_input = integer(),
  n_response_genes_in_prior_background = integer(), n_background_genes_input = integer(),
  n_background_genes_in_prior = integer(), n_source_celltypes = integer(),
  n_candidate_lr_rows = integer(), n_potential_ligands = integer(),
  n_ligand_activities = integer(), status = character()
)
empty_unit_status <- tibble(
  unit_id = character(), source_stage_id = character(), target_stage_id = character(),
  source_stage_label = character(), target_stage_label = character(), receiver = character(),
  mode = character(), input_status = character(), status = character(), detail = character()
)

activity_tables <- list()
sender_tables <- list()
lr_tables <- list()
target_tables <- list()
target_link_error_rows <- list()
coverage_rows <- list()
unit_status_rows <- list()

for (unit_index in seq_len(nrow(units))) {
  unit <- units[unit_index, ]
  unit_id <- as.character(unit$unit_id[[1]])
  input_status <- as.character(unit$status[[1]])
  base_status <- list(
    unit_id = unit_id,
    source_stage_id = as.character(unit$source_stage_id[[1]]),
    target_stage_id = as.character(unit$target_stage_id[[1]]),
    source_stage_label = as.character(unit$source_stage_label[[1]]),
    target_stage_label = as.character(unit$target_stage_label[[1]]),
    receiver = as.character(unit$receiver[[1]]),
    mode = mode,
    input_status = input_status
  )
  if (input_status != "eligible") {
    unit_status_rows[[length(unit_status_rows) + 1]] <- as_tibble(c(base_status, list(status = "skipped_shared_input_ineligible", detail = input_status)))
    next
  }

  unit_dir <- file.path(shared_dir, as.character(unit$unit_dir[[1]]))
  geneset_path <- file.path(unit_dir, "receiver_response_genes.csv")
  background_path <- file.path(unit_dir, "receiver_background_genes.csv")
  if (!file.exists(geneset_path) || !file.exists(background_path)) {
    unit_status_rows[[length(unit_status_rows) + 1]] <- as_tibble(c(base_status, list(status = "error", detail = "missing unit gene files")))
    next
  }
  geneset_input <- read_csv_table(geneset_path)$gene_mouse %>% unique()
  background_input <- read_csv_table(background_path)$gene_mouse %>% unique()
  geneset <- intersect(geneset_input, rownames(ligand_target_matrix))
  background <- intersect(background_input, rownames(ligand_target_matrix))
  geneset <- intersect(geneset, background)

  source_stage <- as.character(unit$source_stage_id[[1]])
  receiver <- as.character(unit$receiver[[1]])
  source_expression <- expression %>% filter(stage_id == source_stage)
  receiver_expression <- source_expression %>% filter(cell_type == receiver)
  sender_expressed <- source_expression %>%
    filter(pct_detected >= min_expression_fraction)
  receiver_expressed_genes <- receiver_expression %>%
    filter(pct_detected >= min_expression_fraction) %>%
    pull(gene_mouse) %>% unique()

  if (mode == "default") {
    candidate_network <- default_lr %>%
      filter(
        ligand %in% colnames(ligand_target_matrix),
        ligand %in% sender_expressed$gene_mouse,
        receptor %in% receiver_expressed_genes
      ) %>%
      mutate(
        receptor_mouse_components = receptor,
        ligand_zebrafish = NA_character_,
        receptor_zebrafish = NA_character_,
        pathways = NA_character_,
        categories = NA_character_,
        receptor_gate_pass = TRUE
      )
  } else {
    if (nrow(custom_lr) == 0) {
      gated_custom <- custom_lr %>%
        mutate(
          components = character(),
          n_components = integer(),
          all_components_present = logical(),
          receptor_gate_pass = logical(),
          receptor_pct_detected_min = double(),
          receptor_mean_normalized_linear_min = double(),
          receptor_mean_log1p_min = double()
        )
    } else {
      gate_rows <- lapply(seq_len(nrow(custom_lr)), function(index) {
        values <- component_values(receiver_expression, custom_lr$receptor_mouse_components[[index]])
        bind_cols(custom_lr[index, ], as_tibble(values))
      })
      gated_custom <- bind_rows(gate_rows)
    }
    candidate_network <- gated_custom %>%
      rename(ligand = ligand_mouse) %>%
      mutate(receptor = receptor_mouse_components) %>%
      filter(
        receptor_gate_pass,
        ligand %in% colnames(ligand_target_matrix),
        ligand %in% sender_expressed$gene_mouse
      )
  }
  potential_ligands <- intersect(unique(candidate_network$ligand), colnames(ligand_target_matrix))

  coverage <- as_tibble(c(
    base_status,
    list(
      n_response_genes_input = length(geneset_input),
      n_response_genes_in_prior_background = length(geneset),
      n_background_genes_input = length(background_input),
      n_background_genes_in_prior = length(background),
      n_source_celltypes = n_distinct(source_expression$cell_type),
      n_candidate_lr_rows = nrow(candidate_network),
      n_potential_ligands = length(potential_ligands)
    )
  ))

  eligibility_reason <- NULL
  if (length(geneset) < min_target_genes) eligibility_reason <- "too_few_response_genes_after_prior_intersection"
  if (length(background) < min_background_genes) eligibility_reason <- "too_few_background_genes_after_prior_intersection"
  if (length(potential_ligands) < 2) eligibility_reason <- "too_few_potential_ligands"
  if (!is.null(eligibility_reason)) {
    coverage$n_ligand_activities <- 0L
    coverage$status <- eligibility_reason
    coverage_rows[[length(coverage_rows) + 1]] <- coverage
    unit_status_rows[[length(unit_status_rows) + 1]] <- as_tibble(c(base_status, list(status = "skipped_nichenet_ineligible", detail = eligibility_reason)))
    next
  }

  activity_result <- tryCatch(
    predict_ligand_activities_impl(
      geneset = geneset,
      background_expressed_genes = background,
      ligand_target_matrix = ligand_target_matrix,
      potential_ligands = potential_ligands
    ),
    error = function(error) error
  )
  if (inherits(activity_result, "error")) {
    coverage$n_ligand_activities <- 0L
    coverage$status <- "nichenetr_error"
    coverage_rows[[length(coverage_rows) + 1]] <- coverage
    unit_status_rows[[length(unit_status_rows) + 1]] <- as_tibble(c(base_status, list(status = "error", detail = conditionMessage(activity_result))))
    next
  }

  activity <- activity_result %>%
    arrange(desc(aupr_corrected), desc(aupr), desc(auroc)) %>%
    mutate(
      rank = rank(-aupr_corrected, ties.method = "min"),
      unit_id = unit_id,
      source_stage_id = source_stage,
      target_stage_id = as.character(unit$target_stage_id[[1]]),
      source_stage_label = as.character(unit$source_stage_label[[1]]),
      target_stage_label = as.character(unit$target_stage_label[[1]]),
      receiver = receiver,
      mode = mode,
      .before = 1
    )
  activity_tables[[length(activity_tables) + 1]] <- activity

  sender_activity <- sender_expressed %>%
    filter(gene_mouse %in% activity$test_ligand) %>%
    transmute(
      sender = cell_type,
      ligand = gene_mouse,
      sender_ligand_pct_detected = pct_detected,
      sender_ligand_mean_normalized_linear = mean_normalized_linear,
      sender_ligand_mean_log1p = mean_log1p
    ) %>%
    inner_join(
      activity %>% select(test_ligand, aupr_corrected, rank),
      by = c("ligand" = "test_ligand")
    ) %>%
    rename(ligand_activity_rank = rank) %>%
    mutate(
      unit_id = unit_id,
      source_stage_id = source_stage,
      target_stage_id = as.character(unit$target_stage_id[[1]]),
      receiver = receiver,
      mode = mode,
      activity_scope = "transition_receiver_ligand_not_sender_specific",
      .before = 1
    )
  sender_tables[[length(sender_tables) + 1]] <- sender_activity

  if (mode == "default") {
    receiver_lr <- receiver_expression %>%
      transmute(
        receptor = gene_mouse,
        receptor_pct_detected_min = pct_detected,
        receptor_mean_normalized_linear_min = mean_normalized_linear,
        receptor_mean_log1p_min = mean_log1p
      )
    network_for_join <- candidate_network %>% inner_join(receiver_lr, by = "receptor")
  } else {
    network_for_join <- candidate_network
  }
  lr_activity <- network_for_join %>%
    inner_join(
      sender_expressed %>%
        transmute(
          sender = cell_type,
          ligand = gene_mouse,
          sender_ligand_pct_detected = pct_detected,
          sender_ligand_mean_normalized_linear = mean_normalized_linear,
          sender_ligand_mean_log1p = mean_log1p
        ),
      by = "ligand"
    ) %>%
    inner_join(
      activity %>% select(test_ligand, aupr_corrected, rank),
      by = c("ligand" = "test_ligand")
    ) %>%
    rename(ligand_activity_rank = rank) %>%
    mutate(
      unit_id = unit_id,
      source_stage_id = source_stage,
      target_stage_id = as.character(unit$target_stage_id[[1]]),
      receiver = receiver,
      mode = mode,
      activity_scope = "transition_receiver_ligand_not_sender_or_receptor_specific",
      .before = 1
    )
  lr_tables[[length(lr_tables) + 1]] <- lr_activity

  best_ligands <- head(activity$test_ligand, top_ligands)
  target_results <- lapply(best_ligands, function(ligand) {
    tryCatch(
      list(
        ligand = ligand,
        data = get_weighted_ligand_target_links_impl(
          ligand = ligand,
          geneset = geneset,
          ligand_target_matrix = ligand_target_matrix,
          n = top_targets_per_ligand
        ),
        error = NULL
      ),
      error = function(error) list(
        ligand = ligand,
        data = tibble(),
        error = conditionMessage(error)
      )
    )
  })
  unit_target_errors <- bind_rows(lapply(target_results, function(result) {
    if (is.null(result$error)) return(tibble())
    tibble(
      unit_id = unit_id,
      source_stage_id = source_stage,
      target_stage_id = as.character(unit$target_stage_id[[1]]),
      receiver = receiver,
      mode = mode,
      ligand = result$ligand,
      error = result$error
    )
  }))
  if (nrow(unit_target_errors) > 0) {
    target_link_error_rows[[length(target_link_error_rows) + 1]] <- unit_target_errors
  }
  target_links <- bind_rows(lapply(target_results, function(result) {
    if (is.null(result$data) || nrow(result$data) == 0) return(tibble())
    result$data
  }))
  if (nrow(target_links) > 0) {
    target_links <- target_links %>%
      rename(ligand_target_score = weight) %>%
      mutate(
        unit_id = unit_id,
        source_stage_id = source_stage,
        target_stage_id = as.character(unit$target_stage_id[[1]]),
        receiver = receiver,
        mode = mode,
        .before = 1
      )
    target_tables[[length(target_tables) + 1]] <- target_links
  }

  coverage$n_ligand_activities <- nrow(activity)
  coverage$status <- if (nrow(unit_target_errors) > 0) "target_link_error" else "complete"
  coverage_rows[[length(coverage_rows) + 1]] <- coverage
  if (nrow(unit_target_errors) > 0) {
    unit_status_rows[[length(unit_status_rows) + 1]] <- as_tibble(c(
      base_status,
      list(
        status = "error",
        detail = paste0(nrow(unit_target_errors), " ligand-target link error(s); see target_link_errors.csv")
      )
    ))
  } else {
    unit_status_rows[[length(unit_status_rows) + 1]] <- as_tibble(c(base_status, list(status = "complete", detail = "")))
  }
}

ligand_activity_all <- if (length(activity_tables)) bind_rows(activity_tables) else empty_activity
sender_activity_all <- if (length(sender_tables)) bind_rows(sender_tables) else empty_sender
lr_activity_all <- if (length(lr_tables)) bind_rows(lr_tables) else empty_lr
target_links_all <- if (length(target_tables)) bind_rows(target_tables) else empty_targets
target_link_errors_all <- if (length(target_link_error_rows)) bind_rows(target_link_error_rows) else empty_target_errors
coverage_all <- if (length(coverage_rows)) bind_rows(coverage_rows) else empty_coverage
unit_status_all <- if (length(unit_status_rows)) bind_rows(unit_status_rows) else empty_unit_status

units_complete <- sum(unit_status_all$status == "complete", na.rm = TRUE)
units_error <- sum(unit_status_all$status == "error", na.rm = TRUE)
run_status <- if (units_error > 0 && units_complete > 0) {
  "partial_failure"
} else if (units_error > 0) {
  "failed"
} else if (units_complete == 0) {
  "no_eligible_units"
} else {
  "complete"
}

output_paths <- c(
  ligand_activity = file.path(out_dir, "ligand_activity.csv"),
  sender_ligand_activity = file.path(out_dir, "sender_ligand_activity.csv"),
  lr_activity = file.path(out_dir, "lr_activity.csv"),
  ligand_target_links = file.path(out_dir, "ligand_target_links.csv"),
  target_link_errors = file.path(out_dir, "target_link_errors.csv"),
  coverage = file.path(out_dir, "coverage.csv"),
  unit_status = file.path(out_dir, "unit_status.csv")
)
write_csv_table(ligand_activity_all, output_paths[["ligand_activity"]])
write_csv_table(sender_activity_all, output_paths[["sender_ligand_activity"]])
write_csv_table(lr_activity_all, output_paths[["lr_activity"]])
write_csv_table(target_links_all, output_paths[["ligand_target_links"]])
write_csv_table(target_link_errors_all, output_paths[["target_link_errors"]])
write_csv_table(coverage_all, output_paths[["coverage"]])
write_csv_table(unit_status_all, output_paths[["unit_status"]])

output_md5 <- unname(tools::md5sum(output_paths))
names(output_md5) <- names(output_paths)

manifest <- list(
  schema_version = 1,
  workflow = "reviewer_zebrafish_cross_species_nichenet_v2",
  status = run_status,
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  mode = mode,
  method_label = if (mode == "default") {
    "cross-species mapped NicheNet-v2 (official mouse LR)"
  } else {
    "custom-LR-constrained cross-species mapped NicheNet-v2"
  },
  shared_dir = normalizePath(shared_dir),
  shared_prepare_manifest = list(
    path = normalizePath(prepare_manifest_path),
    md5 = unname(tools::md5sum(prepare_manifest_path)),
    formal_mode_required = !allow_nonformal_shared,
    shared_file_integrity = shared_file_integrity
  ),
  official_prior = list(
    ligand_target_matrix = normalizePath(ltm_path),
    lr_network = normalizePath(lr_path),
    expected_md5 = as.list(expected_md5),
    observed_md5 = as.list(observed_md5),
    md5_verified = verify_official_md5
  ),
  custom_lr = if (mode == "custom") list(
    path = normalizePath(custom_lr_path),
    md5 = unname(tools::md5sum(custom_lr_path)),
    role = "candidate ligand/receptor gate only; ligand-target/signaling/GRN prior remains fixed"
  ) else NULL,
  parameters = list(
    min_expression_fraction = min_expression_fraction,
    min_target_genes = min_target_genes,
    min_background_genes = min_background_genes,
    top_ligands = top_ligands,
    top_targets_per_ligand = top_targets_per_ligand
  ),
  counts = list(
    units_total = nrow(units),
    units_complete = units_complete,
    units_error = units_error,
    ligand_activity_rows = nrow(ligand_activity_all),
    sender_ligand_activity_rows = nrow(sender_activity_all),
    lr_activity_rows = nrow(lr_activity_all),
    ligand_target_link_rows = nrow(target_links_all),
    target_link_error_rows = nrow(target_link_errors_all)
  ),
  output_md5 = as.list(output_md5),
  output_files = as.list(output_paths),
  software = list(
    R = R.version.string,
    csv_backend = csv_backend,
    nichenetr = nichenetr_engine
  ),
  activity_semantics = paste(
    "aupr_corrected is native NicheNet ligand activity for a transition/receiver/ligand.",
    "Sender and receptor columns are expression/LR candidate assignments and do not make",
    "the activity a direct sender-specific, receptor-specific, spatial, or biochemical strength."
  ),
  cross_species_caveat = paste(
    "NicheNet-v2 has human/mouse priors, not a native zebrafish prior.",
    "This run uses frozen high-confidence strict one-to-one zebrafish-to-mouse mappings."
  )
)
write_json(manifest, file.path(out_dir, "run_manifest.json"), pretty = TRUE, auto_unbox = TRUE, null = "null")
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
print(manifest$counts)
if (run_status != "complete") {
  stop("NicheNet run ended with status: ", run_status, call. = FALSE)
}
