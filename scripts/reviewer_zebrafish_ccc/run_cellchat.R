#!/usr/bin/env Rscript

# Run official CellChat once across every prepared observed zebrafish stage.
#
# The flat CSV is not merely cited in the manifest.  Its database_row values
# are resolved against CellChatDB.zebrafish and expanded ligand/receptor,
# pathway, and category values are checked row by row before computation.

parse_args <- function(values) {
  result <- list()
  i <- 1L
  while (i <= length(values)) {
    key <- values[[i]]
    if (!startsWith(key, "--") || i == length(values)) {
      stop("Arguments must be supplied as --name value pairs.")
    }
    result[[substring(key, 3L)]] <- values[[i + 1L]]
    i <- i + 2L
  }
  result
}

required_path <- function(args, name) {
  value <- args[[name]]
  if (is.null(value) || !nzchar(value)) stop(paste0("Missing --", name))
  normalizePath(value, mustWork = TRUE)
}

as_bool <- function(value, default = FALSE) {
  if (is.null(value)) return(default)
  normalized <- tolower(value)
  if (normalized %in% c("true", "1", "yes")) return(TRUE)
  if (normalized %in% c("false", "0", "no")) return(FALSE)
  stop(paste("Expected a boolean but received", value))
}

sha256_file <- function(path) {
  executable <- Sys.which("sha256sum")
  args <- c(shQuote(normalizePath(path, mustWork = TRUE)))
  if (!nzchar(executable)) {
    executable <- Sys.which("shasum")
    args <- c("-a", "256", args)
  }
  if (!nzchar(executable)) stop("sha256sum or shasum is required for manifests")
  output <- system2(executable, args, stdout = TRUE, stderr = TRUE)
  if (length(output) < 1L || !grepl("^[0-9a-fA-F]{64}[[:space:]]", output[[1L]])) {
    stop(paste("Could not calculate SHA256 for", path))
  }
  tolower(sub("[[:space:]].*$", "", output[[1L]]))
}

file_record <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  list(path = path, bytes = unname(file.info(path)$size), sha256 = sha256_file(path))
}

expand_complex <- function(tokens, complex_table) {
  vapply(as.character(tokens), function(token) {
    if (token %in% rownames(complex_table)) {
      columns <- grep("^subunit", colnames(complex_table))
      genes <- as.character(unlist(complex_table[token, columns, drop = TRUE]))
      genes <- genes[!is.na(genes) & nzchar(genes)]
      paste(genes, collapse = "_")
    } else {
      token
    }
  }, character(1L))
}

write_csv_gz <- function(frame, path) {
  connection <- gzfile(path, open = "wt")
  on.exit(close(connection), add = TRUE)
  write.csv(frame, connection, row.names = FALSE, quote = TRUE, na = "")
}

common_columns <- c(
  "method", "database_variant", "stage", "stage_time", "sender_type",
  "receiver_type", "ligand", "receptor", "pathway", "category",
  "interaction_id", "score", "p_value", "significant", "n_sender_cells",
  "n_receiver_cells", "score_semantics"
)

empty_common <- function() {
  result <- as.data.frame(setNames(replicate(length(common_columns), logical(0), simplify = FALSE), common_columns))
  for (column in setdiff(common_columns, c("score", "p_value", "n_sender_cells", "n_receiver_cells", "significant", "stage_time"))) {
    result[[column]] <- character(0)
  }
  result$stage_time <- numeric(0)
  result$score <- numeric(0)
  result$p_value <- numeric(0)
  result$significant <- logical(0)
  result$n_sender_cells <- integer(0)
  result$n_receiver_cells <- integer(0)
  result$abundance_controlled_score <- numeric(0)
  result
}

bind_nonempty <- function(frames, extra_columns = character(0)) {
  retained <- frames[vapply(frames, nrow, integer(1L)) > 0L]
  if (length(retained) > 0L) return(do.call(rbind, retained))
  result <- empty_common()
  for (column in extra_columns) result[[column]] <- logical(0)
  result
}

matrix_context <- function(
    scores, pvalues, stage, stage_time, ligand, receptor, pathway, category,
    interaction_id, cell_counts, score_semantics, positive_only = TRUE) {
  if (is.null(dim(scores)) || length(dim(scores)) != 2L) stop("Expected a score matrix")
  indices <- if (positive_only) which(scores > 0, arr.ind = TRUE) else which(matrix(TRUE, nrow(scores), ncol(scores)), arr.ind = TRUE)
  if (nrow(indices) == 0L) return(empty_common())
  sender <- rownames(scores)[indices[, 1L]]
  receiver <- colnames(scores)[indices[, 2L]]
  probability <- if (is.null(pvalues)) rep(NA_real_, nrow(indices)) else pvalues[indices]
  result <- data.frame(
    method = "CellChat",
    database_variant = "current_zebrafish_lr_database",
    stage = stage,
    stage_time = as.numeric(stage_time),
    sender_type = sender,
    receiver_type = receiver,
    ligand = ligand,
    receptor = receptor,
    pathway = pathway,
    category = category,
    interaction_id = interaction_id,
    score = as.numeric(scores[indices]),
    p_value = as.numeric(probability),
    significant = if (is.null(pvalues)) NA else as.numeric(probability) < 0.05,
    n_sender_cells = as.integer(cell_counts[sender]),
    n_receiver_cells = as.integer(cell_counts[receiver]),
    score_semantics = score_semantics,
    abundance_controlled_score = as.numeric(scores[indices]),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  result[, c(common_columns, "abundance_controlled_score"), drop = FALSE]
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
input_dir <- required_path(args, "input-dir")
if (is.null(args[["out-dir"]])) stop("Missing --out-dir")
output_dir <- normalizePath(args[["out-dir"]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0L) {
  stop(paste(output_dir, "is not empty"))
}
nboot <- if (is.null(args[["nboot"]])) 100L else as.integer(args[["nboot"]])
seed_use <- if (is.null(args[["seed"]])) 20260722L else as.integer(args[["seed"]])
mean_method <- if (is.null(args[["mean-method"]])) "triMean" else args[["mean-method"]]
population_size <- as_bool(args[["population-size"]], FALSE)
positive_only <- as_bool(args[["positive-only"]], TRUE)
save_rds <- as_bool(args[["save-rds"]], TRUE)
if (is.na(nboot) || nboot < 1L) stop("--nboot must be positive")
if (is.na(seed_use)) stop("--seed must be an integer")

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
})

cellchat_source <- args[["cellchat-source"]]
if (!is.null(cellchat_source)) {
  cellchat_source <- normalizePath(cellchat_source, mustWork = TRUE)
  description_path <- file.path(cellchat_source, "DESCRIPTION")
  database_rda_path <- file.path(cellchat_source, "data", "CellChatDB.zebrafish.rda")
  core_files <- file.path(
    cellchat_source,
    "R",
    c("CellChat_class.R", "utilities.R", "database.R", "modeling.R")
  )
  required_source_files <- c(description_path, database_rda_path, core_files)
  if (!all(file.exists(required_source_files))) {
    stop("--cellchat-source is missing DESCRIPTION, zebrafish database, or core R files")
  }
  suppressPackageStartupMessages({
    library(dplyr)
    library(igraph)
    library(ggplot2)
    library(magrittr)
    library(future)
    library(future.apply)
    library(pbapply)
  })
  for (source_file in core_files) sys.source(source_file, envir = environment())
  load(database_rda_path, envir = environment())
  description <- read.dcf(description_path)
  cellchat_version <- unname(description[1L, "Version"])
  git_executable <- Sys.which("git")
  cellchat_commit <- if (nzchar(git_executable) && dir.exists(file.path(cellchat_source, ".git"))) {
    output <- system2(
      git_executable,
      c("-C", shQuote(cellchat_source), "rev-parse", "HEAD"),
      stdout = TRUE,
      stderr = FALSE
    )
    if (length(output)) output[[1L]] else NA_character_
  } else {
    NA_character_
  }
  cellchat_load_mode <- "pinned official core R source"
} else {
  suppressPackageStartupMessages(library(CellChat))
  data("CellChatDB.zebrafish", package = "CellChat", envir = environment())
  database_rda_path <- NA_character_
  cellchat_version <- as.character(packageVersion("CellChat"))
  cellchat_commit <- NA_character_
  cellchat_load_mode <- "installed R package"
}

input_manifest_path <- file.path(input_dir, "input_manifest.json")
database_path <- file.path(input_dir, "filtered_lr_database.csv")
input_manifest <- jsonlite::fromJSON(input_manifest_path, simplifyVector = FALSE)
flat_database <- read.csv(database_path, check.names = FALSE, stringsAsFactors = FALSE)
required_database_columns <- c(
  "database_row", "ligand", "receptor", "pathway", "category", "interaction_id"
)
if (!all(required_database_columns %in% colnames(flat_database))) {
  stop("Prepared LR database is missing required columns")
}
if (anyDuplicated(flat_database$database_row)) stop("database_row values must be unique")

if (!exists("CellChatDB.zebrafish")) {
  stop("The installed official CellChat package does not provide CellChatDB.zebrafish")
}
official_database <- CellChatDB.zebrafish
indices <- as.integer(flat_database$database_row) + 1L
if (any(is.na(indices)) || any(indices < 1L) || any(indices > nrow(official_database$interaction))) {
  stop("Prepared database_row is outside the installed CellChatDB.zebrafish interaction table")
}
pair_lr <- official_database$interaction[indices, , drop = FALSE]
ligand_expanded <- tolower(expand_complex(pair_lr$ligand, official_database$complex))
receptor_expanded <- tolower(expand_complex(pair_lr$receptor, official_database$complex))
database_matches <- (
  ligand_expanded == tolower(flat_database$ligand) &
  receptor_expanded == tolower(flat_database$receptor) &
  as.character(pair_lr$pathway_name) == flat_database$pathway &
  as.character(pair_lr$annotation) == flat_database$category
)
if (!all(database_matches)) {
  first_bad <- which(!database_matches)[[1L]]
  stop(paste(
    "Installed CellChatDB.zebrafish does not reproduce prepared CSV at database_row",
    flat_database$database_row[[first_bad]]
  ))
}
pair_lr$database_row <- as.integer(flat_database$database_row)
pair_lr$flat_interaction_id <- flat_database$interaction_id
database_use <- official_database
database_use$interaction <- pair_lr

lr_frames <- list()
pathway_frames <- list()
total_frames <- list()
diagnostic_frames <- list()
artifact_paths <- character(0)

for (stage_position in seq_along(input_manifest$stages)) {
  stage_record <- input_manifest$stages[[stage_position]]
  token <- as.character(stage_record$token)
  stage <- as.character(stage_record$stage)
  stage_time <- if (is.null(stage_record$stage_time)) NA_real_ else as.numeric(stage_record$stage_time)
  stage_dir <- file.path(input_dir, "stages", token)
  expression <- as(Matrix::readMM(file.path(stage_dir, "expression_genes_by_cells.mtx")), "dgCMatrix")
  genes <- readLines(file.path(stage_dir, "genes.txt"), warn = FALSE)
  metadata <- read.csv(file.path(stage_dir, "metadata.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  spatial <- read.csv(file.path(stage_dir, "spatial_aligned.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  if (nrow(expression) != length(genes) || ncol(expression) != nrow(metadata)) {
    stop(paste("Matrix/metadata/gene dimension mismatch for", stage))
  }
  if (!identical(as.character(metadata$cell_id), as.character(spatial$cell_id))) {
    stop(paste("Metadata/spatial cell order mismatch for", stage))
  }
  if (anyDuplicated(genes) || anyDuplicated(metadata$cell_id)) {
    stop(paste("Duplicate genes or cell IDs for", stage))
  }
  if (length(expression@x) && (any(!is.finite(expression@x)) || any(expression@x < 0))) {
    stop(paste("Invalid prepared single-log expression for", stage))
  }
  rownames(expression) <- genes
  colnames(expression) <- metadata$cell_id
  rownames(metadata) <- metadata$cell_id
  metadata$label <- droplevels(factor(metadata$label))
  cell_counts <- table(metadata$label)

  cellchat <- createCellChat(
    object = expression,
    meta = metadata,
    group.by = "label",
    do.sparse = TRUE
  )
  cellchat@DB <- database_use
  cellchat <- subsetData(cellchat)
  cellchat@LR$LRsig <- pair_lr
  cellchat <- computeCommunProb(
    cellchat,
    type = mean_method,
    LR.use = pair_lr,
    raw.use = TRUE,
    population.size = population_size,
    nboot = nboot,
    seed.use = seed_use
  )
  probabilities <- cellchat@net$prob
  pvalues <- cellchat@net$pval
  if (!identical(dim(probabilities), dim(pvalues))) {
    stop(paste("CellChat probability/p-value dimensions differ for", stage))
  }
  interaction_ids <- dimnames(probabilities)[[3L]]
  pair_order <- match(interaction_ids, rownames(pair_lr))
  if (anyNA(pair_order) || anyDuplicated(pair_order)) {
    stop(paste("Could not map CellChat interactions back to prepared database for", stage))
  }
  ordered_pairs <- pair_lr[pair_order, , drop = FALSE]
  ordered_flat <- flat_database[match(ordered_pairs$database_row, flat_database$database_row), , drop = FALSE]

  stage_lr <- vector("list", length(interaction_ids))
  for (interaction_position in seq_along(interaction_ids)) {
    score_matrix <- probabilities[, , interaction_position, drop = TRUE]
    pvalue_matrix <- pvalues[, , interaction_position, drop = TRUE]
    stage_lr[[interaction_position]] <- matrix_context(
      score_matrix, pvalue_matrix, stage, stage_time,
      ordered_flat$ligand[[interaction_position]],
      ordered_flat$receptor[[interaction_position]],
      ordered_flat$pathway[[interaction_position]],
      ordered_flat$category[[interaction_position]],
      interaction_ids[[interaction_position]],
      cell_counts,
      "official CellChat type-level LR communication probability",
      positive_only
    )
    if (nrow(stage_lr[[interaction_position]]) > 0L) {
      stage_lr[[interaction_position]]$database_row <- ordered_flat$database_row[[interaction_position]]
      stage_lr[[interaction_position]]$flat_interaction_id <- ordered_flat$interaction_id[[interaction_position]]
      stage_lr[[interaction_position]]$min10_eligible <- (
        stage_lr[[interaction_position]]$n_sender_cells > 10L &
        stage_lr[[interaction_position]]$n_receiver_cells > 10L
      )
    }
  }
  lr_frames[[token]] <- bind_nonempty(
    stage_lr,
    c("database_row", "flat_interaction_id", "min10_eligible")
  )

  stage_pathways <- list()
  pathways <- sort(unique(ordered_flat$pathway))
  for (pathway in pathways) {
    pathway_indices <- which(ordered_flat$pathway == pathway)
    raw_score <- apply(probabilities[, , pathway_indices, drop = FALSE], c(1L, 2L), sum)
    significant_score <- apply(
      ifelse(pvalues[, , pathway_indices, drop = FALSE] < 0.05,
             probabilities[, , pathway_indices, drop = FALSE], 0),
      c(1L, 2L), sum
    )
    categories <- paste(sort(unique(ordered_flat$category[pathway_indices])), collapse = ";")
    context <- matrix_context(
      raw_score, NULL, stage, stage_time, "", "", pathway, categories,
      paste0("pathway:", pathway), cell_counts,
      "sum of unthresholded official CellChat LR probabilities within pathway",
      positive_only
    )
    if (nrow(context) > 0L) {
      context$score_significant_p_lt_0p05 <- significant_score[
        cbind(match(context$sender_type, rownames(raw_score)),
              match(context$receiver_type, colnames(raw_score)))
      ]
      context$n_lr_rows <- length(pathway_indices)
    }
    stage_pathways[[pathway]] <- context
  }
  pathway_frames[[token]] <- bind_nonempty(
    stage_pathways,
    c("score_significant_p_lt_0p05", "n_lr_rows")
  )

  total_score <- apply(probabilities, c(1L, 2L), sum)
  total_significant <- apply(ifelse(pvalues < 0.05, probabilities, 0), c(1L, 2L), sum)
  total_context <- matrix_context(
    total_score, NULL, stage, stage_time, "", "", "__all__", "__all__", "total",
    cell_counts,
    "sum of unthresholded official CellChat LR probabilities over the prepared database",
    positive_only
  )
  if (nrow(total_context) > 0L) {
    total_context$score_significant_p_lt_0p05 <- total_significant[
      cbind(match(total_context$sender_type, rownames(total_score)),
            match(total_context$receiver_type, colnames(total_score)))
    ]
    total_context$n_lr_rows <- dim(probabilities)[[3L]]
  }
  total_frames[[token]] <- total_context

  diagnostic_frames[[token]] <- data.frame(
    stage = stage,
    stage_time = stage_time,
    n_cells = nrow(metadata),
    n_cell_types = length(cell_counts),
    n_lr_rows_requested = nrow(pair_lr),
    n_lr_rows_returned = dim(probabilities)[[3L]],
    n_positive_lr_contexts = sum(probabilities > 0),
    n_significant_lr_contexts = sum(probabilities > 0 & pvalues < 0.05),
    spatial_coordinates_read = TRUE,
    spatial_coordinates_used_by_cellchat = FALSE,
    stringsAsFactors = FALSE
  )
  if (save_rds) {
    rds_path <- file.path(output_dir, paste0("cellchat_network_", token, ".rds"))
    saveRDS(
      list(
        stage = stage,
        net = cellchat@net,
        idents = cellchat@idents,
        ordered_pairs = ordered_pairs,
        flat_database = ordered_flat,
        coordinates = spatial,
        options = cellchat@options
      ),
      rds_path,
      compress = "xz"
    )
    artifact_paths <- c(artifact_paths, rds_path)
  }
  message("Completed CellChat stage ", stage)
}

lr_all <- bind_nonempty(lr_frames, c("database_row", "flat_interaction_id", "min10_eligible"))
pathway_all <- bind_nonempty(pathway_frames, c("score_significant_p_lt_0p05", "n_lr_rows"))
total_all <- bind_nonempty(total_frames, c("score_significant_p_lt_0p05", "n_lr_rows"))
diagnostics <- do.call(rbind, diagnostic_frames)
rownames(lr_all) <- NULL
rownames(pathway_all) <- NULL
rownames(total_all) <- NULL
rownames(diagnostics) <- NULL

lr_path <- file.path(output_dir, "cellchat_lr_scores.csv.gz")
pathway_path <- file.path(output_dir, "cellchat_pathway_scores.csv.gz")
total_path <- file.path(output_dir, "cellchat_type_pair_scores.csv.gz")
diagnostic_path <- file.path(output_dir, "stage_diagnostics.csv")
write_csv_gz(lr_all, lr_path)
write_csv_gz(pathway_all, pathway_path)
write_csv_gz(total_all, total_path)
write.csv(diagnostics, diagnostic_path, row.names = FALSE, quote = TRUE)
artifact_paths <- c(artifact_paths, lr_path, pathway_path, total_path, diagnostic_path)

manifest <- list(
  schema_version = 1,
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  method = "CellChat",
  database_variant = "current_zebrafish_lr_database",
  input_manifest = file_record(input_manifest_path),
  database = file_record(database_path),
  database_validation = list(
    rule = "database_row + expanded ligand + expanded receptor + pathway + category must match installed official CellChatDB.zebrafish rowwise",
    rows_validated = nrow(flat_database),
    all_rows_match = TRUE
  ),
  design = list(
    all_prepared_observed_stages = TRUE,
    expression_retransformed_in_runner = FALSE,
    mean_method = mean_method,
    raw_use = TRUE,
    population_size = population_size,
    nboot = nboot,
    seed_use = seed_use,
    stage_prevalence_filter = NULL,
    min10 = "exported as an eligibility flag; primary probabilities are not discarded",
    spatial_coordinates = "prepared spatial_aligned coordinates are read, order-checked, and retained with RDS",
    spatial_coordinates_used_by_cellchat = FALSE,
    long_table_zero_policy = if (positive_only) {
      "structural zeros omitted; outer-join to input universe and fill zero for comparisons"
    } else {
      "all LR/type contexts exported"
    }
  ),
  score_semantics = list(
    lr_score = "official CellChat type-level LR communication probability",
    abundance_controlled_score = "native score because population.size=false in the primary run",
    pathway_score = "sum of unthresholded LR probabilities within pathway",
    type_pair_score = "sum of unthresholded LR probabilities over prepared database",
    significant_sensitivity = "same sum after retaining permutation p < 0.05",
    raw_cross_method_units_comparable = FALSE
  ),
  stage_diagnostics = diagnostics,
  software = list(
    R = R.version.string,
    platform = R.version$platform,
    CellChat = cellchat_version,
    CellChat_load_mode = cellchat_load_mode,
    CellChat_source = if (is.null(cellchat_source)) NULL else cellchat_source,
    CellChat_source_commit = cellchat_commit,
    CellChat_zebrafish_database = if (is.na(database_rda_path)) NULL else file_record(database_rda_path),
    Matrix = as.character(packageVersion("Matrix")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  artifacts = setNames(lapply(artifact_paths, file_record), basename(artifact_paths))
)
jsonlite::write_json(
  manifest,
  file.path(output_dir, "manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = NA,
  na = "null"
)
message("CellChat completed ", length(input_manifest$stages), " stages in ", output_dir)
