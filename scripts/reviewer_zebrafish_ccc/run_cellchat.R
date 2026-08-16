#!/usr/bin/env Rscript

# Run official CellChat once across every prepared observed spatial stage.
#
# The flat CSV is not merely cited in the manifest.  Its database_row values
# are resolved against the requested official CellChatDB species object and expanded ligand/receptor,
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

pinned_cellchat_commit <- "75253cd0c9e68410e6e721a6d3a0419a1d7e358f"
pinned_cellchat_version <- "2.2.0.9001"

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

cellchat_token_eligibility <- function(token, expression_genes, complex_table, gene_info) {
  token <- as.character(token)
  gene_symbols <- as.character(gene_info$Symbol)
  if (!length(token) || is.na(token) || !nzchar(token)) {
    return(list(
      eligible = FALSE,
      mode = "invalid_empty_token",
      required_genes = character(0),
      missing_genes = character(0),
      reason = "empty_token"
    ))
  }

  # This follows CellChat::extractGeneSubset exactly.  A simple ligand or
  # receptor is retained by subsetData only when it is an official geneInfo
  # symbol.  Any other token is interpreted as a declared complex, for which
  # every subunit must be present in the expression matrix.
  if (token %in% gene_symbols) {
    missing <- setdiff(token, expression_genes)
    return(list(
      eligible = length(missing) == 0L,
      mode = "gene_info_symbol",
      required_genes = token,
      missing_genes = missing,
      reason = if (length(missing)) "simple_gene_missing_expression" else ""
    ))
  }

  if (!(token %in% rownames(complex_table))) {
    return(list(
      eligible = FALSE,
      mode = "undeclared_token",
      required_genes = token,
      missing_genes = token,
      reason = "token_not_geneinfo_or_declared_complex"
    ))
  }
  subunit_columns <- grep("^subunit", colnames(complex_table), value = TRUE)
  if (!length(subunit_columns)) stop("CellChat complex table has no subunit columns")
  subunits <- as.character(unlist(
    complex_table[token, subunit_columns, drop = FALSE],
    use.names = FALSE
  ))
  subunits <- unique(subunits[!is.na(subunits) & nzchar(subunits)])
  if (!length(subunits)) {
    return(list(
      eligible = FALSE,
      mode = "declared_complex",
      required_genes = character(0),
      missing_genes = character(0),
      reason = "declared_complex_without_subunits"
    ))
  }
  missing <- setdiff(subunits, expression_genes)
  list(
    eligible = length(missing) == 0L,
    mode = "declared_complex",
    required_genes = subunits,
    missing_genes = missing,
    reason = if (length(missing)) "complex_subunit_missing_expression" else ""
  )
}

build_cellchat_database_eligibility <- function(
    flat_database, pair_lr, expression_genes, complex_table, gene_info) {
  if (nrow(flat_database) != nrow(pair_lr)) {
    stop("Flat and pinned CellChat interaction tables differ in length")
  }
  ligand_checks <- lapply(
    as.character(pair_lr$ligand),
    cellchat_token_eligibility,
    expression_genes = expression_genes,
    complex_table = complex_table,
    gene_info = gene_info
  )
  receptor_checks <- lapply(
    as.character(pair_lr$receptor),
    cellchat_token_eligibility,
    expression_genes = expression_genes,
    complex_table = complex_table,
    gene_info = gene_info
  )
  ligand_eligible <- vapply(ligand_checks, `[[`, logical(1L), "eligible")
  receptor_eligible <- vapply(receptor_checks, `[[`, logical(1L), "eligible")
  collapse_values <- function(checks, field) {
    vapply(checks, function(value) paste(value[[field]], collapse = ";"), character(1L))
  }
  exclusion_reason <- vapply(seq_len(nrow(pair_lr)), function(position) {
    reasons <- character(0)
    if (!ligand_eligible[[position]]) {
      reasons <- c(reasons, paste0("ligand:", ligand_checks[[position]]$reason))
    }
    if (!receptor_eligible[[position]]) {
      reasons <- c(reasons, paste0("receptor:", receptor_checks[[position]]$reason))
    }
    paste(reasons, collapse = "|")
  }, character(1L))
  data.frame(
    database_row = as.integer(flat_database$database_row),
    interaction_id = as.character(flat_database$interaction_id),
    current_ligand = as.character(flat_database$ligand),
    current_receptor = as.character(flat_database$receptor),
    cellchat_ligand_token = as.character(pair_lr$ligand),
    cellchat_receptor_token = as.character(pair_lr$receptor),
    ligand_mode = collapse_values(ligand_checks, "mode"),
    receptor_mode = collapse_values(receptor_checks, "mode"),
    ligand_required_genes = collapse_values(ligand_checks, "required_genes"),
    receptor_required_genes = collapse_values(receptor_checks, "required_genes"),
    ligand_missing_genes = collapse_values(ligand_checks, "missing_genes"),
    receptor_missing_genes = collapse_values(receptor_checks, "missing_genes"),
    ligand_eligible = ligand_eligible,
    receptor_eligible = receptor_eligible,
    eligible = ligand_eligible & receptor_eligible,
    exclusion_reason = exclusion_reason,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
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
    database_variant = database_variant,
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
cellchat_species <- if (is.null(args[["cellchat-species"]])) "zebrafish" else tolower(args[["cellchat-species"]])
if (!(cellchat_species %in% c("zebrafish", "mouse", "human"))) {
  stop("--cellchat-species must be zebrafish, mouse, or human")
}
database_variant <- if (is.null(args[["database-variant"]])) {
  paste0("project_", cellchat_species, "_lr_database")
} else {
  args[["database-variant"]]
}
database_object_name <- paste0("CellChatDB.", cellchat_species)
database_filename <- paste0(database_object_name, ".rda")
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
  database_rda_path <- file.path(cellchat_source, "data", database_filename)
  core_files <- file.path(
    cellchat_source,
    "R",
    c("CellChat_class.R", "utilities.R", "database.R", "modeling.R")
  )
  required_source_files <- c(description_path, database_rda_path, core_files)
  if (!all(file.exists(required_source_files))) {
    stop("--cellchat-source is missing DESCRIPTION, requested species database, or core R files")
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
  if (
    is.na(cellchat_commit) ||
      tolower(cellchat_commit) != pinned_cellchat_commit ||
      cellchat_version != pinned_cellchat_version
  ) {
    stop(
      "Pinned CellChat source mismatch: expected commit ",
      pinned_cellchat_commit,
      " and version ",
      pinned_cellchat_version
    )
  }
} else {
  suppressPackageStartupMessages(library(CellChat))
  data(list = database_object_name, package = "CellChat", envir = environment())
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

if (!exists(database_object_name)) {
  stop(paste("The installed official CellChat package does not provide", database_object_name))
}
official_database <- get(database_object_name)
if (is.null(rownames(official_database$complex)) || anyDuplicated(rownames(official_database$complex))) {
  stop("Pinned CellChat complex table must have unique non-null row names")
}
if (!("Symbol" %in% colnames(official_database$geneInfo))) {
  stop("Pinned CellChat geneInfo table has no Symbol column")
}
official_interaction <- official_database$interaction
official_ligand_expanded <- tolower(
  expand_complex(official_interaction$ligand, official_database$complex)
)
official_receptor_expanded <- tolower(
  expand_complex(official_interaction$receptor, official_database$complex)
)
official_structure_key <- paste(
  official_ligand_expanded,
  official_receptor_expanded,
  as.character(official_interaction$annotation),
  sep = "\r"
)
requested_structure_key <- paste(
  tolower(flat_database$ligand),
  tolower(flat_database$receptor),
  flat_database$category,
  sep = "\r"
)
requested_indices <- as.integer(flat_database$database_row) + 1L
resolve_official_index <- function(position) {
  requested_index <- requested_indices[[position]]
  requested_key <- requested_structure_key[[position]]
  if (
    !is.na(requested_index) &&
      requested_index >= 1L &&
      requested_index <= nrow(official_interaction) &&
      official_structure_key[[requested_index]] == requested_key
  ) {
    return(requested_index)
  }
  candidates <- which(official_structure_key == requested_key)
  if (length(candidates) > 1L) {
    same_pathway <- candidates[
      as.character(official_interaction$pathway_name[candidates]) ==
        flat_database$pathway[[position]]
    ]
    if (length(same_pathway) > 0L) candidates <- same_pathway
  }
  if (length(candidates) != 1L) {
    stop(paste(
      "Pinned CellChat database cannot uniquely resolve prepared database_row",
      flat_database$database_row[[position]],
      "by expanded ligand, receptor, category, and pathway"
    ))
  }
  candidates[[1L]]
}
indices <- vapply(seq_len(nrow(flat_database)), resolve_official_index, integer(1L))
rows_resolved_by_structural_key <- sum(indices != requested_indices)
pair_lr <- official_interaction[indices, , drop = FALSE]
ligand_expanded <- tolower(expand_complex(pair_lr$ligand, official_database$complex))
receptor_expanded <- tolower(expand_complex(pair_lr$receptor, official_database$complex))
database_structure_matches <- (
  ligand_expanded == tolower(flat_database$ligand) &
  receptor_expanded == tolower(flat_database$receptor) &
  as.character(pair_lr$annotation) == flat_database$category
)
if (!all(database_structure_matches)) {
  first_bad <- which(!database_structure_matches)[[1L]]
  stop(paste(
    "Installed CellChat database does not reproduce the ligand, receptor, and category at database_row",
    flat_database$database_row[[first_bad]]
  ))
}
official_pathway <- as.character(pair_lr$pathway_name)
pathway_mismatch <- official_pathway != flat_database$pathway
# The current project CSV is the requested database and therefore remains the
# authority for pathway labels.  Its ligand/receptor/category structure must
# still reproduce the pinned CellChat database exactly.  At present the only
# known difference is the auditable SOMATOSTATIN/SEMATOSTATIN label spelling;
# probabilities are unaffected, while pathway summaries retain the project
# CSV spelling instead of silently substituting CellChat's label.
pair_lr$pathway_name <- flat_database$pathway
pair_lr$annotation <- flat_database$category
# The requested flat LR database does not define CellChat's optional agonist,
# antagonist, or co-receptor fields.  Retaining those values from the official
# row would silently add biology outside the requested database and can require
# genes absent from the structurally filtered LR expression matrix.  Disable
# only these undeclared modifiers; ligand/receptor complexes remain intact.
undeclared_modifier_columns <- intersect(
  c("agonist", "antagonist", "co_A_receptor", "co_I_receptor"),
  colnames(pair_lr)
)
for (column in undeclared_modifier_columns) pair_lr[[column]] <- ""
pair_lr$database_row <- as.integer(flat_database$database_row)
pair_lr$flat_interaction_id <- flat_database$interaction_id

# Every prepared stage must expose the same global feature universe.  Test
# CellChat executability against that exact universe before calling subsetData:
# simple tokens must survive geneInfo filtering, and declared complexes must
# have every subunit.  Never repair an undeclared token by splitting on `_`,
# because doing so would invent CellChat complex semantics absent from the
# pinned database.
prepared_gene_lists <- lapply(input_manifest$stages, function(stage_record) {
  readLines(
    file.path(input_dir, "stages", as.character(stage_record$token), "genes.txt"),
    warn = FALSE
  )
})
if (!length(prepared_gene_lists) || !length(prepared_gene_lists[[1L]])) {
  stop("Prepared input manifest has no non-empty stage gene universe")
}
if (any(vapply(prepared_gene_lists, anyDuplicated, integer(1L)) > 0L)) {
  stop("Prepared stage gene lists contain duplicate names")
}
if (!all(vapply(
  prepared_gene_lists[-1L],
  identical,
  logical(1L),
  prepared_gene_lists[[1L]]
))) {
  stop("Prepared stages do not share an identical ordered gene universe")
}
expression_genes <- prepared_gene_lists[[1L]]
database_eligibility <- build_cellchat_database_eligibility(
  flat_database,
  pair_lr,
  expression_genes,
  official_database$complex,
  official_database$geneInfo
)
eligibility_path <- file.path(output_dir, "database_eligibility_audit.csv")
exclusion_path <- file.path(output_dir, "excluded_lr_rows.csv")
write.csv(database_eligibility, eligibility_path, row.names = FALSE, quote = TRUE)
write.csv(
  database_eligibility[!database_eligibility$eligible, , drop = FALSE],
  exclusion_path,
  row.names = FALSE,
  quote = TRUE
)
pair_lr_requested <- pair_lr
pair_lr <- pair_lr_requested[database_eligibility$eligible, , drop = FALSE]
flat_database_eligible <- flat_database[database_eligibility$eligible, , drop = FALSE]
if (!nrow(pair_lr)) {
  stop("No requested LR rows are executable by pinned CellChat and the prepared expression universe")
}
if (anyDuplicated(rownames(pair_lr))) {
  stop("Eligible CellChat interaction names must be unique")
}
database_use <- official_database
database_use$interaction <- pair_lr

lr_frames <- list()
pathway_frames <- list()
total_frames <- list()
diagnostic_frames <- list()
artifact_paths <- character(0)
artifact_paths <- c(artifact_paths, eligibility_path, exclusion_path)

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
  ordered_flat <- flat_database_eligible[
    match(ordered_pairs$database_row, flat_database_eligible$database_row),
    ,
    drop = FALSE
  ]

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
    FALSE
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
    n_lr_rows_requested = nrow(pair_lr_requested),
    n_lr_rows_eligible = nrow(pair_lr),
    n_lr_rows_excluded = nrow(pair_lr_requested) - nrow(pair_lr),
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
  database_variant = database_variant,
  cellchat_species = cellchat_species,
  official_database_object = database_object_name,
  input_manifest = file_record(input_manifest_path),
  database = file_record(database_path),
  database_validation = list(
    rule = paste0(
      "database_row is used when structurally identical; otherwise expanded ligand + ",
      "expanded receptor + category + pathway must uniquely resolve a pinned ",
      database_object_name,
      " row; pathway labels are taken exactly from the requested current project CSV"
    ),
    rows_validated = nrow(flat_database),
    all_structural_rows_match = TRUE,
    rows_resolved_by_structural_key = rows_resolved_by_structural_key,
    pathway_values_taken_from_current_csv = TRUE,
    official_pathway_mismatch_count = sum(pathway_mismatch),
    official_pathway_mismatch_database_rows = as.integer(flat_database$database_row[pathway_mismatch]),
    official_pathway_mismatch_labels = unique(data.frame(
      official = official_pathway[pathway_mismatch],
      current_csv = flat_database$pathway[pathway_mismatch],
      stringsAsFactors = FALSE
    )),
    undeclared_official_modifier_columns_disabled = undeclared_modifier_columns,
    modifier_policy = "agonist/antagonist/co-receptor fields are empty because the requested flat current LR database does not provide them",
    cellchat_executability_rule = "simple tokens must be exact geneInfo symbols present in every prepared stage; complex tokens must be declared in the pinned complex table and every exact subunit must be present; underscore splitting never invents an undeclared complex",
    rows_requested = nrow(pair_lr_requested),
    rows_eligible = nrow(pair_lr),
    rows_excluded = nrow(pair_lr_requested) - nrow(pair_lr),
    all_eligible_rows_cellchat_representable = TRUE,
    excluded_rows_are_method_unavailable_not_biological_zero = TRUE,
    eligibility_audit = file_record(eligibility_path),
    exclusion_table = file_record(exclusion_path)
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
      "LR and pathway structural zeros omitted; primary type-pair table exports the complete directed stage-specific cell-type square"
    } else {
      "all LR/type contexts exported"
    },
    type_pair_grid_export = list(
      complete_directed_stage_type_square = TRUE,
      zero_score_semantics = "evaluated CellChat total probability is exactly zero for this sender/receiver type pair",
      universe_source = "input_manifest.stages[].cell_type_counts",
      loader_zero_completion_required = FALSE
    ),
    method_unavailable_policy = "database rows listed in excluded_lr_rows.csv must be excluded from CellChat cross-method universes, never zero-filled"
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
    CellChat_species_database = if (is.na(database_rda_path)) NULL else file_record(database_rda_path),
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
