#!/usr/bin/env Rscript

# Run a receiver-centric temporal NicheNet analysis from precomputed expression
# and differential-expression tables. The official mouse v2 model files are
# supplied as input paths. In matched mode, the official ligand-target matrix is
# unchanged while the candidate ligand-receptor universe is restricted to a
# supplied singleton CellChat prior.

options(stringsAsFactors = FALSE, warn = 1)

NICHENET_FROZEN_COMMIT <- "66f90d5eeafef280b2b2f339b3fd70ffec1781dd"
NICHENET_FROZEN_VERSION <- "2.2.1.1"
NICHENET_SOURCE_FILES <- c(
  "R/supporting_functions.R",
  "R/evaluate_model_target_prediction.R",
  "R/evaluate_model_ligand_prediction.R",
  "R/application_prediction.R"
)
NICHENET_SOURCE_SHA256 <- c(
  "R/supporting_functions.R" =
    "8bbe379e4880ffd51b8fe29052b5ecbef254e17e42678e71817f37cd7a31389b",
  "R/evaluate_model_target_prediction.R" =
    "e120bc6cf97e9cf5ca9cb863522be6b5ab03d226f5c2175d25e314727b1aa2cb",
  "R/evaluate_model_ligand_prediction.R" =
    "dd4b27e29e5f03049c2bdb4ced166497b9724666c2116e24e37abc1371c0046d",
  "R/application_prediction.R" =
    "a3636b54bc58200924d4798f0398d5cbbcc37b1ac17e931aa61f900acfe85c7f"
)

usage <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript scripts/run_temporal_nichenet_reference.R \\\n",
    "    --input-dir INPUT_DIR \\\n",
    "    --out-dir OUT_DIR \\\n",
    "    --ligand-target-matrix ligand_target_matrix_nsga2r_final_mouse.rds \\\n",
    "    --lr-network lr_network_mouse_21122021.rds \\\n",
    "    --prior-mode default|matched [options]\n\n",
    "Required input files in INPUT_DIR:\n",
    "  receiver_de_genes.csv\n",
    "  receiver_expressed_genes.csv\n",
    "  sender_expressed_genes_long.csv\n",
    "  receiver_receptor_expression.csv\n",
    "  input_manifest.json (or manifest.json)\n\n",
    "Matched-prior option:\n",
    "  --matched-lr-tsv FILE   TSV/CSV with ligand/receptor or\n",
    "                          ligand_gene_symbol/receptor_gene_symbol\n\n",
    "NicheNet implementation backend:\n",
    "  --nichenetr-source DIR  Source the four required files from the official\n",
    "                          checkout at commit ",
    NICHENET_FROZEN_COMMIT, ".\n",
    "                          If omitted, use the installed nichenetr namespace.\n\n",
    "Analysis options:\n",
    "  --q-cutoff FLOAT         Used only when selected_target is absent [0.05]\n",
    "  --min-effect FLOAT       Used only when selected_target is absent [0.25]\n",
    "  --top-ligands INT        Ligands exported with target links [30]\n",
    "  --top-targets INT        Official NicheNet top targets per ligand [250]\n",
    "  --expected-*-sha256 HEX Optional asset/input checksum guards\n",
    "  --help\n"
  ))
}

parse_cli <- function(argv) {
  if (length(argv) == 0L || any(argv %in% c("--help", "-h"))) {
    usage()
    quit(save = "no", status = 0L)
  }
  if (length(argv) %% 2L != 0L) {
    stop("Every command-line option must be followed by a value; use --help.")
  }
  out <- list()
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    out[[substring(key, 3L)]] <- argv[[i + 1L]]
    i <- i + 2L
  }
  out
}

arg_or <- function(x, key, default = NULL) {
  value <- x[[key]]
  if (is.null(value)) default else value
}

must_integer <- function(x, label, minimum = 1L) {
  value <- suppressWarnings(as.integer(x))
  if (length(value) != 1L || is.na(value) || value < minimum) {
    stop(label, " must be an integer >= ", minimum, ".")
  }
  value
}

must_number <- function(x, label) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) {
    stop(label, " must be a finite number.")
  }
  value
}

must_file <- function(path, label) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    stop(label, " does not exist: ", ifelse(is.null(path), "<missing>", path))
  }
  normalizePath(path, mustWork = TRUE)
}

path_or_default <- function(value, input_dir, filename) {
  if (is.null(value)) file.path(input_dir, filename) else value
}

resolve_manifest <- function(value, input_dir) {
  if (!is.null(value)) return(must_file(value, "input manifest"))
  candidates <- file.path(input_dir, c("input_manifest.json", "manifest.json"))
  found <- candidates[file.exists(candidates)]
  if (length(found) == 0L) {
    stop("No input_manifest.json or manifest.json found in ", input_dir, ".")
  }
  normalizePath(found[[1L]], mustWork = TRUE)
}

read_delimited <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("tsv", "tab", "txt")) {
    utils::read.delim(path, check.names = FALSE, comment.char = "", quote = "\"",
                      stringsAsFactors = FALSE)
  } else {
    utils::read.csv(path, check.names = FALSE, comment.char = "", quote = "\"",
                    stringsAsFactors = FALSE)
  }
}

resolve_col <- function(df, candidates, label, required = TRUE) {
  lower_names <- tolower(names(df))
  idx <- match(tolower(candidates), lower_names)
  idx <- idx[!is.na(idx)]
  if (length(idx) > 0L) return(names(df)[idx[[1L]]])
  if (required) {
    stop(label, " is missing; accepted columns: ", paste(candidates, collapse = ", "))
  }
  NULL
}

clean_character <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x %in% c("", "NA", "NaN", "None", "null")] <- NA_character_
  x
}

logical_or_na <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x)) {
    out <- rep(NA, length(x))
    out[!is.na(x)] <- x[!is.na(x)] != 0
    return(out)
  }
  y <- tolower(trimws(as.character(x)))
  out <- rep(NA, length(y))
  out[y %in% c("true", "t", "1", "yes", "y")] <- TRUE
  out[y %in% c("false", "f", "0", "no", "n")] <- FALSE
  out
}

numeric_or_na <- function(x) suppressWarnings(as.numeric(x))

max_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else max(x)
}

collapse_unique <- function(x) {
  x <- sort(unique(clean_character(x)))
  x <- x[!is.na(x)]
  paste(x, collapse = ";")
}

sha256_file <- function(path) {
  path <- must_file(path, "file to hash")
  if (requireNamespace("digest", quietly = TRUE)) {
    return(tolower(digest::digest(file = path, algo = "sha256", serialize = FALSE)))
  }
  sha256sum <- Sys.which("sha256sum")
  if (nzchar(sha256sum)) {
    result <- system2(sha256sum, shQuote(path), stdout = TRUE, stderr = TRUE)
    status <- attr(result, "status")
    if (is.null(status) || identical(status, 0L)) {
      return(tolower(strsplit(result[[1L]], "[[:space:]]+")[[1L]][[1L]]))
    }
  }
  shasum <- Sys.which("shasum")
  if (nzchar(shasum)) {
    result <- system2(shasum, c("-a", "256", shQuote(path)), stdout = TRUE, stderr = TRUE)
    status <- attr(result, "status")
    if (is.null(status) || identical(status, 0L)) {
      return(tolower(strsplit(result[[1L]], "[[:space:]]+")[[1L]][[1L]]))
    }
  }
  stop("No SHA-256 implementation found (install digest or provide sha256sum/shasum).")
}

check_expected_hash <- function(path, expected, label) {
  actual <- sha256_file(path)
  if (!is.null(expected) && nzchar(expected) &&
      !identical(tolower(trimws(expected)), actual)) {
    stop(label, " SHA-256 mismatch: expected ", expected, ", observed ", actual)
  }
  actual
}

initialize_nichenet_backend <- function(source_root = NULL) {
  if (is.null(source_root) || !nzchar(source_root)) {
    if (!requireNamespace("nichenetr", quietly = TRUE)) {
      stop(
        "The nichenetr R package is not installed. Provide --nichenetr-source ",
        "pointing to the frozen official checkout at commit ", NICHENET_FROZEN_COMMIT, "."
      )
    }
    return(list(
      predict_ligand_activities = getExportedValue("nichenetr", "predict_ligand_activities"),
      get_weighted_ligand_target_links =
        getExportedValue("nichenetr", "get_weighted_ligand_target_links"),
      info = list(
        backend = "installed_namespace",
        package_version = as.character(utils::packageVersion("nichenetr")),
        frozen_source_commit = NULL,
        source_root = NULL,
        source_file_sha256 = NULL,
        loaded_namespaces = "nichenetr"
      )
    ))
  }

  source_root <- normalizePath(source_root, mustWork = TRUE)
  git_binary <- Sys.which("git")
  if (!nzchar(git_binary)) {
    stop("git is required to validate --nichenetr-source commit provenance.")
  }
  commit_output <- suppressWarnings(system2(
    git_binary,
    c("-C", shQuote(source_root), "rev-parse", "HEAD"),
    stdout = TRUE,
    stderr = TRUE
  ))
  commit_status <- attr(commit_output, "status")
  if (!is.null(commit_status) && !identical(commit_status, 0L)) {
    stop(
      "--nichenetr-source must be a git checkout; git rev-parse failed: ",
      paste(commit_output, collapse = " ")
    )
  }
  observed_commit <- tolower(trimws(commit_output[[1L]]))
  if (!identical(observed_commit, NICHENET_FROZEN_COMMIT)) {
    stop(
      "NicheNet source commit mismatch: expected ", NICHENET_FROZEN_COMMIT,
      ", observed ", observed_commit, "."
    )
  }

  source_paths <- file.path(source_root, NICHENET_SOURCE_FILES)
  names(source_paths) <- NICHENET_SOURCE_FILES
  missing_sources <- names(source_paths)[!file.exists(source_paths)]
  if (length(missing_sources) > 0L) {
    stop("Frozen NicheNet checkout is missing: ", paste(missing_sources, collapse = ", "))
  }
  observed_hashes <- vapply(source_paths, sha256_file, character(1L))
  mismatched <- names(observed_hashes)[
    tolower(observed_hashes) != tolower(NICHENET_SOURCE_SHA256[names(observed_hashes)])
  ]
  if (length(mismatched) > 0L) {
    details <- paste0(
      mismatched, " expected=", NICHENET_SOURCE_SHA256[mismatched],
      " observed=", observed_hashes[mismatched]
    )
    stop(
      "Frozen NicheNet source file SHA-256 mismatch: ",
      paste(details, collapse = "; ")
    )
  }

  description_path <- must_file(file.path(source_root, "DESCRIPTION"),
                                "NicheNet DESCRIPTION")
  description <- read.dcf(description_path)
  source_version <- unname(description[1L, "Version"])
  if (!identical(source_version, NICHENET_FROZEN_VERSION)) {
    stop(
      "Frozen NicheNet DESCRIPTION version mismatch: expected ",
      NICHENET_FROZEN_VERSION, ", observed ", source_version, "."
    )
  }

  required_namespaces <- c(
    "magrittr", "dplyr", "tibble", "tidyr", "ROCR", "caTools", "data.table", "rlang"
  )
  missing_namespaces <- required_namespaces[
    !vapply(required_namespaces, requireNamespace, logical(1L), quietly = TRUE)
  ]
  if (length(missing_namespaces) > 0L) {
    stop(
      "Frozen-source NicheNet backend requires already-installed namespaces: ",
      paste(missing_namespaces, collapse = ", "),
      ". No package installation is performed by this runner."
    )
  }

  # Source into an isolated environment and inject only the unqualified symbols
  # exercised by predict_ligand_activities/get_weighted_ligand_target_links.
  # This avoids attaching the full tidyverse or unrelated NicheNet dependencies.
  source_env <- new.env(parent = globalenv())
  imports <- list(
    "%>%" = getExportedValue("magrittr", "%>%"),
    bind_rows = getExportedValue("dplyr", "bind_rows"),
    select = getExportedValue("dplyr", "select"),
    rename = getExportedValue("dplyr", "rename"),
    bind_cols = getExportedValue("dplyr", "bind_cols"),
    inner_join = getExportedValue("dplyr", "inner_join"),
    mutate = getExportedValue("dplyr", "mutate"),
    desc = getExportedValue("dplyr", "desc"),
    tibble = getExportedValue("tibble", "tibble"),
    replace_na = getExportedValue("tidyr", "replace_na")
  )
  list2env(imports, envir = source_env)
  for (path in unname(source_paths)) {
    sys.source(path, envir = source_env, keep.source = TRUE)
  }

  required_functions <- c("predict_ligand_activities", "get_weighted_ligand_target_links")
  missing_functions <- required_functions[
    !vapply(required_functions, exists, logical(1L), envir = source_env,
            mode = "function", inherits = FALSE)
  ]
  if (length(missing_functions) > 0L) {
    stop(
      "Frozen NicheNet source did not define required functions: ",
      paste(missing_functions, collapse = ", ")
    )
  }
  list(
    predict_ligand_activities =
      get("predict_ligand_activities", envir = source_env, inherits = FALSE),
    get_weighted_ligand_target_links =
      get("get_weighted_ligand_target_links", envir = source_env, inherits = FALSE),
    info = list(
      backend = "frozen_official_source",
      package_version = source_version,
      frozen_source_commit = observed_commit,
      source_root = source_root,
      source_file_sha256 = as.list(observed_hashes),
      description_sha256 = sha256_file(description_path),
      loaded_namespaces = required_namespaces,
      injected_unqualified_symbols = names(imports)
    )
  )
}

bind_rows_fill <- function(rows, template) {
  rows <- Filter(function(x) !is.null(x) && nrow(x) > 0L, rows)
  if (length(rows) == 0L) return(template[0, , drop = FALSE])
  all_names <- unique(c(names(template), unlist(lapply(rows, names), use.names = FALSE)))
  normalize <- function(x) {
    missing <- setdiff(all_names, names(x))
    for (column in missing) x[[column]] <- NA
    x[, all_names, drop = FALSE]
  }
  out <- do.call(rbind, lapply(rows, normalize))
  rownames(out) <- NULL
  out
}

write_csv_stable <- function(df, path) {
  utils::write.csv(df, path, row.names = FALSE, na = "", quote = TRUE)
}

normalize_de <- function(df) {
  transition_col <- resolve_col(df, c("transition", "time_transition", "contrast"),
                                "receiver DE transition")
  receiver_col <- resolve_col(df, c("receiver", "receiver_label", "cell_type", "celltype"),
                              "receiver DE receiver")
  gene_col <- resolve_col(df, c("gene", "gene_symbol", "target"), "receiver DE gene")
  selected_col <- resolve_col(df, c("selected_target", "is_target", "target_selected"),
                              "receiver DE selected-target flag", required = FALSE)
  q_col <- resolve_col(df, c("q_value", "qvalue", "padj", "fdr", "adj_p_value"),
                       "receiver DE adjusted p-value", required = FALSE)
  effect_col <- resolve_col(
    df,
    c("effect", "log2fc", "log2fc_late_vs_early", "log2fc_response_vs_baseline",
      "beta", "time_effect", "coefficient"),
    "receiver DE effect", required = FALSE
  )
  if (is.null(selected_col) && (is.null(q_col) || is.null(effect_col))) {
    stop("receiver_de_genes.csv must have selected_target, or both q-value and effect columns.")
  }
  out <- data.frame(
    transition = clean_character(df[[transition_col]]),
    receiver = clean_character(df[[receiver_col]]),
    gene = clean_character(df[[gene_col]]),
    q_value = if (is.null(q_col)) NA_real_ else numeric_or_na(df[[q_col]]),
    effect = if (is.null(effect_col)) NA_real_ else numeric_or_na(df[[effect_col]]),
    selected_target = if (is.null(selected_col)) NA else logical_or_na(df[[selected_col]]),
    stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$transition) & !is.na(out$receiver) & !is.na(out$gene), , drop = FALSE]
  attr(out, "selected_column_present") <- !is.null(selected_col)
  attr(out, "source_columns") <- list(
    transition = transition_col, receiver = receiver_col, gene = gene_col,
    selected_target = selected_col, q_value = q_col, effect = effect_col
  )
  out
}

normalize_receiver_expression <- function(df, receptor = FALSE) {
  prefix <- if (receptor) "receiver receptor expression" else "receiver expression"
  transition_col <- resolve_col(df, c("transition", "time_transition", "contrast"),
                                paste(prefix, "transition"))
  receiver_col <- resolve_col(df, c("receiver", "receiver_label", "cell_type", "celltype"),
                              paste(prefix, "receiver"))
  gene_candidates <- if (receptor) c("gene", "receptor", "gene_symbol", "receptor_gene_symbol") else
    c("gene", "gene_symbol")
  gene_col <- resolve_col(df, gene_candidates, paste(prefix, "gene"))
  mean_col <- resolve_col(df, c("mean_expression", "mean_expr", "avg_expression", "expression_mean"),
                          paste(prefix, "mean expression"), required = FALSE)
  fraction_col <- resolve_col(
    df,
    c("expression_fraction", "fraction_expressed", "pct_expressed", "detection_fraction",
      "fraction_expressing"),
    paste(prefix, "expression fraction"), required = FALSE
  )
  out <- data.frame(
    transition = clean_character(df[[transition_col]]),
    receiver = clean_character(df[[receiver_col]]),
    gene = clean_character(df[[gene_col]]),
    mean_expression = if (is.null(mean_col)) NA_real_ else numeric_or_na(df[[mean_col]]),
    expression_fraction = if (is.null(fraction_col)) NA_real_ else numeric_or_na(df[[fraction_col]]),
    stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$transition) & !is.na(out$receiver) & !is.na(out$gene), , drop = FALSE]
  if (nrow(out) == 0L) return(out)
  stats <- stats::aggregate(
    out[, c("mean_expression", "expression_fraction"), drop = FALSE],
    by = out[, c("transition", "receiver", "gene"), drop = FALSE],
    FUN = max_or_na
  )
  rownames(stats) <- NULL
  stats
}

normalize_sender_expression <- function(df) {
  transition_col <- resolve_col(df, c("transition", "time_transition", "contrast"),
                                "sender expression transition")
  sender_col <- resolve_col(df, c("sender", "sender_label", "cell_type", "celltype"),
                            "sender expression sender")
  gene_col <- resolve_col(df, c("gene", "gene_symbol", "ligand"), "sender expression gene")
  mean_col <- resolve_col(df, c("mean_expression", "mean_expr", "avg_expression", "expression_mean"),
                          "sender mean expression", required = FALSE)
  fraction_col <- resolve_col(
    df,
    c("expression_fraction", "fraction_expressed", "pct_expressed", "detection_fraction",
      "fraction_expressing"),
    "sender expression fraction", required = FALSE
  )
  out <- data.frame(
    transition = clean_character(df[[transition_col]]),
    sender = clean_character(df[[sender_col]]),
    gene = clean_character(df[[gene_col]]),
    mean_expression = if (is.null(mean_col)) NA_real_ else numeric_or_na(df[[mean_col]]),
    expression_fraction = if (is.null(fraction_col)) NA_real_ else numeric_or_na(df[[fraction_col]]),
    stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$transition) & !is.na(out$sender) & !is.na(out$gene), , drop = FALSE]
  if (nrow(out) == 0L) return(out)
  stats <- stats::aggregate(
    out[, c("mean_expression", "expression_fraction"), drop = FALSE],
    by = out[, c("transition", "sender", "gene"), drop = FALSE],
    FUN = max_or_na
  )
  rownames(stats) <- NULL
  stats
}

normalize_lr <- function(df, label) {
  ligand_col <- resolve_col(df, c("ligand", "from", "ligand_gene_symbol", "source"),
                            paste(label, "ligand"))
  receptor_col <- resolve_col(df, c("receptor", "to", "receptor_gene_symbol", "target"),
                              paste(label, "receptor"))
  out <- data.frame(
    ligand = clean_character(df[[ligand_col]]),
    receptor = clean_character(df[[receptor_col]]),
    stringsAsFactors = FALSE
  )
  out <- unique(out[!is.na(out$ligand) & !is.na(out$receptor), , drop = FALSE])
  rownames(out) <- NULL
  out
}

make_sender_signal <- function(mean_expression, expression_fraction) {
  out <- rep(1, length(mean_expression))
  use_fraction <- is.finite(expression_fraction) & expression_fraction > 0
  out[use_fraction] <- expression_fraction[use_fraction]
  use_mean <- is.finite(mean_expression) & mean_expression > 0
  out[use_mean] <- mean_expression[use_mean]
  out
}

make_receptor_support <- function(expression_fraction) {
  out <- rep(1, length(expression_fraction))
  use_fraction <- is.finite(expression_fraction) & expression_fraction > 0
  out[use_fraction] <- pmin(expression_fraction[use_fraction], 1)
  out
}

argv <- parse_cli(commandArgs(trailingOnly = TRUE))

input_dir <- normalizePath(arg_or(argv, "input-dir", stop("--input-dir is required.")),
                           mustWork = TRUE)
out_dir <- arg_or(argv, "out-dir", stop("--out-dir is required."))
ltm_path <- must_file(arg_or(argv, "ligand-target-matrix",
                             stop("--ligand-target-matrix is required.")),
                      "ligand-target matrix")
official_lr_path <- must_file(arg_or(argv, "lr-network", stop("--lr-network is required.")),
                              "official mouse LR network")
prior_mode <- tolower(arg_or(argv, "prior-mode", "default"))
if (!prior_mode %in% c("default", "matched")) {
  stop("--prior-mode must be default or matched.")
}
matched_lr_path <- arg_or(argv, "matched-lr-tsv", NULL)
if (prior_mode == "matched") {
  matched_lr_path <- must_file(matched_lr_path, "matched singleton LR table")
}

q_cutoff <- must_number(arg_or(argv, "q-cutoff", "0.05"), "--q-cutoff")
min_effect <- must_number(arg_or(argv, "min-effect", "0.25"), "--min-effect")
top_ligands <- must_integer(arg_or(argv, "top-ligands", "30"), "--top-ligands")
top_targets <- must_integer(arg_or(argv, "top-targets", "250"), "--top-targets")

de_path <- must_file(path_or_default(arg_or(argv, "receiver-de", NULL), input_dir,
                                     "receiver_de_genes.csv"), "receiver DE table")
receiver_expr_path <- must_file(path_or_default(arg_or(argv, "receiver-expression", NULL), input_dir,
                                                "receiver_expressed_genes.csv"),
                                "receiver expressed genes table")
sender_expr_path <- must_file(path_or_default(arg_or(argv, "sender-expression", NULL), input_dir,
                                              "sender_expressed_genes_long.csv"),
                              "sender expressed genes table")
receptor_expr_path <- must_file(path_or_default(arg_or(argv, "receptor-expression", NULL), input_dir,
                                                "receiver_receptor_expression.csv"),
                                "receiver receptor expression table")
input_manifest_path <- resolve_manifest(arg_or(argv, "input-manifest", NULL), input_dir)

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("The jsonlite R package is required.")
nichenet_backend <- initialize_nichenet_backend(arg_or(argv, "nichenetr-source", NULL))
predict_ligand_activities_fn <- nichenet_backend$predict_ligand_activities
get_weighted_ligand_target_links_fn <- nichenet_backend$get_weighted_ligand_target_links

asset_hashes <- list(
  ligand_target_matrix_sha256 = check_expected_hash(
    ltm_path, arg_or(argv, "expected-ligand-target-sha256", NULL), "ligand-target matrix"
  ),
  official_lr_network_sha256 = check_expected_hash(
    official_lr_path, arg_or(argv, "expected-lr-network-sha256", NULL), "official LR network"
  ),
  input_manifest_sha256 = check_expected_hash(
    input_manifest_path, arg_or(argv, "expected-input-manifest-sha256", NULL), "input manifest"
  )
)
if (prior_mode == "matched") {
  asset_hashes$matched_lr_sha256 <- check_expected_hash(
    matched_lr_path, arg_or(argv, "expected-matched-lr-sha256", NULL), "matched LR table"
  )
}

input_hashes <- list(
  receiver_de_genes_sha256 = sha256_file(de_path),
  receiver_expressed_genes_sha256 = sha256_file(receiver_expr_path),
  sender_expressed_genes_long_sha256 = sha256_file(sender_expr_path),
  receiver_receptor_expression_sha256 = sha256_file(receptor_expr_path)
)

upstream_manifest <- tryCatch(
  jsonlite::fromJSON(input_manifest_path, simplifyVector = FALSE),
  error = function(e) stop("Could not parse input manifest: ", conditionMessage(e))
)

de <- normalize_de(read_delimited(de_path))
selected_column_present <- isTRUE(attr(de, "selected_column_present"))
de_source_columns <- attr(de, "source_columns")
receiver_expr <- normalize_receiver_expression(read_delimited(receiver_expr_path), receptor = FALSE)
sender_expr <- normalize_sender_expression(read_delimited(sender_expr_path))
receptor_expr <- normalize_receiver_expression(read_delimited(receptor_expr_path), receptor = TRUE)

ligand_target_matrix <- readRDS(ltm_path)
if (is.null(dim(ligand_target_matrix)) || length(dim(ligand_target_matrix)) != 2L ||
    is.null(rownames(ligand_target_matrix)) || is.null(colnames(ligand_target_matrix))) {
  stop("The ligand-target RDS must be a target-by-ligand matrix with row and column names.")
}
if (anyDuplicated(rownames(ligand_target_matrix)) || anyDuplicated(colnames(ligand_target_matrix))) {
  stop("The ligand-target matrix must have unique target row names and ligand column names.")
}

official_lr <- normalize_lr(as.data.frame(readRDS(official_lr_path)), "official LR network")
if (nrow(official_lr) == 0L) stop("Official LR network contains no valid ligand-receptor rows.")
if (prior_mode == "matched") {
  matched_lr <- normalize_lr(read_delimited(matched_lr_path), "matched LR table")
  if (nrow(matched_lr) == 0L) stop("Matched LR table contains no valid ligand-receptor rows.")
  if (any(grepl("_", matched_lr$ligand, fixed = TRUE)) ||
      any(grepl("_", matched_lr$receptor, fixed = TRUE))) {
    stop(
      "Matched LR table contains '_' complex/subunit encodings. ",
      "Provide the frozen singleton-only LR export."
    )
  }
  active_lr <- matched_lr
  active_lr_label <- "frozen_cellchat_singleton_lr"
} else {
  matched_lr <- NULL
  active_lr <- official_lr
  active_lr_label <- "official_nichenet_mouse_v2_lr_network"
}

pair_key <- function(df) paste(df$ligand, df$receptor, sep = "\r")
official_keys <- unique(pair_key(official_lr))
active_keys <- unique(pair_key(active_lr))
prior_coverage <- data.frame(
  prior_mode = prior_mode,
  active_lr_label = active_lr_label,
  n_active_lr_edges = nrow(active_lr),
  n_active_ligands = length(unique(active_lr$ligand)),
  n_active_receptors = length(unique(active_lr$receptor)),
  n_official_lr_edges = nrow(official_lr),
  n_active_edges_in_official_lr = sum(active_keys %in% official_keys),
  fraction_active_edges_in_official_lr = sum(active_keys %in% official_keys) / length(active_keys),
  n_active_ligands_in_ligand_target_matrix = sum(unique(active_lr$ligand) %in%
                                                   colnames(ligand_target_matrix)),
  fraction_active_ligands_in_ligand_target_matrix =
    mean(unique(active_lr$ligand) %in% colnames(ligand_target_matrix)),
  stringsAsFactors = FALSE
)

contexts <- unique(de[, c("transition", "receiver"), drop = FALSE])
contexts <- contexts[order(contexts$transition, contexts$receiver), , drop = FALSE]
if (nrow(contexts) == 0L) stop("No transition-by-receiver contexts were found in receiver DE input.")

activity_template <- data.frame(
  prior_mode = character(), transition = character(), receiver = character(),
  test_ligand = character(), ligand = character(), auroc = numeric(), aupr = numeric(),
  aupr_corrected = numeric(), pearson = numeric(), ligand_activity_rank = integer(),
  n_candidate_senders = integer(), candidate_senders = character(),
  n_candidate_receptors = integer(), candidate_receptors = character(),
  n_target_genes = integer(), n_background_genes = integer(), stringsAsFactors = FALSE
)
link_template <- data.frame(
  prior_mode = character(), transition = character(), receiver = character(),
  ligand = character(), target = character(), regulatory_potential = numeric(),
  ligand_activity_rank = integer(), ligand_aupr_corrected = numeric(),
  ligand_pearson = numeric(), stringsAsFactors = FALSE
)
candidate_template <- data.frame(
  prior_mode = character(), transition = character(), sender = character(), receiver = character(),
  ligand = character(), receptor = character(), sender_mean_expression = numeric(),
  sender_expression_fraction = numeric(), sender_expression_signal = numeric(),
  receiver_receptor_mean_expression = numeric(), receiver_receptor_expression_fraction = numeric(),
  receiver_receptor_support = numeric(), stringsAsFactors = FALSE
)
component_template <- data.frame(
  prior_mode = character(), transition = character(), sender = character(), receiver = character(),
  ligand = character(), ligand_aupr_corrected = numeric(), positive_ligand_activity = numeric(),
  sender_expression_signal = numeric(), sender_expression_share = numeric(),
  receiver_receptor_support = numeric(), sender_support_component = numeric(),
  stringsAsFactors = FALSE
)
support_template <- data.frame(
  prior_mode = character(), transition = character(), sender = character(), receiver = character(),
  sender_support_score = numeric(), sender_support_fraction = numeric(),
  n_candidate_ligands = integer(), n_positive_activity_ligands = integer(),
  score_definition = character(), is_native_nichenet_edge_strength = logical(),
  stringsAsFactors = FALSE
)
coverage_template <- data.frame(
  prior_mode = character(), transition = character(), receiver = character(), status = character(),
  empty_or_skip_reason = character(), target_selection_rule = character(),
  n_de_rows = integer(), n_selected_targets_input = integer(), n_selected_targets_in_background = integer(),
  n_background_genes_input = integer(), n_background_genes_in_matrix = integer(),
  n_receiver_receptors_input = integer(), n_sender_types_input = integer(),
  n_sender_ligands_input = integer(), n_lr_prior_edges_total = integer(),
  n_lr_edges_with_expressed_receptor = integer(), n_directed_candidate_lr_edges = integer(),
  n_potential_ligands = integer(), n_ligand_activities_returned = integer(),
  n_ligand_target_links_returned = integer(), stringsAsFactors = FALSE
)
gene_set_template <- data.frame(
  prior_mode = character(), transition = character(), receiver = character(), gene = character(),
  set = character(), stringsAsFactors = FALSE
)

activity_rows <- list()
link_rows <- list()
candidate_rows <- list()
component_rows <- list()
support_rows <- list()
coverage_rows <- list()
gene_set_rows <- list()

score_definition <- paste0(
  "CUSTOM (not native NicheNet edge strength): sum_l[max(aupr_corrected_l,0) * ",
  "sender_expression_share_(sender,l) * max_receptor_expression_fraction_(l,receiver)]. ",
  "Mean ligand expression is used for the sender share when available; otherwise detection ",
  "fraction, then binary expressed-gene presence. Receptor detection fraction is used when ",
  "available; otherwise binary expressed-receptor presence."
)

for (context_index in seq_len(nrow(contexts))) {
  transition_i <- contexts$transition[[context_index]]
  receiver_i <- contexts$receiver[[context_index]]
  de_i <- de[de$transition == transition_i & de$receiver == receiver_i, , drop = FALSE]
  bg_input <- unique(receiver_expr$gene[
    receiver_expr$transition == transition_i & receiver_expr$receiver == receiver_i
  ])
  bg_input <- bg_input[!is.na(bg_input)]
  background <- intersect(bg_input, rownames(ligand_target_matrix))

  if (selected_column_present && any(!is.na(de_i$selected_target))) {
    selected <- unique(de_i$gene[!is.na(de_i$selected_target) & de_i$selected_target])
    target_rule <- "upstream_selected_target"
  } else {
    selected <- unique(de_i$gene[
      is.finite(de_i$q_value) & de_i$q_value <= q_cutoff &
        is.finite(de_i$effect) & de_i$effect >= min_effect
    ])
    target_rule <- paste0("q_value<=", q_cutoff, "_and_effect>=", min_effect)
  }
  selected <- selected[!is.na(selected)]
  geneset <- intersect(selected, background)

  sender_i <- sender_expr[sender_expr$transition == transition_i, , drop = FALSE]
  receptor_i <- receptor_expr[
    receptor_expr$transition == transition_i & receptor_expr$receiver == receiver_i,
    , drop = FALSE
  ]
  sender_types <- sort(unique(sender_i$sender))
  sender_ligands <- unique(sender_i$gene)
  receiver_receptors <- unique(receptor_i$gene)

  lr_with_receptor <- active_lr[active_lr$receptor %in% receiver_receptors, , drop = FALSE]
  lr_context <- lr_with_receptor[
    lr_with_receptor$ligand %in% sender_ligands &
      lr_with_receptor$ligand %in% colnames(ligand_target_matrix),
    , drop = FALSE
  ]

  sender_join <- sender_i[, c("sender", "gene", "mean_expression", "expression_fraction"),
                          drop = FALSE]
  names(sender_join)[names(sender_join) == "gene"] <- "ligand"
  names(sender_join)[names(sender_join) == "mean_expression"] <- "sender_mean_expression"
  names(sender_join)[names(sender_join) == "expression_fraction"] <- "sender_expression_fraction"
  receptor_join <- receptor_i[, c("gene", "mean_expression", "expression_fraction"), drop = FALSE]
  names(receptor_join) <- c("receptor", "receiver_receptor_mean_expression",
                            "receiver_receptor_expression_fraction")

  if (nrow(lr_context) > 0L) {
    candidates <- merge(lr_context, sender_join, by = "ligand", all = FALSE, sort = FALSE)
    candidates <- merge(candidates, receptor_join, by = "receptor", all = FALSE, sort = FALSE)
    candidates <- unique(candidates)
    candidates$sender_expression_signal <- make_sender_signal(
      candidates$sender_mean_expression, candidates$sender_expression_fraction
    )
    candidates$receiver_receptor_support <- make_receptor_support(
      candidates$receiver_receptor_expression_fraction
    )
    candidates$prior_mode <- prior_mode
    candidates$transition <- transition_i
    candidates$receiver <- receiver_i
    candidates <- candidates[, names(candidate_template), drop = FALSE]
  } else {
    candidates <- candidate_template[0, , drop = FALSE]
  }
  candidate_rows[[length(candidate_rows) + 1L]] <- candidates
  potential_ligands <- sort(unique(candidates$ligand))

  target_set_rows <- gene_set_template[0, , drop = FALSE]
  if (length(geneset) > 0L) {
    target_set_rows <- data.frame(
      prior_mode = rep(prior_mode, length(geneset)),
      transition = rep(transition_i, length(geneset)),
      receiver = rep(receiver_i, length(geneset)),
      gene = geneset,
      set = rep("receiver_response_target", length(geneset)),
      stringsAsFactors = FALSE
    )
  }
  background_set_rows <- gene_set_template[0, , drop = FALSE]
  if (length(background) > 0L) {
    background_set_rows <- data.frame(
      prior_mode = rep(prior_mode, length(background)),
      transition = rep(transition_i, length(background)),
      receiver = rep(receiver_i, length(background)),
      gene = background,
      set = rep("receiver_expressed_background", length(background)),
      stringsAsFactors = FALSE
    )
  }
  gene_set_rows[[length(gene_set_rows) + 1L]] <- rbind(target_set_rows, background_set_rows)

  coverage <- list(
    prior_mode = prior_mode, transition = transition_i, receiver = receiver_i,
    status = "pending", empty_or_skip_reason = "", target_selection_rule = target_rule,
    n_de_rows = nrow(de_i), n_selected_targets_input = length(selected),
    n_selected_targets_in_background = length(geneset), n_background_genes_input = length(bg_input),
    n_background_genes_in_matrix = length(background),
    n_receiver_receptors_input = length(receiver_receptors),
    n_sender_types_input = length(sender_types), n_sender_ligands_input = length(sender_ligands),
    n_lr_prior_edges_total = nrow(active_lr),
    n_lr_edges_with_expressed_receptor = nrow(lr_with_receptor),
    n_directed_candidate_lr_edges = nrow(candidates), n_potential_ligands = length(potential_ligands),
    n_ligand_activities_returned = 0L, n_ligand_target_links_returned = 0L
  )

  skip_reason <- ""
  if (length(background) < 2L) {
    skip_reason <- "fewer_than_2_receiver_background_genes_in_ligand_target_matrix"
  } else if (length(geneset) == 0L) {
    skip_reason <- "no_selected_response_targets_in_receiver_background_and_ligand_target_matrix"
  } else if (length(setdiff(background, geneset)) == 0L) {
    skip_reason <- "no_negative_background_genes_after_target_selection"
  } else if (length(receiver_receptors) == 0L) {
    skip_reason <- "no_expressed_receiver_receptors"
  } else if (nrow(sender_i) == 0L) {
    skip_reason <- "no_expressed_sender_genes"
  } else if (nrow(lr_with_receptor) == 0L) {
    skip_reason <- "no_prior_lr_edges_with_expressed_receiver_receptors"
  } else if (nrow(candidates) == 0L) {
    skip_reason <- "no_sender_expressed_matrix_covered_ligands_with_expressed_receptors"
  } else if (length(potential_ligands) == 0L) {
    skip_reason <- "no_potential_ligands"
  }

  if (nzchar(skip_reason)) {
    coverage$status <- "skipped_empty_context"
    coverage$empty_or_skip_reason <- skip_reason
    coverage_rows[[length(coverage_rows) + 1L]] <- as.data.frame(coverage, stringsAsFactors = FALSE)
    next
  }

  nichenet_error <- NULL
  activities <- tryCatch(
    as.data.frame(predict_ligand_activities_fn(
      geneset = geneset,
      background_expressed_genes = background,
      ligand_target_matrix = ligand_target_matrix,
      potential_ligands = potential_ligands,
      single = TRUE
    )),
    error = function(e) {
      nichenet_error <<- conditionMessage(e)
      NULL
    }
  )
  if (is.null(activities) || nrow(activities) == 0L) {
    coverage$status <- if (is.null(nichenet_error)) "skipped_empty_activity" else "skipped_nichenet_error"
    coverage$empty_or_skip_reason <- if (is.null(nichenet_error)) {
      "predict_ligand_activities_returned_zero_rows"
    } else {
      paste0("predict_ligand_activities_error: ", nichenet_error)
    }
    coverage_rows[[length(coverage_rows) + 1L]] <- as.data.frame(coverage, stringsAsFactors = FALSE)
    next
  }

  expected_activity_columns <- c("test_ligand", "auroc", "aupr", "aupr_corrected", "pearson")
  for (column in expected_activity_columns) {
    if (!column %in% names(activities)) activities[[column]] <- NA
  }
  activities <- activities[, expected_activity_columns, drop = FALSE]
  activities$test_ligand <- clean_character(activities$test_ligand)
  activities <- activities[!is.na(activities$test_ligand), , drop = FALSE]
  for (column in c("auroc", "aupr", "aupr_corrected", "pearson")) {
    activities[[column]] <- numeric_or_na(activities[[column]])
  }
  if (nrow(activities) == 0L) {
    coverage$status <- "skipped_empty_activity"
    coverage$empty_or_skip_reason <- "predict_ligand_activities_returned_no_valid_ligands"
    coverage_rows[[length(coverage_rows) + 1L]] <- as.data.frame(coverage, stringsAsFactors = FALSE)
    next
  }

  activity_order_score <- activities$aupr_corrected
  activity_order_score[!is.finite(activity_order_score)] <- -Inf
  pearson_order_score <- activities$pearson
  pearson_order_score[!is.finite(pearson_order_score)] <- -Inf
  activity_order <- order(-activity_order_score, -pearson_order_score, activities$test_ligand)
  activities <- activities[activity_order, , drop = FALSE]
  activities$ligand_activity_rank <- seq_len(nrow(activities))
  activities$ligand <- activities$test_ligand

  candidate_summary <- lapply(activities$ligand, function(ligand_i) {
    subset_i <- candidates[candidates$ligand == ligand_i, , drop = FALSE]
    data.frame(
      ligand = ligand_i,
      n_candidate_senders = length(unique(subset_i$sender)),
      candidate_senders = collapse_unique(subset_i$sender),
      n_candidate_receptors = length(unique(subset_i$receptor)),
      candidate_receptors = collapse_unique(subset_i$receptor),
      stringsAsFactors = FALSE
    )
  })
  candidate_summary <- do.call(rbind, candidate_summary)
  activities <- merge(activities, candidate_summary, by = "ligand", all.x = TRUE, sort = FALSE)
  activities <- activities[match(candidate_summary$ligand, activities$ligand), , drop = FALSE]
  activities$prior_mode <- prior_mode
  activities$transition <- transition_i
  activities$receiver <- receiver_i
  activities$n_target_genes <- length(geneset)
  activities$n_background_genes <- length(background)
  activities <- activities[, names(activity_template), drop = FALSE]
  activity_rows[[length(activity_rows) + 1L]] <- activities

  selected_top_ligands <- head(activities$ligand, top_ligands)
  link_context <- list()
  for (ligand_i in selected_top_ligands) {
    links_i <- tryCatch(
      as.data.frame(get_weighted_ligand_target_links_fn(
        ligand = ligand_i, geneset = geneset,
        ligand_target_matrix = ligand_target_matrix, n = top_targets
      )),
      error = function(e) NULL
    )
    if (is.null(links_i) || nrow(links_i) == 0L || !all(c("ligand", "target", "weight") %in% names(links_i))) {
      next
    }
    links_i <- links_i[!is.na(links_i$target) & !is.na(links_i$weight), , drop = FALSE]
    if (nrow(links_i) == 0L) next
    act_i <- activities[activities$ligand == ligand_i, , drop = FALSE][1L, ]
    link_context[[length(link_context) + 1L]] <- data.frame(
      prior_mode = prior_mode, transition = transition_i, receiver = receiver_i,
      ligand = clean_character(links_i$ligand), target = clean_character(links_i$target),
      regulatory_potential = numeric_or_na(links_i$weight),
      ligand_activity_rank = act_i$ligand_activity_rank,
      ligand_aupr_corrected = act_i$aupr_corrected,
      ligand_pearson = act_i$pearson,
      stringsAsFactors = FALSE
    )
  }
  links <- bind_rows_fill(link_context, link_template)
  link_rows[[length(link_rows) + 1L]] <- links

  # Custom sender decomposition.  This is intentionally separate from the
  # receiver-centric NicheNet activity table and must not be described as a
  # native NicheNet cell-cell edge strength.
  sender_ligand <- unique(candidates[, c("sender", "ligand", "sender_expression_signal"), drop = FALSE])
  if (nrow(sender_ligand) > 0L) {
    ligand_denominator <- stats::aggregate(
      sender_expression_signal ~ ligand, data = sender_ligand, FUN = sum
    )
    names(ligand_denominator)[[2L]] <- "sender_signal_total"
    sender_ligand <- merge(sender_ligand, ligand_denominator, by = "ligand", all.x = TRUE,
                           sort = FALSE)
    sender_ligand$sender_expression_share <- ifelse(
      is.finite(sender_ligand$sender_signal_total) & sender_ligand$sender_signal_total > 0,
      sender_ligand$sender_expression_signal / sender_ligand$sender_signal_total,
      0
    )
    receptor_support <- stats::aggregate(
      receiver_receptor_support ~ ligand, data = candidates, FUN = max_or_na
    )
    sender_ligand <- merge(sender_ligand, receptor_support, by = "ligand", all.x = TRUE,
                           sort = FALSE)
    activity_for_support <- activities[, c("ligand", "aupr_corrected"), drop = FALSE]
    names(activity_for_support)[[2L]] <- "ligand_aupr_corrected"
    sender_ligand <- merge(sender_ligand, activity_for_support, by = "ligand", all.x = TRUE,
                           sort = FALSE)
    sender_ligand$positive_ligand_activity <- pmax(sender_ligand$ligand_aupr_corrected, 0,
                                                   na.rm = FALSE)
    sender_ligand$positive_ligand_activity[
      !is.finite(sender_ligand$positive_ligand_activity)
    ] <- 0
    sender_ligand$sender_support_component <-
      sender_ligand$positive_ligand_activity * sender_ligand$sender_expression_share *
      sender_ligand$receiver_receptor_support
    sender_ligand$prior_mode <- prior_mode
    sender_ligand$transition <- transition_i
    sender_ligand$receiver <- receiver_i
    components <- sender_ligand[, names(component_template), drop = FALSE]
    component_rows[[length(component_rows) + 1L]] <- components

    support <- stats::aggregate(
      sender_support_component ~ sender,
      data = components,
      FUN = function(x) sum(x, na.rm = TRUE)
    )
    names(support)[names(support) == "sender_support_component"] <- "sender_support_score"
    ligand_counts <- do.call(rbind, lapply(split(components, components$sender), function(df_i) {
      data.frame(
        sender = df_i$sender[[1L]],
        n_candidate_ligands = length(unique(df_i$ligand)),
        n_positive_activity_ligands = length(unique(df_i$ligand[
          is.finite(df_i$positive_ligand_activity) & df_i$positive_ligand_activity > 0
        ])),
        stringsAsFactors = FALSE
      )
    }))
    support <- merge(data.frame(sender = sender_types, stringsAsFactors = FALSE), support,
                     by = "sender", all.x = TRUE, sort = FALSE)
    support <- merge(support, ligand_counts, by = "sender", all.x = TRUE, sort = FALSE)
  } else {
    support <- data.frame(
      sender = sender_types, sender_support_score = 0, n_candidate_ligands = 0L,
      n_positive_activity_ligands = 0L, stringsAsFactors = FALSE
    )
  }
  support$sender_support_score[is.na(support$sender_support_score)] <- 0
  support$n_candidate_ligands[is.na(support$n_candidate_ligands)] <- 0L
  support$n_positive_activity_ligands[is.na(support$n_positive_activity_ligands)] <- 0L
  support_total <- sum(support$sender_support_score, na.rm = TRUE)
  support$sender_support_fraction <- if (support_total > 0) {
    support$sender_support_score / support_total
  } else {
    0
  }
  support$prior_mode <- prior_mode
  support$transition <- transition_i
  support$receiver <- receiver_i
  support$score_definition <- score_definition
  support$is_native_nichenet_edge_strength <- FALSE
  support <- support[, names(support_template), drop = FALSE]
  support <- support[order(-support$sender_support_score, support$sender), , drop = FALSE]
  support_rows[[length(support_rows) + 1L]] <- support

  coverage$status <- "ok"
  coverage$empty_or_skip_reason <- ""
  coverage$n_ligand_activities_returned <- nrow(activities)
  coverage$n_ligand_target_links_returned <- nrow(links)
  coverage_rows[[length(coverage_rows) + 1L]] <- as.data.frame(coverage, stringsAsFactors = FALSE)
}

activities_all <- bind_rows_fill(activity_rows, activity_template)
links_all <- bind_rows_fill(link_rows, link_template)
candidates_all <- bind_rows_fill(candidate_rows, candidate_template)
components_all <- bind_rows_fill(component_rows, component_template)
support_all <- bind_rows_fill(support_rows, support_template)
coverage_all <- bind_rows_fill(coverage_rows, coverage_template)
gene_sets_all <- bind_rows_fill(gene_set_rows, gene_set_template)

if (dir.exists(out_dir) && length(list.files(out_dir, all.files = TRUE, no.. = TRUE)) > 0L) {
  stop("Output directory is not empty; use a new directory to avoid mixing runs: ", out_dir)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)

output_paths <- c(
  ligand_activities = file.path(out_dir, "nichenet_ligand_activities.csv"),
  ligand_target_links = file.path(out_dir, "nichenet_top_ligand_target_links.csv"),
  sender_support_scores = file.path(out_dir, "nichenet_custom_sender_support_scores.csv"),
  sender_support_components = file.path(out_dir, "nichenet_custom_sender_support_components.csv"),
  candidate_lr_edges = file.path(out_dir, "nichenet_candidate_lr_edges.csv"),
  context_coverage = file.path(out_dir, "nichenet_context_coverage.csv"),
  analyzed_gene_sets = file.path(out_dir, "nichenet_analyzed_gene_sets.csv"),
  prior_coverage = file.path(out_dir, "nichenet_prior_coverage.csv")
)
write_csv_stable(activities_all, output_paths[["ligand_activities"]])
write_csv_stable(links_all, output_paths[["ligand_target_links"]])
write_csv_stable(support_all, output_paths[["sender_support_scores"]])
write_csv_stable(components_all, output_paths[["sender_support_components"]])
write_csv_stable(candidates_all, output_paths[["candidate_lr_edges"]])
write_csv_stable(coverage_all, output_paths[["context_coverage"]])
write_csv_stable(gene_sets_all, output_paths[["analyzed_gene_sets"]])
write_csv_stable(prior_coverage, output_paths[["prior_coverage"]])

method_notes_path <- file.path(out_dir, "METHOD_NOTES.txt")
writeLines(c(
  "Temporal NicheNet reference runner",
  "",
  paste0("Prior mode: ", prior_mode),
  paste0("NicheNet implementation backend: ", nichenet_backend$info$backend),
  paste0("NicheNet version: ", nichenet_backend$info$package_version),
  if (identical(nichenet_backend$info$backend, "frozen_official_source")) {
    paste0("Frozen official source commit: ", nichenet_backend$info$frozen_source_commit)
  } else {
    "Installed nichenetr namespace used; package version is recorded in run_manifest.json."
  },
  paste0("Active ligand-receptor universe: ", active_lr_label),
  "The official mouse v2 ligand-target matrix is used without modification in both modes.",
  if (prior_mode == "default") {
    "Default mode uses the official mouse v2 NicheNet ligand-receptor network."
  } else {
    "Matched mode uses the supplied frozen CellChat singleton ligand-receptor table only to constrain candidate LR pairs."
  },
  "NicheNet ligand activities are receiver-context ligand-to-target response prediction metrics.",
  "They are not a native sender-to-receiver cell-cell communication edge strength.",
  score_definition,
  "SPRING coordinates and other spatial coordinates are not read or used by this runner.",
  "Empty or non-analyzable contexts are retained with an explicit reason in nichenet_context_coverage.csv."
), method_notes_path)

session_path <- file.path(out_dir, "session_info.txt")
capture.output(sessionInfo(), file = session_path)
output_paths <- c(output_paths, method_notes = method_notes_path, session_info = session_path)

output_hash_table <- data.frame(
  output = names(output_paths),
  path = unname(output_paths),
  sha256 = vapply(unname(output_paths), sha256_file, character(1L)),
  stringsAsFactors = FALSE
)
output_hash_path <- file.path(out_dir, "output_file_sha256.csv")
write_csv_stable(output_hash_table, output_hash_path)
output_hash_index_sha256 <- sha256_file(output_hash_path)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0L) sub("^--file=", "", script_arg[[1L]]) else NA_character_
if (!is.na(script_path) && file.exists(script_path)) script_path <- normalizePath(script_path, mustWork = TRUE)

n_ok <- sum(coverage_all$status == "ok")
n_skipped <- nrow(coverage_all) - n_ok
run_status <- if (n_ok == 0L) "completed_with_no_analyzable_contexts" else if (n_skipped > 0L) {
  "completed_with_skipped_contexts"
} else {
  "complete"
}

manifest <- list(
  schema_version = "cytobridge.temporal_nichenet_reference.v1",
  status = run_status,
  created_at_utc = format(Sys.time(), tz = "UTC", format = "%Y-%m-%dT%H:%M:%SZ"),
  method = list(
    package = "nichenetr",
    package_version = nichenet_backend$info$package_version,
    implementation_backend = nichenet_backend$info$backend,
    frozen_source_commit_expected = NICHENET_FROZEN_COMMIT,
    frozen_source_commit_observed = nichenet_backend$info$frozen_source_commit,
    frozen_source_root = nichenet_backend$info$source_root,
    frozen_source_file_sha256 = nichenet_backend$info$source_file_sha256,
    frozen_source_description_sha256 = nichenet_backend$info$description_sha256,
    backend_loaded_namespaces = nichenet_backend$info$loaded_namespaces,
    backend_injected_unqualified_symbols =
      nichenet_backend$info$injected_unqualified_symbols,
    native_function = "predict_ligand_activities(single=TRUE)",
    target_link_function = "get_weighted_ligand_target_links",
    native_activity_columns = c("pearson", "auroc", "aupr", "aupr_corrected"),
    prior_mode = prior_mode,
    active_lr_label = active_lr_label,
    ligand_target_matrix_policy = "same_official_mouse_v2_matrix_in_default_and_matched_modes",
    custom_sender_support_is_native_nichenet_edge_strength = FALSE,
    custom_sender_support_definition = score_definition,
    spatial_coordinates_used = FALSE
  ),
  parameters = list(
    q_cutoff_if_selected_target_absent = q_cutoff,
    min_effect_if_selected_target_absent = min_effect,
    top_ligands_for_target_links = top_ligands,
    top_targets_per_ligand = top_targets,
    selected_target_column_present = selected_column_present,
    de_source_columns = de_source_columns
  ),
  inputs = list(
    input_dir = input_dir,
    receiver_de_genes = de_path,
    receiver_expressed_genes = receiver_expr_path,
    sender_expressed_genes_long = sender_expr_path,
    receiver_receptor_expression = receptor_expr_path,
    input_manifest = input_manifest_path,
    ligand_target_matrix = ltm_path,
    official_lr_network = official_lr_path,
    matched_lr_table = if (prior_mode == "matched") matched_lr_path else NULL,
    sha256 = c(asset_hashes, input_hashes),
    upstream_manifest = upstream_manifest
  ),
  prior_coverage = as.list(prior_coverage[1L, , drop = FALSE]),
  context_summary = list(
    n_contexts_total = nrow(coverage_all),
    n_contexts_ok = n_ok,
    n_contexts_skipped = n_skipped,
    status_counts = as.list(table(coverage_all$status, useNA = "ifany"))
  ),
  output_hashes = list(
    artifacts = split(output_hash_table[, c("path", "sha256"), drop = FALSE],
                      output_hash_table$output),
    index_path = output_hash_path,
    index_sha256 = output_hash_index_sha256
  ),
  runner = list(
    script = script_path,
    script_sha256 = if (!is.na(script_path) && file.exists(script_path)) sha256_file(script_path) else NULL,
    r_version = R.version.string,
    command = commandArgs(trailingOnly = FALSE)
  )
)

manifest_path <- file.path(out_dir, "run_manifest.json")
writeLines(jsonlite::toJSON(manifest, pretty = TRUE, auto_unbox = TRUE, na = "null",
                            null = "null", digits = 16), manifest_path)

message("NicheNet run status: ", run_status)
message("Prior mode: ", prior_mode)
message("Contexts OK/skipped: ", n_ok, "/", n_skipped)
message("Outputs: ", out_dir)
