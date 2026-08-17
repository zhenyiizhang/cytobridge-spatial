#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(magrittr)
  library(ROCR)
  library(tibble)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) {
  where <- match(flag, args)
  if (is.na(where) || where == length(args)) stop(paste("missing", flag))
  args[[where + 1]]
}

source_root <- normalizePath(get_arg("--nichenetr-source"), mustWork = TRUE)
matrix_path <- normalizePath(get_arg("--ligand-target-matrix"), mustWork = TRUE)
gene_sets_path <- normalizePath(get_arg("--receiver-gene-sets"), mustWork = TRUE)
candidates_path <- normalizePath(get_arg("--lr-candidates"), mustWork = TRUE)
output_dir <- get_arg("--output-dir")
if (dir.exists(output_dir)) stop("output directory already exists")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (script in sort(list.files(file.path(source_root, "R"), pattern = "\\.R$", full.names = TRUE))) {
  sys.source(script, envir = .GlobalEnv)
}

ligand_target_matrix <- readRDS(matrix_path)
gene_sets <- read.csv(gene_sets_path, stringsAsFactors = FALSE, check.names = FALSE)
candidates <- read.csv(candidates_path, stringsAsFactors = FALSE, check.names = FALSE)
required_gene_sets <- c("dataset", "receiver", "gene", "is_response")
required_candidates <- c("dataset", "sender", "receiver", "ligand", "receptor")
if (!all(required_gene_sets %in% colnames(gene_sets))) stop("receiver gene-set schema mismatch")
if (!all(required_candidates %in% colnames(candidates))) stop("candidate schema mismatch")
gene_sets$is_response <- tolower(as.character(gene_sets$is_response)) == "true"
dataset_name <- unique(gene_sets$dataset)
candidate_dataset <- unique(candidates$dataset)
if (length(dataset_name) != 1) stop("receiver gene sets must contain one dataset")
if (length(candidate_dataset) != 1 || candidate_dataset != dataset_name) {
  stop("candidate and receiver gene-set datasets must match exactly")
}

activity_outputs <- list()
target_outputs <- list()
receiver_status_outputs <- list()
for (receiver_oi in sort(unique(gene_sets$receiver))) {
  receiver_sets <- gene_sets %>% filter(receiver == receiver_oi)
  geneset_oi <- receiver_sets %>% filter(is_response) %>% pull(gene) %>% unique()
  background <- receiver_sets %>% pull(gene) %>% unique()
  potential_ligands <- candidates %>%
    filter(receiver == receiver_oi) %>%
    pull(ligand) %>% unique()
  geneset_oi <- intersect(geneset_oi, rownames(ligand_target_matrix))
  background <- intersect(background, rownames(ligand_target_matrix))
  potential_ligands <- intersect(potential_ligands, colnames(ligand_target_matrix))
  if (length(geneset_oi) < 10) stop(paste(receiver_oi, "has fewer than ten response genes"))
  if (length(background) < 20) stop(paste(receiver_oi, "has fewer than twenty background genes"))
  if (length(potential_ligands) == 0) {
    receiver_status_outputs[[receiver_oi]] <- tibble(
      dataset = dataset_name,
      receiver = receiver_oi,
      status = "skipped_no_potential_ligands",
      reason = "no candidate ligand is represented in the frozen NicheNet ligand-target matrix",
      n_response_genes = length(geneset_oi),
      n_background_genes = length(background),
      n_potential_ligands = 0L
    )
    next
  }

  activity <- predict_ligand_activities(
    geneset = geneset_oi,
    background_expressed_genes = background,
    ligand_target_matrix = ligand_target_matrix,
    potential_ligands = potential_ligands
  ) %>%
    rename(ligand = test_ligand) %>%
    mutate(receiver = receiver_oi)
  activity_outputs[[receiver_oi]] <- activity

  top_ligands <- activity %>% arrange(desc(aupr_corrected)) %>% head(30) %>% pull(ligand)
  links <- lapply(
    top_ligands,
    get_weighted_ligand_target_links,
    geneset = geneset_oi,
    ligand_target_matrix = ligand_target_matrix,
    n = 250
  ) %>% bind_rows() %>%
    filter(!is.na(target), !is.na(weight)) %>%
    transmute(
      ligand = as.character(ligand),
      target = as.character(target),
      weight = as.numeric(weight),
      receiver = receiver_oi
    )
  target_outputs[[receiver_oi]] <- links
  receiver_status_outputs[[receiver_oi]] <- tibble(
    dataset = dataset_name,
    receiver = receiver_oi,
    status = "complete",
    reason = "",
    n_response_genes = length(geneset_oi),
    n_background_genes = length(background),
    n_potential_ligands = length(potential_ligands)
  )
}

if (length(activity_outputs) == 0) {
  stop("no receiver has a candidate ligand represented in the frozen NicheNet ligand-target matrix")
}
activities <- bind_rows(activity_outputs) %>%
  mutate(dataset = dataset_name) %>%
  select(dataset, receiver, ligand, auroc, aupr, aupr_corrected, pearson) %>%
  arrange(receiver, desc(aupr_corrected), ligand)
targets <- bind_rows(target_outputs) %>%
  mutate(dataset = dataset_name) %>%
  select(dataset, receiver, ligand, target, weight) %>%
  arrange(receiver, ligand, desc(weight), target)
receiver_status <- bind_rows(receiver_status_outputs) %>%
  arrange(receiver)

write.csv(activities, file.path(output_dir, "ligand_activities.csv"), row.names = FALSE, quote = TRUE)
write.csv(targets, file.path(output_dir, "ligand_target_links.csv"), row.names = FALSE, quote = TRUE)
write.csv(receiver_status, file.path(output_dir, "receiver_status.csv"), row.names = FALSE, quote = TRUE)
session <- capture.output(sessionInfo())
writeLines(session, file.path(output_dir, "R_sessionInfo.txt"))
