#!/usr/bin/env Rscript

# Use the server's ggplot2 3.3.5 stack, which is compatible with the archived
# dependencies required by the exact NicheNet mushroom function in this clone.
script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..', '..')); data_root <- dirname(final_root)
.libPaths(c(file.path(data_root, 'Rlib_nichenet_official_old'), .libPaths()))
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(magrittr)
  library(ggnewscale)
  library(shadowtext)
  library(cowplot)
})
.libPaths(c(file.path(data_root, 'Rlib_nichenet_official_old'),
            file.path(data_root, 'Rlib_nichenet_vis'), .libPaths()))
suppressPackageStartupMessages(library(ggforce))

root <- file.path(data_root, 'nichenet_official_reconstructed_20260827')
official_repo <- file.path(dirname(dirname(script_dir)), "nichenetr_official_R")
source(file.path(official_repo, "application_visualization.R"))

expr <- read.csv(file.path(root, "data", "reconstructed_expression_summary_three_states.csv"),
                 check.names = FALSE)
lr <- read.csv(file.path(root, "data", "candidate_ligand_receptor_pairs.csv"),
               check.names = FALSE) %>% distinct(ligand, receptor, .keep_all = TRUE)
activity <- read.csv(
  file.path(data_root, 'nichenet', 'new_interpolation', 'results', 'nichenet_adjacent_0p05', 'ligand_activity_all_50_windows.csv'),
  check.names = FALSE
)

minmax <- function(x) {
  r <- range(x, na.rm = TRUE)
  if (!all(is.finite(r)) || diff(r) == 0) return(rep(0.5, length(x)))
  (x - r[1]) / diff(r)
}

states <- data.frame(
  model_time = c(0.50, 1.30, 2.00),
  age = c(4.10, 9.36, 17.90),
  window = c("t0p45_to_t0p50", "t1p25_to_t1p30", "t1p95_to_t2p00"),
  label = c("t0p50", "t1p30", "t2p00")
)

for (i in seq_len(nrow(states))) {
  st <- states[i, ]
  expr_st <- expr %>% filter(abs(model_time - st$model_time) < 1e-8)
  lig_expr <- expr_st %>%
    select(sender = celltype, ligand = gene,
           avg_ligand = avg_reconstructed_log1p,
           pct_expressed_sender = pct_reconstructed_positive)
  rec_expr <- expr_st %>% filter(celltype == "Microglia") %>%
    select(receptor = gene, avg_receptor = avg_reconstructed_log1p,
           pct_expressed_receiver = pct_reconstructed_positive)
  act_st <- activity %>% filter(window == st$window) %>%
    select(ligand, aupr, activity_rank)

  tab <- lr %>%
    inner_join(lig_expr, by = "ligand") %>%
    inner_join(rec_expr, by = "receptor") %>%
    inner_join(act_st, by = "ligand") %>%
    mutate(
      scaled_avg_exprs_ligand = minmax(avg_ligand),
      scaled_avg_exprs_receptor = minmax(avg_receptor),
      activity_quantile = 1 - (activity_rank - 1) / max(activity_rank - 1),
      prioritization_score = rowMeans(cbind(
        activity_quantile, scaled_avg_exprs_ligand, scaled_avg_exprs_receptor,
        pct_expressed_sender, pct_expressed_receiver
      )),
      prioritization_rank = rank(-prioritization_score, ties.method = "first"),
      model_time = st$model_time,
      age_months = st$age,
      receiver = "Microglia"
    ) %>% arrange(prioritization_rank)

  write.csv(tab,
            file.path(root, "data", sprintf("descriptive_prioritization_table_%s.csv", st$label)),
            row.names = FALSE)

  # Exact NicheNet official plotting function. Rankings are hidden because this
  # is a descriptive, reconstructed-expression prioritization table rather than
  # an output from the replicate-aware generate_prioritization_tables workflow.
  p <- make_mushroom_plot(
    prioritization_table = tab,
    top_n = 18,
    show_rankings = FALSE,
    show_all_datapoints = FALSE,
    true_color_range = TRUE,
    size = "pct_expressed",
    color = "scaled_avg_exprs",
    ligand_fill_colors = c("#D9EAF7", "#2166AC"),
    receptor_fill_colors = c("#FAD9D4", "#B2182B")
  ) + ggtitle(sprintf("Ligand-receptor prioritization | model t=%.2f (%.2f months)",
                      st$model_time, st$age))

  ggsave(file.path(root, "figures", sprintf("02_nichenet_official_mushroom_%s.png", st$label)),
         p, width = 13.5, height = 8.0, dpi = 400)
  ggsave(file.path(root, "figures", sprintf("02_nichenet_official_mushroom_%s.pdf", st$label)),
         p, width = 13.5, height = 8.0, device = cairo_pdf)
}

writeLines(capture.output(sessionInfo()), file.path(root, "data", "session_info_mushroom.txt"))
