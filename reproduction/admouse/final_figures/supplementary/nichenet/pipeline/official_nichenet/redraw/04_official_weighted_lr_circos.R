#!/usr/bin/env Rscript

script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..', '..')); data_root <- dirname(final_root)
.libPaths(c(file.path(data_root, 'Rlib_nichenet_vis'), .libPaths()))
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(purrr)
  library(magrittr)
  library(circlize)
})

root <- file.path(data_root, 'nichenet_official_reconstructed_20260827')
official_repo <- file.path(dirname(dirname(script_dir)), "nichenetr_official_R")
source(file.path(official_repo, "application_prediction.R"))
source(file.path(official_repo, "application_visualization.R"))

weighted_networks <- readRDS(file.path(root, "external_data", "weighted_networks_nsga2r_final_mouse.rds"))
lr_pairs <- read.csv(file.path(root, "data", "candidate_ligand_receptor_pairs.csv"))
lr_network <- lr_pairs %>% transmute(from = ligand, to = receptor) %>% distinct()
expr <- read.csv(file.path(root, "data", "reconstructed_expression_summary_three_states.csv"))
activity <- read.csv(
  file.path(data_root, 'nichenet', 'new_interpolation', 'results', 'nichenet_adjacent_0p05', 'ligand_activity_all_50_windows.csv')
)

states <- data.frame(
  model_time = c(0.50, 1.30, 2.00),
  age = c(4.10, 9.36, 17.90),
  window = c("t0p45_to_t0p50", "t1p25_to_t1p30", "t1p95_to_t2p00"),
  label = c("t0p50", "t1p30", "t2p00")
)

for (i in seq_len(nrow(states))) {
  st <- states[i, ]
  top_ligands <- activity %>% filter(window == st$window) %>%
    arrange(activity_rank) %>% slice_head(n = 12) %>% pull(ligand)
  expressed_receptors <- expr %>%
    filter(abs(model_time - st$model_time) < 1e-8,
           celltype == "Microglia",
           gene %in% lr_pairs$receptor,
           pct_reconstructed_positive > 0.05) %>% pull(gene) %>% unique()

  # Official NicheNet weighted LR extraction.
  weighted_lr <- get_weighted_ligand_receptor_links(
    best_upstream_ligands = top_ligands,
    expressed_receptors = expressed_receptors,
    lr_network = lr_network,
    weighted_networks_lr_sig = weighted_networks$lr_sig
  )
  if (nrow(weighted_lr) == 0) stop(sprintf("No weighted LR links for %s", st$label))

  assignment <- read.csv(file.path(root, "data",
                                   sprintf("official_ligand_sender_assignment_%s.csv", st$label)))
  circos_links <- weighted_lr %>%
    transmute(ligand = from, target = to, weight = weight) %>%
    inner_join(assignment %>% select(ligand, ligand_type), by = "ligand") %>%
    mutate(target_type = "Microglia receptor") %>%
    distinct(ligand, target, ligand_type, target_type, .keep_all = TRUE)

  write.csv(circos_links,
            file.path(root, "data", sprintf("official_weighted_lr_links_%s.csv", st$label)),
            row.names = FALSE)

  ligand_types <- unique(circos_links$ligand_type)
  ligand_palette <- grDevices::hcl.colors(length(ligand_types), "Dark 3")
  names(ligand_palette) <- ligand_types
  receptor_palette <- c("Microglia receptor" = "#B2182B")

  # Official NicheNet Circos preparation and drawing functions.
  vis <- prepare_circos_visualization(
    circos_links,
    ligand_colors = ligand_palette,
    target_colors = receptor_palette
  )
  # make_circos_plot expects the cutoff attribute normally added by
  # get_ligand_target_links_oi(); weighted LR extraction has no cutoff helper.
  # Set it to the minimum retained weight so every extracted official LR link
  # is visible, without changing any weight.
  attr(vis$links_circle, "cutoff_include_all_ligands") <- min(vis$links_circle$weight)

  draw_one <- function(device, filename) {
    if (device == "png") {
      png(filename, width = 2800, height = 2800, res = 300)
    } else {
      cairo_pdf(filename, width = 9.3, height = 9.3)
    }
    circos.clear()
    make_circos_plot(vis, transparency = TRUE,
                     args.circos.text = list(cex = 0.65))
    title(sprintf("Weighted ligand-receptor network\nmodel t=%.2f (%.2f months)",
                  st$model_time, st$age), cex.main = 1.1)
    circos.clear()
    dev.off()
  }
  draw_one("png", file.path(root, "figures", sprintf("03_nichenet_official_weighted_lr_circos_%s.png", st$label)))
  draw_one("pdf", file.path(root, "figures", sprintf("03_nichenet_official_weighted_lr_circos_%s.pdf", st$label)))
}

writeLines(capture.output(sessionInfo()), file.path(root, "data", "session_info_weighted_lr_circos.txt"))
