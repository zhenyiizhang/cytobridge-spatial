#!/usr/bin/env Rscript

script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..', '..')); data_root <- dirname(final_root)
.libPaths(c(file.path(data_root, 'Rlib_nichenet_vis'), .libPaths()))
suppressPackageStartupMessages({library(dplyr); library(tibble); library(purrr); library(magrittr); library(igraph)})
root <- file.path(data_root, 'nichenet_official_reconstructed_20260827')
repo <- file.path(dirname(dirname(script_dir)), "nichenetr_official_R")
source(file.path(repo, "supporting_functions.R"))
source(file.path(repo, "application_visualization.R"))
weighted <- readRDS(file.path(root, "external_data", "weighted_networks_nsga2r_final_mouse.rds"))
ltf <- readRDS(file.path(root, "external_data", "ligand_tf_matrix_nsga2r_final_mouse.rds"))
axes <- read.csv(file.path(root, "data", "signaling_path_selected_axes_t2p00.csv"))

for (lig in unique(axes$ligand)) {
  one <- axes %>% filter(ligand == lig)
  rec <- unique(one$receptor)
  tar <- unique(one$target)
  path <- get_ligand_signaling_path_with_receptor(
    ligand_tf_matrix = ltf, ligands_all = lig, receptors_all = rec,
    targets_all = tar, top_n_regulators = 2, weighted_networks = weighted,
    ligands_position = "cols"
  )
  stem <- sprintf("official_signaling_path_%s_%s_t2p00", tolower(lig), tolower(rec))
  write.csv(path$sig, file.path(root, "data", paste0(stem, "_sig_edges.csv")), row.names = FALSE)
  write.csv(path$gr, file.path(root, "data", paste0(stem, "_gr_edges.csv")), row.names = FALSE)
  saveRDS(path, file.path(root, "data", paste0(stem, ".rds")))

  edges <- bind_rows(path$sig %>% mutate(edge_type = "signaling"),
                     path$gr %>% mutate(edge_type = "gene regulation")) %>%
    group_by(from, to, edge_type) %>% summarise(weight = sum(weight), .groups = "drop")
  g <- graph_from_data_frame(edges, directed = TRUE)
  nm <- V(g)$name
  tp <- ifelse(nm == lig, "Ligand", ifelse(nm == rec, "Receptor",
        ifelse(nm %in% tar, "Target", "Signaling / TF")))
  vc <- c("Ligand"="#2166AC", "Receptor"="#1B9E77", "Signaling / TF"="#737373", "Target"="#D6604D")[tp]
  ec <- ifelse(E(g)$edge_type == "signaling", "#4C78A8AA", "#E17C47AA")
  ew <- E(g)$weight
  wd <- 0.6 + 3.0 * (ew - min(ew)) / max(diff(range(ew)), .Machine$double.eps)
  lay <- layout_with_sugiyama(g)$layout

  draw <- function(kind, file) {
    if (kind == "png") png(file, width = 2700, height = 2200, res = 350, bg = "white")
    else cairo_pdf(file, width = 7.7, height = 6.3)
    par(mar = c(1,1,4,1))
    plot(g, layout = lay, vertex.color = vc, vertex.frame.color = "white",
         vertex.size = 20, vertex.label.cex = 0.73, vertex.label.color = "black",
         edge.color = ec, edge.width = wd, edge.arrow.size = 0.38,
         main = sprintf("%s–%s signaling paths | model t=2.00", lig, rec))
    legend("topleft", legend = c("Ligand","Receptor","Signaling / TF","Target"),
           col = c("#2166AC","#1B9E77","#737373","#D6604D"), pch=19,
           pt.cex=1.5, bty="n", cex=.75)
    dev.off()
  }
  draw("png", file.path(root, "figures", paste0("05b_", stem, ".png")))
  draw("pdf", file.path(root, "figures", paste0("05b_", stem, ".pdf")))
}
writeLines(capture.output(sessionInfo()), file.path(root, "data", "session_info_signaling_paths_by_axis.txt"))
