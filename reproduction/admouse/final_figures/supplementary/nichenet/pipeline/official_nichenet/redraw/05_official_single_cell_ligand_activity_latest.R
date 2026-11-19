#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(purrr)
  library(magrittr)
  library(ggplot2)
  library(ROCR)
  library(caTools)
  library(Hmisc)
  library(parallel)
})

script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
repo_r <- file.path(dirname(dirname(script_dir)), 'nichenetr_official_R')
for (f in c("utils.R", "supporting_functions.R", "evaluate_model_target_prediction.R",
            "evaluate_model_ligand_prediction.R", "application_prediction.R")) {
  source(file.path(repo_r, f))
}

x <- read.csv(gzfile(file.path(root, "intermediate",
                               "single_cell_microglia_scaled_expression_t1p00.csv.gz")),
              check.names = FALSE)
cell_ids <- x$cell
expression_scaled <- as.matrix(x[, -1, drop = FALSE])
rownames(expression_scaled) <- cell_ids

ligand_target_matrix <- readRDS(file.path(root, "external_data",
                                           "ligand_tf_matrix_nsga2r_final_mouse.rds"))
potential_ligands <- read.csv(file.path(root, "data", "candidate_ligand_receptor_pairs.csv"))$ligand %>% unique()
genes_use <- intersect(rownames(ligand_target_matrix), colnames(expression_scaled))
ligands_use <- intersect(colnames(ligand_target_matrix), potential_ligands)
ligand_target_use <- ligand_target_matrix[genes_use, ligands_use, drop = FALSE]
expression_use <- expression_scaled[, genes_use, drop = FALSE]

# Invoke the exact official NicheNet function in four independent chunks. The
# split changes execution scheduling only; every cell is evaluated by the same
# official predict_single_cell_ligand_activities() implementation.
chunks <- split(cell_ids, cut(seq_along(cell_ids), breaks = 4, labels = FALSE))
activity_parts <- mclapply(chunks, function(ids) {
  predict_single_cell_ligand_activities(
    cell_ids = ids,
    expression_scaled = expression_use,
    ligand_target_matrix = ligand_target_use,
    potential_ligands = ligands_use,
    single = TRUE
  )
}, mc.cores = 4)
activity <- bind_rows(activity_parts) %>% rename(cell = setting, ligand = test_ligand)
write.csv(activity, file.path(root, "data", "official_single_cell_ligand_activity_raw_t1p00.csv"),
          row.names = FALSE)

# Exact NicheNet modified-z-score utility, applied cell-wise to AUPR.
activity_norm <- activity %>% group_by(cell) %>%
  mutate(normalized_aupr = scaling_modified_zscore(aupr)) %>% ungroup()
write.csv(activity_norm,
          file.path(root, "data", "official_single_cell_ligand_activity_normalized_t1p00.csv"),
          row.names = FALSE)

modules <- read.csv(file.path(root, "data", "single_cell_microglia_module_scores_t1p00.csv"))
joined <- activity_norm %>% inner_join(modules %>% select(cell, module, module_score), by = "cell")
cors <- joined %>% group_by(ligand, module) %>%
  summarise(n_cells = n(), pearson_r = cor(normalized_aupr, module_score),
            spearman_r = cor(normalized_aupr, module_score, method = "spearman"), .groups = "drop")
write.csv(cors, file.path(root, "data", "single_cell_activity_module_correlations_t1p00.csv"),
          row.names = FALSE)

top_axes <- cors %>% group_by(module) %>% slice_max(abs(pearson_r), n = 1, with_ties = FALSE) %>%
  ungroup() %>% select(ligand, module, pearson_r)
plot_df <- joined %>% inner_join(top_axes, by = c("ligand", "module")) %>%
  mutate(panel = sprintf("%s activity vs %s\nr = %.2f", ligand, module, pearson_r))

# NicheNet provides the activity predictor but no dedicated scatter wrapper;
# this follows the official single-cell vignette's ggplot + linear smooth form.
p <- ggplot(plot_df, aes(normalized_aupr, module_score)) +
  geom_point(size = 1.05, alpha = 0.46, color = "#35608D") +
  geom_smooth(method = "lm", se = TRUE, color = "#B2182B", fill = "#E8B6B6",
              size = 0.8) +
  facet_wrap(~panel, scales = "free_y", nrow = 1) +
  labs(title = "Single-cell NicheNet ligand activity and Microglia state",
       subtitle = "Official predict_single_cell_ligand_activities() | 500 t=1.00 Microglia, seed 42",
       x = "Cell-wise ligand activity (modified-z AUPR)", y = "Reconstructed module score") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank())

ggsave(file.path(root, "figures", "04_official_single_cell_activity_scatter_t1p00.png"),
       p, width = 13.2, height = 4.4, dpi = 400)
ggsave(file.path(root, "figures", "04_official_single_cell_activity_scatter_t1p00.pdf"),
       p, width = 13.2, height = 4.4, device = cairo_pdf)

writeLines(capture.output(sessionInfo()), file.path(root, "data", "session_info_single_cell_activity.txt"))
