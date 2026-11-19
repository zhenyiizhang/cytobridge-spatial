#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(ggplot2)
  library(dplyr)
  library(purrr)
})

script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet_official_reconstructed_20260827')
official_source <- file.path(dirname(dirname(script_dir)), 'nichenetr_official_R', 'application_visualization.R')
source(official_source)
dir.create(file.path(root, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(root, "data"), recursive = TRUE, showWarnings = FALSE)

states <- data.frame(
  model_time = c(0.50, 1.30, 2.00),
  age = c(4.10, 9.36, 17.90),
  label = c("t0p50", "t1p30", "t2p00")
)
ligands <- read.csv(file.path(root, "data", "candidate_ligand_receptor_pairs.csv"),
                    check.names = FALSE)$ligand |> unique()

for (i in seq_len(nrow(states))) {
  st <- states[i, ]
  in_file <- sprintf("%s/intermediate/reconstructed_selected_genes_t%.2f.csv.gz", root, st$model_time)
  dat <- read.csv(in_file, check.names = FALSE)
  rownames(dat) <- dat$cell
  meta <- data.frame(major_annotation = dat$major_annotation, row.names = dat$cell)
  expr <- as.matrix(dat[, ligands, drop = FALSE])
  storage.mode(expr) <- "double"
  mat <- Matrix(t(expr), sparse = TRUE)

  # The reconstructed values are already in the processed log1p expression space.
  obj <- CreateSeuratObject(counts = mat, meta.data = meta, min.cells = 0, min.features = 0)
  obj <- SetAssayData(obj, assay = "RNA", slot = "data", new.data = mat)

  # Official Seurat visualization used by NicheNet sender-expression workflows.
  p <- DotPlot(obj, features = ligands, group.by = "major_annotation", scale = TRUE,
               cols = c("#2166AC", "#F7F7F7", "#B2182B"), dot.scale = 7) +
    RotatedAxis() +
    labs(
      title = sprintf("Sender ligand expression | model t=%.2f (%.2f months)", st$model_time, st$age),
      subtitle = "Official Seurat::DotPlot on inverse-PCA reconstructed log1p expression",
      x = "Candidate ligand", y = "Cell type", color = "Scaled average\nexpression",
      size = "Percent\npositive"
    ) +
    theme_bw(base_size = 11) +
    theme(panel.grid.major = element_line(color = "grey92", size = 0.25),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold"))

  png_file <- file.path(root, "figures", sprintf("01_sender_expression_dotplot_%s.png", st$label))
  pdf_file <- file.path(root, "figures", sprintf("01_sender_expression_dotplot_%s.pdf", st$label))
  ggsave(png_file, p, width = 12.5, height = 5.8, dpi = 400)
  ggsave(pdf_file, p, width = 12.5, height = 5.8, device = cairo_pdf)

  dot_data <- p$data
  dot_data$model_time <- st$model_time
  dot_data$age_months <- st$age
  write.csv(dot_data,
            file.path(root, "data", sprintf("official_seurat_dotplot_data_%s.csv", st$label)),
            row.names = FALSE)

  # Official NicheNet sender assignment helper, later reused by Circos.
  assignment <- assign_ligands_to_celltype(
    seuratObj = obj, ligands = ligands, celltype_col = "major_annotation", slot = "data"
  )
  assignment$model_time <- st$model_time
  assignment$age_months <- st$age
  write.csv(assignment,
            file.path(root, "data", sprintf("official_ligand_sender_assignment_%s.csv", st$label)),
            row.names = FALSE)
}

writeLines(capture.output(sessionInfo()), file.path(root, "data", "session_info_seurat_dotplot.txt"))
