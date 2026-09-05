#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(dplyr);library(tidyr);library(tibble);library(purrr);library(magrittr);library(ROCR);library(caTools);library(Hmisc)})
script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
src <- file.path(dirname(script_dir), 'nichenetr_official_R')
for (f in c('utils.R','supporting_functions.R','evaluate_model_target_prediction.R','evaluate_model_ligand_prediction.R','application_prediction.R')) source(file.path(src,f))
ltm <- readRDS(file.path(root,'external_data','ligand_target_matrix_nsga2r_final_mouse.rds'))
lr <- readRDS(file.path(root,'external_data','lr_network_mouse_21122021.rds'))
from <- if ('from' %in% names(lr)) 'from' else 'ligand'
to <- if ('to' %in% names(lr)) 'to' else 'receptor'
inp <- read.csv(file.path(root,'data','official_nichenet_window_inputs.csv'),check.names=FALSE)
split_genes <- function(x) { z <- strsplit(x,';',fixed=TRUE)[[1]]; z[nzchar(z)] }
all_activity <- list(); all_links <- list(); manifest <- list()
for(i in seq_len(nrow(inp))) {
  x <- inp[i,]; bg <- base::intersect(split_genes(x$background_genes),rownames(ltm)); gs <- base::intersect(split_genes(x$target_genes),bg)
  send <- split_genes(x$sender_genes); rec <- base::intersect(bg,lr[[to]])
  potential <- base::intersect(unique(lr[[from]][lr[[from]] %in% send & lr[[to]] %in% rec]),colnames(ltm))
  if(length(gs)<10 || length(potential)<1) stop(paste('invalid official input',x$window,length(gs),length(potential)))
  a <- predict_ligand_activities(geneset=gs,background_expressed_genes=bg,ligand_target_matrix=ltm,potential_ligands=potential,single=TRUE) %>%
    arrange(desc(aupr_corrected),desc(aupr),desc(auroc),test_ligand) %>% mutate(activity_rank=row_number(),window=x$window,baseline_model_time=x$baseline_model_time,response_model_time=x$response_model_time,n_background=length(bg),n_targets=length(gs))
  all_activity[[x$window]] <- a
  best <- head(a$test_ligand,5)
  links <- bind_rows(lapply(best,function(lig) get_weighted_ligand_target_links(ligand=lig,geneset=gs,ligand_target_matrix=ltm,n=250))) %>% mutate(window=x$window,baseline_model_time=x$baseline_model_time,response_model_time=x$response_model_time)
  all_links[[x$window]] <- links
  manifest[[x$window]] <- data.frame(window=x$window,n_background=length(bg),n_targets=length(gs),n_potential_ligands=length(potential))
}
write.csv(bind_rows(all_activity),file.path(root,'results','official_ligand_activity_all_50_windows.csv'),row.names=FALSE)
write.csv(bind_rows(all_links),file.path(root,'results','official_weighted_ligand_target_links_top5_all_50_windows.csv'),row.names=FALSE)
write.csv(bind_rows(manifest),file.path(root,'results','official_run_manifest.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(root,'provenance','session_info_official_nichenet.txt'))
