#!/usr/bin/env Rscript
# Continuous LR Circos using the same official NicheNet functions as the
# original 03 figure, but current official NicheNet activity outputs.
script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..')); data_root <- dirname(final_root)
.libPaths(c(file.path(data_root, 'Rlib_nichenet_vis'), .libPaths()))
suppressPackageStartupMessages({library(dplyr);library(tibble);library(purrr);library(magrittr);library(circlize)})
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
src <- file.path(dirname(script_dir), 'nichenetr_official_R')
source(file.path(src,'application_prediction.R'));source(file.path(src,'application_visualization.R'))
out <- file.path(root,'figures','official_weighted_lr_circos_continuous_20260827'); dat <- file.path(root,'data','official_weighted_lr_circos_continuous_20260827')
dir.create(out,recursive=TRUE,showWarnings=FALSE);dir.create(dat,recursive=TRUE,showWarnings=FALSE)
weighted <- readRDS(file.path(root,'external_data','weighted_networks_nsga2r_final_mouse.rds'))
pairs <- read.csv(file.path(root,'data','candidate_ligand_receptor_pairs.csv'))
lr <- pairs %>% transmute(from=ligand,to=receptor) %>% distinct()
expr <- read.csv(file.path(root,'data','model_expression_summary_51_states.csv'))
act <- read.csv(file.path(root,'results','official_ligand_activity_all_50_windows.csv'))
states <- data.frame(model_time=c(.50,.90,1.30,1.65,2.00),age=c(4.10,5.38,9.36,13.63,17.90),window=c('t0p45_to_t0p50','t0p85_to_t0p90','t1p25_to_t1p30','t1p60_to_t1p65','t1p95_to_t2p00'),label=c('t0p50','t0p90','t1p30','t1p65','t2p00'))
sender_type <- function(time, ligands) {z<-expr %>% filter(abs(model_time-time)<1e-8,gene %in% ligands,pct_positive>=.10) %>% arrange(gene,desc(pct_positive),desc(mean_log1p)) %>% group_by(gene) %>% slice(1) %>% ungroup() %>% transmute(ligand=gene,ligand_type=celltype);z}
all_links <- list()
for(i in seq_len(nrow(states))) {
 st <- states[i,]; x <- act %>% filter(window==st$window) %>% arrange(activity_rank) %>% slice_head(n=12); top <- x$test_ligand
 receptors <- expr %>% filter(abs(model_time-st$model_time)<1e-8,celltype=='Microglia',gene %in% pairs$receptor,pct_positive>.05) %>% pull(gene) %>% unique()
 wl <- get_weighted_ligand_receptor_links(best_upstream_ligands=top,expressed_receptors=receptors,lr_network=lr,weighted_networks_lr_sig=weighted$lr_sig)
 if(nrow(wl)==0) stop(paste('No weighted links:',st$label))
 links <- wl %>% transmute(ligand=from,target=to,weight=weight) %>% inner_join(sender_type(st$model_time,top),by='ligand') %>% mutate(target_type='Microglia receptor') %>% distinct(ligand,target,ligand_type,target_type,.keep_all=TRUE)
 if(nrow(links)==0) stop(paste('No assigned links:',st$label))
 write.csv(links,file.path(dat,paste0('official_weighted_lr_links_',st$label,'.csv')),row.names=FALSE)
 lig_types <- unique(links$ligand_type); lig_pal <- grDevices::hcl.colors(length(lig_types),'Dark 3'); names(lig_pal)<-lig_types
 vis <- prepare_circos_visualization(links,ligand_colors=lig_pal,target_colors=c('Microglia receptor'='#B2182B'))
 attr(vis$links_circle,'cutoff_include_all_ligands') <- min(vis$links_circle$weight)
 draw <- function(type) {fn<-file.path(out,paste0('official_weighted_lr_circos_',st$label,'.',type));if(type=='png') png(fn,width=2800,height=2800,res=300) else cairo_pdf(fn,width=9.3,height=9.3);circos.clear();make_circos_plot(vis,transparency=TRUE,args.circos.text=list(cex=.65));title(sprintf('Weighted ligand–receptor network\nmodel t=%.2f (%.2f months)',st$model_time,st$age),cex.main=1.1);circos.clear();dev.off()}
 draw('png');draw('pdf');links$model_time<-st$model_time;links$age_months<-st$age;links$window<-st$window;all_links[[st$label]]<-links
}
write.csv(bind_rows(all_links),file.path(dat,'all_continuous_official_weighted_lr_links.csv'),row.names=FALSE)
writeLines(c('All plots use official NicheNet get_weighted_ligand_receptor_links(), prepare_circos_visualization(), and make_circos_plot().','Top ligands come from current latest_nichenet official predict_ligand_activities() output.','Edge weights are the NicheNet weighted LR prior, not CytoBridge attention or measured binding strength.'),file.path(out,'README.txt'))
