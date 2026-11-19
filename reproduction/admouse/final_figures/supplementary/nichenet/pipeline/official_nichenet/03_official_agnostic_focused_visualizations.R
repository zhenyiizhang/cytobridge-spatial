#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(dplyr);library(tidyr);library(tibble);library(purrr);library(magrittr);library(ggplot2);library(cowplot);library(ROCR);library(caTools);library(Hmisc)})
script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
src <- file.path(dirname(script_dir), 'nichenetr_official_R')
for (f in c('utils.R','supporting_functions.R','evaluate_model_target_prediction.R','evaluate_model_ligand_prediction.R','application_prediction.R','application_visualization.R')) source(file.path(src,f))
ltm<-readRDS(file.path(root,'external_data','ligand_target_matrix_nsga2r_final_mouse.rds')); lr<-readRDS(file.path(root,'external_data','lr_network_mouse_21122021.rds'))
inp<-read.csv(file.path(root,'data','official_nichenet_window_inputs.csv')); focused<-read.csv(file.path(root,'results','official_ligand_activity_all_50_windows.csv'))
sp<-function(x){z<-strsplit(x,';',fixed=TRUE)[[1]];z[nzchar(z)]}
reps<-c('t0p45_to_t0p50','t1p25_to_t1p30','t1p95_to_t2p00'); labels<-c('4.10 months','9.36 months','17.90 months')
ag_all<-list(); hist_all<-list()
for(i in seq_along(reps)){
 x<-inp[inp$window==reps[i],]; bg<-base::intersect(sp(x$background_genes),rownames(ltm)); gs<-base::intersect(sp(x$target_genes),bg)
 rec<-base::intersect(bg,lr$to); agnostic<-base::intersect(unique(lr$from[lr$to%in%rec]),colnames(ltm))
 a<-predict_ligand_activities(gs,bg,ltm,agnostic,single=TRUE) %>% arrange(desc(aupr_corrected),desc(aupr),desc(auroc),test_ligand) %>% mutate(rank=row_number(),window=reps[i],label=labels[i])
 f<-focused %>% filter(window==reps[i]) %>% select(test_ligand) %>% pull()
 view_ligands <- union(head(a$test_ligand,20), f)
 a_view <- a %>% filter(test_ligand %in% view_ligands)
 f_view <- intersect(f, a_view$test_ligand)
 p<-make_line_plot(a_view, f_view, ranking_range=c(1,min(30,nrow(a_view))))+ggtitle(paste('Official NicheNet ranking |',labels[i]))
 ggsave(file.path(root,'figures',paste0('06_official_agnostic_vs_focused_',reps[i],'.png')),p,width=7,height=6,dpi=400)
 ggsave(file.path(root,'figures',paste0('06_official_agnostic_vs_focused_',reps[i],'.pdf')),p,width=7,height=6,device=cairo_pdf)
 ag_all[[reps[i]]]<-a
 hist_all[[reps[i]]]<-bind_rows(a%>%mutate(mode='Sender-agnostic'),focused%>%filter(window==reps[i])%>%transmute(test_ligand,aupr_corrected,mode='Sender-focused'))%>%mutate(window_label=labels[i])
}
write.csv(bind_rows(ag_all),file.path(root,'results','official_sender_agnostic_activity_three_windows.csv'),row.names=FALSE)
h<-bind_rows(hist_all)
# Same histogram construction as the official NicheNet seurat_steps vignette; values are official activity outputs.
p_hist<-ggplot(h,aes(aupr_corrected,fill=mode))+geom_histogram(bins=28,position='identity',alpha=.58)+facet_wrap(~window_label,nrow=1)+labs(title='Official NicheNet ligand activity distributions',x='Corrected AUPR',y='Number of ligands',fill=NULL)+theme_bw(base_size=11)+theme(plot.title=element_text(face='bold'),panel.grid.minor=element_blank())
ggsave(file.path(root,'figures','07_official_ligand_activity_distribution_three_windows.png'),p_hist,width=12.5,height=4,dpi=400)
ggsave(file.path(root,'figures','07_official_ligand_activity_distribution_three_windows.pdf'),p_hist,width=12.5,height=4,device=cairo_pdf)
write.csv(h,file.path(root,'results','official_ligand_activity_distribution_data_three_windows.csv'),row.names=FALSE)
