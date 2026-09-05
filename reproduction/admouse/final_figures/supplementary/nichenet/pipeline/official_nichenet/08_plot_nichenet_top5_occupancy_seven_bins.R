#!/usr/bin/env Rscript
# Seven-bin version of 09d; read-only summary of existing official activity.
suppressPackageStartupMessages(library(ggplot2))
script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
fig <- file.path(root,'figures','si_network_preview_20260827')
dat <- file.path(root,'data','si_network_preview_20260827')
a <- read.csv(file.path(root,'results','official_ligand_activity_all_50_windows.csv'),stringsAsFactors=FALSE)
a <- a[a$response_model_time <= 2+1e-9,]
breaks <- c(0,.3,.6,.9,1.2,1.5,1.8,2.000001)
labels <- c('0–0.3','0.3–0.6','0.6–0.9','0.9–1.2','1.2–1.5','1.5–1.8','1.8–2.0')
a$time_bin <- cut(a$response_model_time,breaks=breaks,labels=labels,include.lowest=TRUE,right=TRUE)
occ <- aggregate(activity_rank~time_bin+test_ligand,a,FUN=function(x)mean(x<=5));names(occ)[3] <- 'top5_occupancy'
sc <- aggregate(aupr_corrected~time_bin+test_ligand,a,mean);occ <- merge(occ,sc,by=c('time_bin','test_ligand'))
rnk <- aggregate(top5_occupancy~test_ligand,occ,max);keep <- head(rnk[order(-rnk$top5_occupancy,rnk$test_ligand),'test_ligand'],12)
occ <- occ[occ$test_ligand%in%keep,];occ$test_ligand <- factor(occ$test_ligand,levels=rev(keep));occ$time_bin <- factor(occ$time_bin,levels=labels)
p <- ggplot(occ,aes(time_bin,test_ligand))+geom_point(aes(size=top5_occupancy,colour=aupr_corrected))+scale_size_area(max_size=13,breaks=c(.1,.3,.5,.8),labels=scales::percent_format(accuracy=1))+scale_colour_gradient(low='#D9EAF7',high='#9B1B30')+theme_classic(base_size=12)+theme(axis.title.y=element_blank(),axis.text.x=element_text(angle=35,hjust=1))+labs(title='Dense-window persistence of NicheNet candidate ligands',subtitle='Seven contiguous model-time bins; point size: fraction of windows ranked in top five; colour: mean corrected AUPR',x='Model-time interval',colour='Mean corrected AUPR',size='Top-five occupancy')
stem <- '09d2_top5_ligand_occupancy_seven_model_time_bins'
ggsave(file.path(fig,paste0(stem,'.png')),p,width=10.2,height=5.8,dpi=300);ggsave(file.path(fig,paste0(stem,'.pdf')),p,width=10.2,height=5.8)
write.csv(occ,file.path(dat,paste0(stem,'.csv')),row.names=FALSE)
