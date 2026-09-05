#!/usr/bin/env Rscript
# Custom SI preview plots from existing official NicheNet scores; no rescoring.
suppressPackageStartupMessages({library(ggplot2); library(grid); library(png)})
script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
fig <- file.path(root,'figures','si_network_preview_20260827'); dat <- file.path(root,'data','si_network_preview_20260827')
dir.create(fig,recursive=TRUE,showWarnings=FALSE); dir.create(dat,recursive=TRUE,showWarnings=FALSE)
a <- read.csv(file.path(root,'results','official_ligand_activity_all_50_windows.csv'),stringsAsFactors=FALSE)
l <- read.csv(file.path(root,'results','official_weighted_ligand_target_links_top5_all_50_windows.csv'),stringsAsFactors=FALSE)
a <- a[a$response_model_time<=2+1e-9,]; l <- l[l$response_model_time<=2+1e-9,]
stage <- data.frame(stage=c('Early','Intermediate','Late'),window=c('t0p45_to_t0p50','t1p25_to_t1p30','t1p95_to_t2p00'),time=c('4.0–4.1 months','9.1–9.4 months','17.6–17.9 months'),assignment=c('t0p50','t1p30','t2p00'),stringsAsFactors=FALSE)
pal <- c(C1qa='#2166AC',C1qb='#4D4D4D',Col1a1='#B2182B',Spp1='#00A6A6',Dcn='#1B7837',Igf2='#756BB1',Sema3a='#E08214',Pecam1='#7B3294',Cdh4='#C05A8A',Sema3e='#8C564B',Nts='#E17C05')
col_for <- function(x) ifelse(x %in% names(pal),pal[x],'#888888')
bez <- function(p0,p1,p2,p3,n=80) {t<-seq(0,1,length.out=n);cbind((1-t)^3*p0[1]+3*(1-t)^2*t*p1[1]+3*(1-t)*t^2*p2[1]+t^3*p3[1],(1-t)^3*p0[2]+3*(1-t)^2*t*p1[2]+3*(1-t)*t^2*p2[2]+t^3*p3[2])}
links_for <- function(w,nlig=4,ntarget=4) {x<-a[a$window==w,]; top<-head(x[order(x$activity_rank),'test_ligand'],nlig); z<-l[l$window==w & l$ligand%in%top,];do.call(rbind,lapply(top,function(q)head(z[z$ligand==q,][order(-z[z$ligand==q,'weight']),],ntarget)))}
save_base <- function(stem,w,h,fun) {png(file.path(fig,paste0(stem,'.png')),width=w*300,height=h*300,res=300);fun();dev.off();pdf(file.path(fig,paste0(stem,'.pdf')),width=w,height=h);fun();dev.off()}

# 09a: chord-like ligand–target Circos based only on official weighted links.
draw_circos <- function(){par(mfrow=c(1,3),mar=c(1,1,3,1),xpd=NA);for(i in 1:nrow(stage)){d<-links_for(stage$window[i]);lig<-unique(d$ligand);tar<-unique(d$target);nodes<-c(lig,tar);ang<-seq(pi/2,pi/2+2*pi,length.out=length(nodes)+1)[-length(nodes)-1];xy<-cbind(cos(ang),sin(ang));rownames(xy)<-nodes;plot(0,0,type='n',xlim=c(-1.55,1.55),ylim=c(-1.42,1.42),axes=FALSE,xlab='',ylab='',main=paste0(stage$stage[i],'\n',stage$time[i]));for(j in 1:nrow(d)){r<-d[j,];q<-bez(xy[r$ligand,],c(.25,xy[r$ligand,2]),c(-.25,xy[r$target,2]),xy[r$target,]);lines(q[,1],q[,2],col=adjustcolor(col_for(r$ligand),.46),lwd=.6+4*r$weight/max(d$weight))};points(xy[lig,1],xy[lig,2],pch=21,bg=col_for(lig),cex=1.35);points(xy[tar,1],xy[tar,2],pch=21,bg='#E2E2E2',cex=1.05);for(n in nodes){x<-xy[n,1];y<-xy[n,2];text(1.17*x,1.17*y,n,cex=.72,adj=ifelse(x>=0,c(0,.5),c(1,.5)),font=ifelse(n%in%lig,2,1))}};mtext('NicheNet ligand–target prior networks | colored = ligand; grey = Microglia transition target; width = prior weight',1,outer=TRUE,line=-1.2,cex=.8)}
save_base('09a_nichenet_ligand_target_circos_three_stages',16,6,draw_circos)
circos_data<-do.call(rbind,lapply(1:nrow(stage),function(i){z<-links_for(stage$window[i]);z$stage<-stage$stage[i];z$time<-stage$time[i];z}));write.csv(circos_data,file.path(dat,'09a_circos_links.csv'),row.names=FALSE)

# 09b: sender -> ligand -> target alluvial-style triptych.
sender<-do.call(rbind,lapply(1:nrow(stage),function(i){x<-read.csv(file.path(root,'data',paste0('official_ligand_sender_assignment_',stage$assignment[i],'.csv')),stringsAsFactors=FALSE);x$stage<-stage$stage[i];x}))
draw_alluvial <- function(){par(mfrow=c(1,3),mar=c(1,1,3,1),xpd=NA);for(i in 1:nrow(stage)){d<-links_for(stage$window[i],4,3);s<-sender[sender$stage==stage$stage[i],c('ligand','ligand_type')];s<-s[!duplicated(s$ligand),];d<-merge(d,s,by='ligand',all.x=TRUE);d$ligand_type[is.na(d$ligand_type)]<-'Other';snd<-unique(d$ligand_type);lig<-unique(d$ligand);tar<-unique(d$target);yy<-function(v)setNames(seq(.86,.16,length.out=length(v)),v);ys<-yy(snd);yl<-yy(lig);yt<-yy(tar);plot(0,0,type='n',xlim=c(-.3,2.4),ylim=c(0,1),axes=FALSE,xlab='',ylab='',main=paste0(stage$stage[i],'\n',stage$time[i]));for(j in 1:nrow(d)){r<-d[j,];q<-bez(c(1.03,yl[r$ligand]),c(1.34,yl[r$ligand]),c(1.66,yt[r$target]),c(1.97,yt[r$target]));lines(q[,1],q[,2],col=adjustcolor(col_for(r$ligand),.40),lwd=.6+5*r$weight/max(d$weight));q<-bez(c(.03,ys[r$ligand_type]),c(.32,ys[r$ligand_type]),c(.68,yl[r$ligand]),c(.97,yl[r$ligand]));lines(q[,1],q[,2],col=adjustcolor(col_for(r$ligand),.28),lwd=1.2)};points(rep(0,length(snd)),ys,pch=22,bg='#E6E6E6',cex=1.15);points(rep(1,length(lig)),yl,pch=21,bg=col_for(lig),cex=1.18);points(rep(2,length(tar)),yt,pch=21,bg='#E6E6E6',cex=1.0);text(-.05,ys,snd,adj=1,cex=.64);text(1,yl,lig,pos=3,cex=.68,font=2);text(2.06,yt,tar,adj=0,cex=.65);text(0,.98,'sender',font=2,cex=.76);text(1,.98,'ligand',font=2,cex=.76);text(2,.98,'target',font=2,cex=.76)}}
save_base('09b_sender_ligand_target_alluvial_three_stages',16,6,draw_alluvial)
write.csv(merge(circos_data,sender[,c('ligand','stage','ligand_type')],by=c('ligand','stage'),all.x=TRUE),file.path(dat,'09b_alluvial_links_and_senders.csv'),row.names=FALSE)

# Window stage labels: 20 early, 10 intermediate, 10 late intervals.
phase<-function(x)ifelse(x$response_model_time<=1,'Early',ifelse(x$response_model_time<=1.5,'Intermediate','Late'));a$phase<-phase(a);l$phase<-phase(l)
edge_rows<-list();for(w in unique(a$window)){top<-a[a$window==w & a$activity_rank<=5,'test_ligand'];z<-l[l$window==w & l$ligand%in%top,];for(q in top)edge_rows[[length(edge_rows)+1]]<-head(z[z$ligand==q,][order(-z[z$ligand==q,'weight']),],3)}
edges<-do.call(rbind,edge_rows);edges$edge<-paste(edges$ligand,edges$target,sep=' → ');sets<-lapply(c('Early','Intermediate','Late'),function(q)unique(edges$edge[edges$phase==q]));names(sets)<-c('Early','Intermediate','Late');mem<-data.frame(edge=sort(unique(unlist(sets))),stringsAsFactors=FALSE);for(q in names(sets))mem[[q]]<-mem$edge%in%sets[[q]];mem$membership<-apply(mem[,names(sets)],1,function(x)paste0(ifelse(x,c('E','I','L'),''),collapse=''))
cnt<-as.data.frame(table(factor(mem$membership,levels=c('E','I','L','EI','EL','IL','EIL'))));names(cnt)<-c('membership','n_links');cnt$membership<-factor(cnt$membership,levels=rev(c('E','I','L','EI','EL','IL','EIL')))
p_overlap<-ggplot(cnt,aes(membership,n_links))+geom_col(fill='#4C78A8')+coord_flip()+theme_classic(base_size=12)+labs(title='Stage-specific overlap of leading NicheNet ligand–target links',subtitle='Top-five ligands per dense window; top-three prior-supported targets per selected ligand',x='Stage membership (E = early; I = intermediate; L = late)',y='Unique ligand–target links')
ggsave(file.path(fig,'09c_stage_specific_ligand_target_overlap.png'),p_overlap,width=8,height=4.8,dpi=300);ggsave(file.path(fig,'09c_stage_specific_ligand_target_overlap.pdf'),p_overlap,width=8,height=4.8);write.csv(mem,file.path(dat,'09c_stage_link_membership.csv'),row.names=FALSE)

# 09d: dense windows summarized as Top-5 occupancy, rather than trajectory lines.
occ<-aggregate(activity_rank~phase+test_ligand,a,FUN=function(x)mean(x<=5));names(occ)[3]<-'top5_occupancy';sc<-aggregate(aupr_corrected~phase+test_ligand,a,mean);occ<-merge(occ,sc,by=c('phase','test_ligand'));rnk<-aggregate(top5_occupancy~test_ligand,occ,max);keep<-head(rnk[order(-rnk$top5_occupancy),'test_ligand'],12);occ<-occ[occ$test_ligand%in%keep,];occ$test_ligand<-factor(occ$test_ligand,levels=rev(keep));occ$phase<-factor(occ$phase,levels=c('Early','Intermediate','Late'))
p_occ<-ggplot(occ,aes(phase,test_ligand))+geom_point(aes(size=top5_occupancy,colour=aupr_corrected))+scale_size_area(max_size=14,breaks=c(.1,.3,.5,.8),labels=scales::percent_format(accuracy=1))+scale_colour_gradient(low='#D9EAF7',high='#9B1B30')+theme_classic(base_size=12)+theme(axis.title.y=element_blank())+labs(title='Dense-window persistence of NicheNet candidate ligands',subtitle='Point size: fraction of windows ranked in top five; colour: mean corrected AUPR',x='Model-defined transition period',colour='Mean corrected AUPR',size='Top-five occupancy')
ggsave(file.path(fig,'09d_top5_ligand_occupancy_by_stage.png'),p_occ,width=8.2,height=5.8,dpi=300);ggsave(file.path(fig,'09d_top5_ligand_occupancy_by_stage.pdf'),p_occ,width=8.2,height=5.8);write.csv(occ,file.path(dat,'09d_top5_ligand_occupancy.csv'),row.names=FALSE)

# 09e: exact composite of existing official Seurat DotPlots.
draw_dot_composite<-function(){grid.newpage();lay<-grid.layout(1,3);for(i in 1:3){p<-file.path(root,'figures',paste0('01_sender_expression_dotplot_',stage$assignment[i],'.png'));grid.raster(readPNG(p),vp=viewport(layout.pos.row=1,layout.pos.col=i,layout=lay))}}
png(file.path(fig,'09e_sender_expression_dotplot_three_stages.png'),width=3000,height=1100,res=250);draw_dot_composite();dev.off()
pdf(file.path(fig,'09e_sender_expression_dotplot_three_stages.pdf'),width=15,height=5.2);draw_dot_composite();dev.off()

# 09f: target turnover input-control view.
inp<-read.csv(file.path(root,'data','official_nichenet_window_inputs.csv'),stringsAsFactors=FALSE);inp<-inp[inp$response_model_time<=2+1e-9,];inp$phase<-phase(inp);splitg<-function(x)strsplit(x,';',fixed=TRUE)[[1]];tg<-lapply(inp$target_genes,splitg);jac<-sapply(2:length(tg),function(i)length(intersect(tg[[i-1]],tg[[i]]))/length(union(tg[[i-1]],tg[[i]])));turn<-data.frame(window=inp$window[-1],phase=inp$phase[-1],jaccard=jac)
p_turn<-ggplot(turn,aes(phase,jaccard,fill=phase))+geom_boxplot(width=.55,outlier.size=1.3)+geom_jitter(width=.10,size=1.2,alpha=.65)+scale_fill_manual(values=c(Early='#8DA0CB',Intermediate='#FC8D62',Late='#66C2A5'))+theme_classic(base_size=12)+theme(legend.position='none')+labs(title='Turnover of adjacent Microglia transition target sets',subtitle='Jaccard overlap of consecutive top-50 model-derived receiver target sets',x='Transition period',y='Consecutive-window target-set overlap')
ggsave(file.path(fig,'09f_receiver_target_turnover_by_stage.png'),p_turn,width=7,height=4.8,dpi=300);ggsave(file.path(fig,'09f_receiver_target_turnover_by_stage.pdf'),p_turn,width=7,height=4.8);write.csv(turn,file.path(dat,'09f_consecutive_receiver_target_overlap.csv'),row.names=FALSE)
# 09g: observed 17.9-month spatial context for late candidate sender expression.
# This is expression context only, not a spatial interaction map.
suppressPackageStartupMessages(library(hdf5r))
h5 <- H5File$new(file.path(data_root, 'nichenet', 'new_interpolation', 'results', 'interpolation', 'slice_data', 'time_2p00.h5ad'),'r')
xy <- t(h5[['obsm/spatial']]$read()); codes <- h5[['obs/major_annotation/codes']]$read(); cats <- h5[['obs/major_annotation/categories']]$read(); h5$close_all()
expr <- read.csv(gzfile(file.path(root,'intermediate','reconstructed_selected_genes_t2.00.csv.gz')),stringsAsFactors=FALSE)
stopifnot(nrow(expr)==nrow(xy)); expr$major_annotation <- cats[codes+1]
norm <- function(x) {q<-quantile(x[x>0],.99,na.rm=TRUE); pmin(x/q,1)}
sp <- function(v,ct,ttl){bg<-sample(seq_len(nrow(xy)),min(18000,nrow(xy))); plot(xy[bg,1],xy[bg,2],pch=16,cex=.18,col=adjustcolor('#D9D9D9',.45),axes=FALSE,xlab='',ylab='',main=ttl);ix<-which(expr$major_annotation==ct & expr[[v]]>0);points(xy[ix,1],xy[ix,2],pch=16,cex=.30,col=grDevices::colorRampPalette(c('#F7FBFF','#6BAED6','#08306B'))(101)[1+pmax(0,pmin(100,round(100*norm(expr[[v]][ix]))))])}
draw_spatial<-function(){par(mfrow=c(1,3),mar=c(1,1,3,1));sp('Spp1','Fibroblast','Fibroblast Spp1 | observed 17.9 months');sp('Col1a1','Fibroblast','Fibroblast Col1a1 | observed 17.9 months');sp('C1qb','Microglia','Microglia C1qb | observed 17.9 months');mtext('Spatial sender-expression context for late NicheNet candidates; grey = other cells',1,outer=TRUE,line=-1.3,cex=.8)}
save_base('09g_late_candidate_sender_spatial_context',15,5.5,draw_spatial)
write.csv(data.frame(x=xy[,1],y=xy[,2],cell_type=expr$major_annotation,Spp1=expr$Spp1,Col1a1=expr$Col1a1,C1qb=expr$C1qb),file.path(dat,'09g_late_candidate_sender_spatial_data_t2p00.csv'),row.names=FALSE)
writeLines(c('NicheNet SI network preview bank','Inputs: existing official NicheNet output CSVs and exact model-derived NicheNet input CSV.','09a/09b are custom network renderings; 09c/09d/09f custom summaries; 09e is a composite of official Seurat DotPlot panels.','No figure quantifies CytoBridge LR attention or performs NicheNet rescoring.'),file.path(fig,'README.txt'))
