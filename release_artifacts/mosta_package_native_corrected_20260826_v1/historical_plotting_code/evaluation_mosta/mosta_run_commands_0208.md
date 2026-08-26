# MOSTA Run Commands (0208)

## Environment

```bash
conda activate DeepRUOTv2
```

## 1) Mixed real/interp LR batch (n_samples=50000)

Real timepoints (`0/1/2/3`) use real h5ad with Y inverted; interpolation only runs for `0.5/1.5`.

```bash
bash evaluation/mosta/code/run_mosta_lr_batch.sh \
  --only-interp \
  --target-times "0.5,1.5" \
  --project-time-keys "0,0.5,1,1.5,2,3" \
  --ts-points "0.0,0.5,1.0,1.5,2.0,3.0" \
  --start-time 0.0 \
  --n-samples 50000 \
  --lr-pairs "Wnt3a_Fzd7_Lrp6" \
  --interp-comm-pkl "results/mosta_interp_0_3_0208_n_pc_12/mosta_all_time_communications.pkl" \
  --interp-classifier-cache-dir "/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/results/mosta_interp_0_3_0208_n_pc_12" \
  --interp-classifier-n-pcs 12 \
  --interp-classifier-best-metric bacc \
  --interp-classifier-train-on-full-data \
  --time-key-map "0=E12.5,1=E13.5,2=E14.5,3=E15.5" \
  --base-out "results/mosta_lr_batch_0208_mixed_n50000"
```

## 2) Incoming multipanel (6 panels)

Panels in order: `E12.5, 0.5, E13.5, 1.5, E14.5, E15.5`.

```bash
mkdir -p /tmp/mpl_cyto results/mosta_lr_batch_0208_mixed_n50000/incoming_multipanel_nature

MPLCONFIGDIR=/tmp/mpl_cyto conda run -n DeepRUOTv2 \
python evaluation/mosta/code/mosta_incoming_multipanel_nature_local.py \
  --lr-pair Wnt3a_Fzd7_Lrp6 \
  --labels "E12.5,0.5,E13.5,1.5,E14.5,E15.5" \
  --scores-pkls "results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_0.0.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_0.5.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_1.0.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_1.5.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_2.0.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_3.0.pkl" \
  --h5ads "spatial_data/Mouse_embryo_all_stage.h5ad,results/mosta_lr_batch_0208_mixed_n50000/interp_h5ad_0208_n50000/t0p500/adata_t0p500_with_genes.h5ad,spatial_data/Mouse_embryo_all_stage.h5ad,results/mosta_lr_batch_0208_mixed_n50000/interp_h5ad_0208_n50000/t1p500/adata_t1p500_with_genes.h5ad,spatial_data/Mouse_embryo_all_stage.h5ad,spatial_data/Mouse_embryo_all_stage.h5ad" \
  --annotation-cols "annotation,Annotation,annotation,Annotation,annotation,annotation" \
  --time-keys "E12.5,NA,E13.5,NA,E14.5,E15.5" \
  --no-filter-flags "0,1,0,1,0,0" \
  --invert-y-flags "1,0,1,0,1,1" \
  --norm-mode per_panel \
  --out-prefix "results/mosta_lr_batch_0208_mixed_n50000/incoming_multipanel_nature/Wnt3a_Fzd7_Lrp6_incoming_6panel"
```

## 3) Incoming multipanel (5 panels, no 1.5)

Panels in order: `E12.5, 0.5, E13.5, E14.5, E15.5`.

```bash
mkdir -p /tmp/mpl_cyto results/mosta_lr_batch_0208_mixed_n50000/incoming_multipanel_nature

MPLCONFIGDIR=/tmp/mpl_cyto conda run -n DeepRUOTv2 \
python evaluation/mosta/code/mosta_incoming_multipanel_nature_local.py \
  --lr-pair Wnt3a_Fzd7_Lrp6 \
  --labels "E12.5,0.5,E13.5,E14.5,E15.5" \
  --scores-pkls "results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_0.0.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_0.5.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_1.0.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_2.0.pkl,results/mosta_lr_batch_0208_mixed_n50000/interp_projection_0208_n50000/lr_scores_3.0.pkl" \
  --h5ads "spatial_data/Mouse_embryo_all_stage.h5ad,results/mosta_lr_batch_0208_mixed_n50000/interp_h5ad_0208_n50000/t0p500/adata_t0p500_with_genes.h5ad,spatial_data/Mouse_embryo_all_stage.h5ad,spatial_data/Mouse_embryo_all_stage.h5ad,spatial_data/Mouse_embryo_all_stage.h5ad" \
  --annotation-cols "annotation,Annotation,annotation,annotation,annotation" \
  --time-keys "E12.5,NA,E13.5,E14.5,E15.5" \
  --no-filter-flags "0,1,0,0,0" \
  --invert-y-flags "1,0,1,1,1" \
  --norm-mode per_panel \
  --out-prefix "results/mosta_lr_batch_0208_mixed_n50000/incoming_multipanel_nature/Wnt3a_Fzd7_Lrp6_incoming_5panel_no1p5"
```
