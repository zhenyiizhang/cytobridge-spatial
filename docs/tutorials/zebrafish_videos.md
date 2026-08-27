# Zebrafish videos

The release-assets directory contains the two virtual-removal videos, the
baseline tissue-dynamics export, poster images, rendering scripts, and compact
derived inputs:

`release_assets/zebrafish_videos/`

## Virtual-removal videos

The YSL and EVL videos compare each virtual-removal trajectory with the same
baseline trajectory from t = 0 to t = 4. The rendering command takes three
trajectory arrays, the packaged label classifier inputs, a color table, and an
empty output directory.

```bash
python release_assets/zebrafish_videos/source/render_latest_ablation_videos.py \
  --package-root /path/to/cytobridge-spatial \
  --trajectory-root /path/to/trajectory_arrays \
  --classifier /path/to/classifier.pt \
  --pca /path/to/ablation_classifier_pca10.npz \
  --colors /path/to/label_to_color.json \
  --reference-labels /path/to/global_t0_labeled_sources.npz \
  --output-dir video_outputs
```

## Baseline tissue dynamics

The baseline renderer has two commands. `extract` evaluates learned directed
interactions on the stored trajectory. `render` uses that compact interaction
bundle to create MP4, GIF, and poster outputs.

```bash
python release_assets/zebrafish_videos/source/render_zebrafish_baseline_virtual_tissue_dynamics.py extract \
  --release /path/to/cytobridge-spatial \
  --trajectory /path/to/baseline_points.npy \
  --h5ad /path/to/zebrafish_aligned.h5ad \
  --model-dir /path/to/zebrafish_model \
  --device cuda:0 \
  --output interaction_outputs

python release_assets/zebrafish_videos/source/render_zebrafish_baseline_virtual_tissue_dynamics.py render \
  --trajectory /path/to/baseline_points.npy \
  --extraction interaction_outputs \
  --output baseline_video_outputs
```

The baseline export is archived with the code and derived inputs. Its final
supplementary-video number will be added when the paper files assign one.

