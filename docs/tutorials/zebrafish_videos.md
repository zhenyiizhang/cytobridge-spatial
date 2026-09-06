# Zebrafish videos

These videos follow the simulated tissue from 5.25 to 24 hours post-fertilization.
Videos 4 and 5 compare the baseline with virtual removal of YSL or EVL cells.
A third video shows the baseline with directed interaction edges.

Run the commands from a [CytoBridge code folder](../installation.md). Install the
notebook dependencies, which include a video encoder:

```bash
python -m pip install -e '.[notebook,velocity]'
```

## Download the trajectories and model

```python
import CytoBridge as cb

cb.datasets.download("zebrafish", kind="analysis")
cb.datasets.download("zebrafish", kind="zebrafish_video_data.zip")
```

The video download contains `baseline_points.npy`, `remove_YSL_points.npy`, and
`remove_EVL_points.npy` in `data/zebrafish/videos/trajectories/`. Each array has
81 simulated states on the model-time grid 0–4. The first two columns of a
state are spatial coordinates. The remaining columns are expression features.
The number of cells can change between frames because growth is enabled.

## Videos 4 and 5: virtual removal

The following command assigns cell types to each simulated state, draws all
frames, and encodes both videos. It uses the same classifier as S33, reading
the original 50 expression PCs, two spatial coordinates and model time.

```bash
python release_assets/zebrafish_videos/source/render_latest_ablation_videos.py \
  --package-root . \
  --trajectory-root data/zebrafish/videos/trajectories \
  --classifier data/zebrafish/classifier_cache/classifier_resmlp_25f65c49dc60ea4c.pt \
  --colors data/zebrafish/videos/label_to_color.json \
  --output-dir outputs/zebrafish_videos
```

The output directory contains the MP4 files, representative frames, assigned
cell types, and the display-region coordinates. Choose a new output directory
when running the command again.

## Baseline with interaction edges

First evaluate the trained interaction model on the downloaded baseline
trajectory. This step uses a GPU and writes `learned_interactions.npz` plus a
per-frame summary. Up to 260 directed edges with the largest scores are displayed
at each selected frame.

```bash
python release_assets/zebrafish_videos/source/render_zebrafish_baseline_virtual_tissue_dynamics.py extract \
  --release . \
  --trajectory data/zebrafish/videos/trajectories/baseline_points.npy \
  --h5ad data/zebrafish/aligned.h5ad \
  --model-dir data/zebrafish/model \
  --device cuda:0 \
  --output outputs/zebrafish_interactions
```

Then draw the frames and write the MP4, GIF, and preview image:

```bash
python release_assets/zebrafish_videos/source/render_zebrafish_baseline_virtual_tissue_dynamics.py render \
  --trajectory data/zebrafish/videos/trajectories/baseline_points.npy \
  --extraction outputs/zebrafish_interactions \
  --output outputs/zebrafish_baseline_video
```

To redraw directly from the paper's saved interaction arrays, use
`--extraction release_assets/zebrafish_videos/derived_inputs` in the second
command. This skips model evaluation, but still draws every frame from
numerical coordinates and edges.
