# Supplementary Videos 1–5

These videos animate the spatial populations and interactions shown in the paper.
All five use the simulated coordinates directly, without spatial warp.
The code draws each frame from numerical cell states and then encodes an MP4.

| Video | Dataset | Display |
| --- | --- | --- |
| 1 | MOSTA | Mouse organogenesis, E12.5–E15.5 |
| 2 | ARTISTA | Axolotl brain regeneration, 2–10 days post injury |
| 3 | AD mouse | TgCRND8 brain, 2.5–17.9 months |
| 4 | Zebrafish | Baseline and virtual YSL removal |
| 5 | Zebrafish | Baseline and virtual EVL removal |

## Before you start

Run these commands from the [source-code folder](../installation.md).
Install the model and plotting dependencies once:

```bash
python -m pip install -e '.[notebook,velocity]'
```

Videos 1–3 have two steps: simulate populations from a trained model, then draw
the video. Simulation uses a GPU by default. Rendering uses the saved arrays
and runs on the CPU. If you already have the video simulation output described
below, go directly to the rendering command. No training is needed.

Each command below is separate. A trailing `\` continues the same command on
the next line. Use a new output directory when repeating a run.
The encoder is included through `imageio-ffmpeg`. Install Arial to match the
paper typography; Matplotlib uses a fallback font if Arial is unavailable.

The data commands below download and extract the required archives.
[Download the five submission videos](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-revision-20261104/CytoBridge_Supplementary_Videos_1-5_submission_20261104.zip)
to view the final MP4 files.

The commands below extract each archive into `data/`. If you already downloaded
and extracted an archive for the dataset tutorials, reuse it rather than
extracting it again.

The examples assume your files are under `data/` in the source-code folder.
If the dataset tutorials put them under `paper_run/data/`, replace the input
paths below with that prefix, for example `--data-dir paper_run/data/mosta`.
This also applies to the zebrafish trajectory and classifier paths. Keep running
the commands from the source-code folder so Python can find `reproduction/`.

## Video 1: MOSTA

```{image} ../_static/supplementary_videos/video_1.png
:alt: MOSTA video frame showing tissue cell types and Brain-related interaction arrows
:width: 70%
```

Download the aligned data, trained model and classifier:

```bash
python -m CytoBridge.datasets mosta --output-dir .
```

Generate the cell states, labels and attention arrays, then render:

```bash
python -m reproduction.videos.simulate mosta \
  --data-dir data/mosta \
  --output outputs/videos/mosta_states \
  --device cuda:0
python -m reproduction.videos.render mosta --input outputs/videos/mosta_states \
  --output outputs/videos/mosta
```

The camera limits are fixed across frames. Cell labels use k=10 spatial voting.
Arrows show the three highest-ranked directed cell-type pairs involving Brain.
A selected pair supported by fewer than two distinct cell-cell edges is omitted,
without replacing it with a lower-ranked pair. The saved attention values and
rankings are unchanged.

## Video 2: ARTISTA

```{image} ../_static/supplementary_videos/video_2.png
:alt: ARTISTA video frame with a zoomed inset of reaEGC interactions
```

```bash
python -m CytoBridge.datasets arista --output-dir .
python -m reproduction.videos.simulate arista \
  --data-dir data/arista \
  --output outputs/videos/arista_states \
  --device cuda:0
python -m reproduction.videos.render arista --input outputs/videos/arista_states \
  --output outputs/videos/arista
```

The video covers the first three measured stages, at 2, 5 and 10 days post injury.
The inset enlarges interactions involving reactive ependymoglial cells (reaEGC).
Cell labels use k=10 spatial voting.

## Video 3: AD mouse

```{image} ../_static/supplementary_videos/video_3.png
:alt: AD mouse video frame with the main-figure cell-type colors
:width: 70%
```

```bash
python -m CytoBridge.datasets admouse --output-dir .
python -m reproduction.videos.simulate admouse \
  --data-dir data/admouse \
  --output outputs/videos/admouse_states \
  --device cuda:0
python -m reproduction.videos.render admouse --input outputs/videos/admouse_states \
  --output outputs/videos/admouse
```

Colors match the main-figure cell types, with no additional desaturation. Cell
labels use k=1. The title converts model time to age using the measured stages
at 2.5, 5.7 and 17.9 months.

## Videos 4 and 5: zebrafish

```{image} ../_static/supplementary_videos/video_4.png
:alt: Final frame of Supplementary Video 4, baseline beside YSL removal
```

```{image} ../_static/supplementary_videos/video_5.png
:alt: Final frame of Supplementary Video 5, baseline beside EVL removal with the R2 box
```

Download the trained classifier and the saved numerical trajectories:

```bash
python -m CytoBridge.datasets zebrafish --kind zebrafish_model.zip --output-dir .
python -m CytoBridge.datasets zebrafish --kind zebrafish_video_data.zip --output-dir .
```

Render both paired comparisons:

```bash
python -m reproduction.videos.zebrafish \
  --trajectory-root data/zebrafish/videos/trajectories \
  --classifier data/zebrafish/classifier_cache/classifier_resmlp_25f65c49dc60ea4c.pt \
  --colors CytoBridge/results/data/zebrafish_si/celltype_colors.json \
  --output-dir outputs/videos/zebrafish
```

This command classifies every frame using k=10, draws the baseline beside each
removal condition, and exports both MP4 files. Video 5 includes the left-shifted R2 box. The same boxes appear in both columns.

To calculate these trajectories again, follow the virtual-removal analysis in
the [zebrafish tutorial](dataset_workflows/zebrafish.ipynb), or run:

```bash
python -m CytoBridge.datasets zebrafish --kind zebrafish_analysis_data.zip --output-dir .
python -m reproduction.zebrafish.virtual_removal \
  --data-dir data/zebrafish \
  --output-dir outputs/videos/zebrafish_simulation \
  --seed 42 \
  --device cuda:0
```

Then use `outputs/videos/zebrafish_simulation/experiment/trajectories` as
`--trajectory-root` in the rendering command. This is the same virtual-removal
analysis used for S39 and S40, not a second training run.

## Files produced

For Videos 1–3, simulation writes `states/frame_*.npz` containing model time,
cell states and labels. The first two columns of each state are spatial
coordinates. MOSTA and ARTISTA also write `attention/` and `communication.json`.
The renderer reads this directory and writes numbered PNG frames and an MP4.
It does not read a finished figure or video.

The zebrafish renderer reads `baseline_points.npy`, `remove_YSL_points.npy` and
`remove_EVL_points.npy`. It saves the assigned labels, region boxes, preview
images and two MP4 files. Input arrays with pickled objects should come from
your own runs or the project downloads.

The exported videos are:

```text
Supplementary_Video_1_Figure4_Mouse_Organogenesis.mp4
Supplementary_Video_2_Figure5_Axolotl_Regeneration.mp4
Supplementary_Video_3_Figure6_Alzheimer_Progression.mp4
Supplementary_Video_4_Zebrafish_YSL_Ablation.mp4
Supplementary_Video_5_Zebrafish_EVL_Ablation.mp4
```

To change a layout, edit the rendering functions in `reproduction/videos/` and
rerun only the rendering command. Keep the numerical inputs to reuse the same
trajectories. A new stochastic simulation may differ slightly across software
and hardware versions.

The [additional zebrafish video](zebrafish_videos.md)
shows individual interaction edges and is separate from Supplementary Videos 1–5.
