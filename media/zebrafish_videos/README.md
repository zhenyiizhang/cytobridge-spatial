# Zebrafish videos

This directory keeps the final YSL- and EVL-removal videos together with their
poster images, rendering code, and compact derived inputs. The baseline tissue
dynamics export is also included, but its supplementary-video number has not
yet been assigned in the paper files.

## Media

- `Supplementary_Video_4_Zebrafish_YSL_Ablation.mp4`: baseline and Yolk
  Syncytial Layer removal, rendered from t = 0 to t = 4.
- `Supplementary_Video_5_Zebrafish_EVL_Ablation.mp4`: baseline and EVL removal,
  rendered from t = 0 to t = 4.
- `zebrafish_baseline_virtual_tissue_dynamics.mp4`: growth-on baseline with
  learned directed interactions.
- `zebrafish_baseline_virtual_tissue_dynamics.gif`: browser-friendly version of
  the baseline export.

Videos 4 and 5 use 81 frames, 10 frames per second, H.264 encoding, and a
2520 × 1260 frame. Colors denote classifier-assigned cell states. Dashed boxes
mark the spatial regions highlighted in the paired trajectories.

## Rendering code and inputs

- `source/render_latest_ablation_videos.py` renders Videos 4 and 5 from the
  baseline, YSL-removal, and EVL-removal trajectory arrays.
- `source/render_zebrafish_baseline_virtual_tissue_dynamics.py` extracts learned
  interactions and renders the baseline video.
- `derived_inputs/classifier_assigned_labels.npz` stores the assigned labels
  used in Videos 4 and 5.
- `derived_inputs/roi_schedule.json` stores the displayed region schedule.
- `derived_inputs/learned_interactions.npz` and
  `derived_inputs/interaction_frame_summary.csv` store the interaction overlay
  used in the baseline export.

The trajectory arrays, model directory, aligned AnnData object, classifier,
and color table are external inputs. Their paths are provided
as command-line arguments; they are not copied into the Python wheel.

## Captions

**Supplementary Video 4.** Baseline and YSL-removal trajectories generated
continuously from t = 0 to t = 4.

**Supplementary Video 5.** Baseline and EVL-removal trajectories generated
continuously from t = 0 to t = 4.

Colors denote classifier-assigned cell states. Dashed boxes mark regions with
the largest spatial differences between the paired trajectories.

**Baseline tissue dynamics.** The growth-on baseline is propagated continuously
from t = 0 to t = 4. Arrows show the strongest directed cell-cell interactions
selected by the frozen learned edge predictor at each generated state; color
indicates first-layer attention strength.

Videos 4 and 5 use `classifier_resmlp_25f65c49dc60ea4c.pt`, the same classifier
as S33. Its inputs are time, two spatial coordinates and 50 expression PCs.
