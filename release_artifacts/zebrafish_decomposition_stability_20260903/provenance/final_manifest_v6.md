# Final three-panel figure provenance

## Source paths

- Fixed aligned input: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-de-novo-20260813-r2/zebrafish/preprocess/zebrafish_aligned.h5ad`
- Fixed LR edge predictor: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-de-novo-20260813-r2/zebrafish/preprocess/edge_classifier/zebrafish_edge_model.pt`
- Package release: `/data/cytobridge/projects/CytoBridge-ST-1104/software/cytobridge-release-e80c18a`, commit `e80c18a`
- Versioned training run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-decomposition-stability-20260903-e80c18a-r1`
- Accepted evaluated arrays and model-level manifests: `evaluation_edge_centered/` in the versioned server run above. These cell-level files are not duplicated in the public compact archive.
- Accepted panel tables: `panel_data_final/`
- Accepted figure: `figure/zebrafish_decomposition_stability_a4.pdf` and `figure/zebrafish_decomposition_stability_a4.png`
- Canonical plotting script: `code/plot_zebrafish_decomposition_stability_v2.py`
- Complete machine-readable archive inventory: `archive_manifest.json`

Every model-level `evaluation_manifest.json` records the training configuration, checkpoint path, and SHA-256 checkpoint hashes. Panel a compares five complete default fits. Panel c compares only matched changes to model settings. Training-seed reconstruction variability is reported across all five complete fits in the caption rather than as a percentage relative to one selected seed. The displayed model settings exclude the archived 1:10 transport-to-mass stress test. Training seed 45 failed twice with a non-finite score-matching loss before Finetune. Those attempts are retained in the server run and status logs but excluded from the five complete default fits used in the figure.

## Rebuild

From this archive directory:

```bash
python code/plot_zebrafish_decomposition_stability_v2.py \
  --panel-data panel_data_final \
  --output-dir figure_rebuilt
```

The plotting script reads numerical CSV and JSON tables. It does not load or re-export a pre-rendered figure.

## SHA-256

- PDF: `1b3bb9329b395959752e830ae25eada17c771d9726f85facdf5aa710dcd798ec`
- PNG: `d7255f3794c1e33a00fd02a50bea11fa71efe70eacbca4160adf084e7b907b3c`
- Plotting script: `08f4152c5bf59c26e378cf7fcd8cd6d147a5a1a7f89e12d6a6337ebbcf25ef84`
- Analysis summary: `749ce77002562062264c83a3f684099ba0bcc429e224d08b3f25a882aae10f7d`
- Training-seed component table: `9995f11d90954b5e265d7b100cc7928001c33d47737c674985520be183b0c938`
- Training-seed directed-pair table: `587b4032b04f7060950bcfe74868734ee8726a992af8bb088ee8deea318a9703`
- Model-setting component table: `d58ed94650f85a4023e4406469a2323937cd47e3be2898ee373c15b7dd0fb3b5`
- Model-setting directed-pair table: `8b52618d38f9ffaf23f056c6c084ad7c370a65dd5a283151271ce20e93d46696`
- Reconstruction table: `10ed3c88c7fb83f2d5109022432c101946d6f0c265da554a9fddc94783e39885`
