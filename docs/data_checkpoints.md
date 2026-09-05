# Data and models

For downstream analysis, get the model and analysis-data archives for your
dataset. Extract both into your working directory. You can then run the
dataset notebook without training.

## Dataset downloads

The [paper data release](https://github.com/zhenyiizhang/cytobridge-spatial/releases/tag/paper-data-20260905)
contains the files used below. After installing CytoBridge, download a dataset
into your working directory:

```python
import CytoBridge as cb

data_dir = cb.datasets.download("chicken_heart", destination=".")
```

This downloads the model and analysis data and extracts them into
`data/chicken_heart/`. You can now run the chicken-heart notebook. Use the
same working directory for `PROJECT_DIR` in the notebook.

For the command line:

```bash
python -m CytoBridge.datasets chicken_heart --output-dir .
```

Use `kind="all"` in Python, or `--kind all` on the command line, to include
the additional files listed below. Completed downloads are reused. MOSTA's
two parts are joined automatically. Allow about 20 GB of free space for
downloading and extracting MOSTA.

You can also download and extract the files manually:

| Dataset | Model archive | Analysis data |
| --- | --- | --- |
| Chicken heart | [chicken_heart_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/chicken_heart_model.zip) | [chicken_heart_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/chicken_heart_analysis_data.zip) |
| MOSTA | [mosta_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/mosta_model.zip) | [mosta_analysis_data.zip.part01](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/mosta_analysis_data.zip.part01) · [mosta_analysis_data.zip.part02](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/mosta_analysis_data.zip.part02) |
| ARISTA | [arista_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/arista_model.zip) | [arista_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/arista_analysis_data.zip) |
| AD mouse | [admouse_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/admouse_model.zip) | [admouse_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/admouse_analysis_data.zip) |
| Zebrafish | [zebrafish_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/zebrafish_model.zip) | [zebrafish_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/zebrafish_analysis_data.zip) |
| Weinreb | [weinreb_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/weinreb_model.zip) | [weinreb_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/weinreb_analysis_data.zip) |
| scNT cortex | [scnt_cortex_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/scnt_cortex_model.zip) | [scnt_cortex_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/scnt_cortex_analysis_data.zip) |
| Spatial simulation (S3) | [simulation_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/simulation_model.zip) | [simulation_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/simulation_analysis_data.zip) |
| AGIST (Figure 2) | [agist_model.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/agist_model.zip) | [agist_analysis_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/agist_analysis_data.zip) |

For the five spatial datasets, the model archive contains the final dynamical model, score model, LR edge
predictor, fitted cell-type classifier, and settings. The analysis-data archive
contains the matching aligned H5AD.

For example, extracting the two chicken-heart archives gives:

```text
data/chicken_heart/
├── workflow.json
├── aligned.h5ad
├── model/
│   ├── config.yaml
│   ├── Finetune/
│   └── Score_Refine/
├── edge_classifier/
└── classifier_cache/
```

Open the [chicken-heart tutorial](tutorials/dataset_workflows/chicken_heart.ipynb)
and set `PROJECT_DIR` to the folder where you extracted the downloads. It reads
these files and writes new results to `outputs/`. The notebook specifies its
output subdirectory. Non-spatial Full and No-interaction models occupy separate
directories within their dataset folder.

## Additional paper analyses

| Download | Used for |
| --- | --- |
| [mosta_figure_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/mosta_figure_data.zip) | Saved populations and numerical tables for Figure 4 and S11–S18 |
| [arista_figure_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/arista_figure_data.zip) | Saved populations, communication summaries, and lineage labels for Figure 5a–b |
| [arista_spatial_display_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/arista_spatial_display_data.zip) | Spatially anchored and unwarped numerical populations for Figure 5a–b and S19, plus per-cell growth for S20 |
| [admouse_population_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/admouse_population_data.zip) | The saved states and cell-type labels for Figure 6b and S26 |
| [admouse_perturbation_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/admouse_perturbation_data.zip) | Trem2 states, attention edges, and module scores for Figure 6f–g |
| [admouse_nichenet_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/admouse_nichenet_data.zip) | The 51 states, expression summaries, and NicheNet reference networks for S29 |
| [zebrafish_video_data.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/zebrafish_video_data.zip) | Numerical trajectories and labels for the zebrafish videos |
| [chicken_heart_training_inputs.zip](https://github.com/zhenyiizhang/cytobridge-spatial/releases/download/paper-data-20260905/chicken_heart_training_inputs.zip) | Count matrices, section annotations, and alignment inputs |

For example, add the saved AD populations to an existing download:

```python
cb.datasets.download("admouse", kind="admouse_population_data.zip")
```

NicheNet's reference networks and functions are from
[NicheNet](https://github.com/saeyslab/nichenetr). Cite the original dataset
and analysis method when using their data.

## Train from counts

Training needs counts, annotations, and the original coordinates. These files
are separate from the aligned H5AD used to reopen an existing model.

For chicken heart, `chicken_heart_training_inputs.zip` contains the count
matrices and section annotations. Follow
[Prepare the chicken-heart counts](chicken_heart_preparation.md) to combine
them into a training input.
For other studies, the source data and AnnData fields are listed below.

## Raw inputs

| Application | Source | Time and label keys | Expression and coordinates |
| --- | --- | --- | --- |
| Zebrafish | [CNGB STDS0000057](https://db.cngb.org/stomics/datasets/STDS0000057/data) | `time`, `bin_annotation` | `layers['counts']`, `obs[['spatial_x', 'spatial_y']]` |
| MOSTA | [MOSTA portal](https://db.cngb.org/stomics/mosta/download/) | `timepoint`, `annotation` | `layers['count']`, `obsm['spatial']` |
| ARISTA axolotl | [CNGB STDS0000056](https://db.cngb.org/stomics/datasets/STDS0000056/data) | `Batch`, `Annotation` | `layers['counts']`, `obsm['spatial']` |
| AD mouse | [10x Genomics TgCRND8 Xenium time course](https://www.10xgenomics.com/datasets/xenium-in-situ-analysis-of-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard) | `Timepoint`, `major_annotation` | `layers['counts']`, `obsm['spatial']` |
| Chicken heart | [GEO GSE149457](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149457) | `timepoint`, `celltype_prediction` | four raw 10x matrices, `obsm['spatial_original']` |

The dataset configuration maps these source keys to a common representation during
preprocessing.


## Draw a figure from saved results

The [figure notebooks](tutorials/paper_figures/index.md) use numerical results
included with the source repository. Their pages list any additional study
files needed for a particular analysis.

[Input and output formats](file_formats.md) describes the AnnData fields,
model directories, and analysis tables.

```{toctree}
:hidden:

chicken_heart_preparation
```
