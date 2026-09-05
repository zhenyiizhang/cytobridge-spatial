# Train a model

This page starts from expression counts and ends with a trained model. Each
step uses the object or file produced by the preceding step. If you already
have a trained model, go directly to an [analysis tutorial](tutorials/dataset_workflows/index.md).

The example uses `layers['counts']`, `obs['stage']`, `obs['cell_type']`, and
`obsm['spatial']`. Change those names and the time mapping to match your AnnData.
Use an LR table for your species with four columns: `Ligand`, `Receptor`,
`Pathway`, and `Annotation`. Keep the pathway and interaction-type annotations
from the database. In particular, `Cell-Cell Contact` pairs are evaluated using
the contact-distance rule. The package's included CellChatDB tables use the
equivalent four-column format.

```text
Ligand,Receptor,Pathway,Annotation
Tgfb1,Tgfbr1_Tgfbr2,TGFb,Secreted Signaling
Tgfb2,Tgfbr1_Tgfbr2,TGFb,Secreted Signaling
```

## Prepare expression features

```python
from pathlib import Path
import scanpy as sc
import CytoBridge as cb

PROJECT_DIR = Path(".").resolve()
OUTPUT_DIR = PROJECT_DIR / "outputs" / "my_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
adata = sc.read_h5ad(PROJECT_DIR / "data" / "my_experiment.h5ad")
lr_table = PROJECT_DIR / "data" / "ligand_receptor_pairs.csv"

adata = cb.pp.preprocess(
    adata, time_key="stage", expression_layer="counts",
    time_mapping={"D0": 0.0, "D2": 1.0, "D4": 2.0},
    n_top_genes=2000, n_pcs=50,
)
adata.obsm["X_latent"].shape
```

`preprocess` normalizes expression, selects variable genes, and calculates PCA
features. It stores those features in `obsm['X_latent']` and numeric model times
in `obs['time_point_processed']`. Keep the relative spacing between collection
times when choosing the mapping. The original counts remain available for the
LR calculation below.

## Align spatial coordinates

```python
alignment = cb.pp.AlignConfig(
    input_spatial_key="spatial", spatial_dim=2,
    center_x=True, center_y=True,
)
adata = cb.pp.align_spatial(
    adata, time_key="stage", cfg=alignment, device="cuda",
)
adata.obsm["spatial_aligned"].shape
```

`align_spatial` reads the processed features from the previous step and aligns
the tissue coordinates. It does not repeat expression preprocessing. Inspect
the aligned coordinates before training:

```python
sc.pl.embedding(adata, basis="spatial_aligned", color="stage")
```

## Construct the LR graph and fit its edge predictor

Choose a neighbourhood radius in the aligned coordinate units. The function
below estimates an initial value from cell spacing. Inspect that value against
the scale of your tissue and adjust it if needed.

```python
radius, spot_diameter, spacing, coordinate_key = (
    cb.pp.estimate_neighborhood_threshold_from_aligned_spatial(
        adata, time_key="time_point_processed",
    )
)
radius
```

Build a graph at each observed time using the aligned positions, raw expression,
and LR table. These graph files are the training input for the edge predictor.

```python
graph_dir = OUTPUT_DIR / "graphs"
times = sorted(adata.obs["time_point_processed"].unique())
for i, time in enumerate(times):
    cb.pp.generate_interaction_graph(
        data_name=f"my_experiment_t{i}", data_from=adata,
        data_to=str(graph_dir / f"my_experiment_t{i}"),
        database_path=str(lr_table), time_value=float(time),
        expression_layer="counts", spatial_key="spatial_aligned",
        neighborhood_threshold=radius, auto_neighborhood_threshold=False,
        spot_diameter=spot_diameter,
    )

edge_model_path = OUTPUT_DIR / "edge_classifier" / "model.pt"
edge_model_path.parent.mkdir(parents=True, exist_ok=True)
edge_result = cb.pp.train_edge_predictor(
    data_name="my_experiment", adata_or_h5ad=adata,
    graph_input_dir=str(graph_dir), output_model_path=str(edge_model_path),
    distance_threshold=radius, epochs=100, device="cuda",
)
```

The returned `edge_result` includes the decision threshold selected during
training. The fitted edge predictor is saved at `edge_model_path`.

## Fit cell-state dynamics

Load an existing training schedule as a starting point. This example uses the
Zebrafish schedule, then supplies this experiment's neighbourhood radius and
newly trained edge predictor. Review the schedule and loss weights for your data.

```python
from CytoBridge.utils.config import load_config

training = load_config("zebrafish_spatial_full_alpha_express_0015")
training.pop("matched_ablation", None)  # This is a new experiment.
cb.pp.sanitize_interaction_graph_uns(adata)
aligned_path = OUTPUT_DIR / "aligned.h5ad"
adata.write_h5ad(aligned_path)

trained = cb.tl.fit(
    str(aligned_path), config=training, device="cuda",
    interaction_cutoff=radius,
    edge_predictor_path=str(edge_model_path),
    edge_predictor_threshold=edge_result["edge_predictor_threshold"],
    ckpt_dir=OUTPUT_DIR / "model", evaluate_after_training=False,
)
```

`fit` reads the aligned expression features and coordinates, fits the dynamical
model, and saves its configuration and checkpoints in `OUTPUT_DIR / 'model'`.
It does not rerun preprocessing or graph construction.

To continue in the same Python session, load this model and its states:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

states = cb.tl.model_state_adata(adata)
model = cb.tl.load_dynamical_model_from_dir(
    OUTPUT_DIR / "model", dim=states.n_vars, device="cuda",
    edge_predictor_path=edge_model_path,
).model
time_key = "time_point_processed"
annotation_key = "cell_type"
observed_times = sorted(states.obs[time_key].unique())
DEVICE = "cuda"
```

You can now use the [analysis tutorial](tutorials/dataset_workflows/chicken_heart.ipynb)
from **Calculate velocity components** onward, with these variables. Skip its
dataset download and model-loading sections. No study `workflow.json` is needed
for these velocity, growth, or attention calculations.

## Paper training settings

The [dataset configurations](https://github.com/zhenyiizhang/cytobridge-spatial/tree/main/CytoBridge/workflow_configs)
specify the preprocessing, alignment, LR database, and training settings for
each study. Use those settings and the matching study input when reproducing
a paper model. The generic example above is for learning the API with a new
experiment.
