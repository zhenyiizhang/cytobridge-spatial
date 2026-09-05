# Chicken-heart trained model

This is the final model from `chicken-heart-full-ot-20260823-r2`, trained with
CytoBridge commit `c72e592d0dea70941bc4971a79c3c903d7454b08`.
The input contains 3,550 spots: 147 at D4, 528 at D7, 908 at D10, and 1,967 at D14.

The final dynamical and score checkpoints and the edge predictor are included.
The aligned H5AD is distributed separately. Original training settings and the
training record are retained unchanged.

## Load the model

From the repository root:

```python
from pathlib import Path
import CytoBridge as cb

release = Path("release_artifacts/chicken_heart_model_c72e592")
loaded = cb.tl.load_dynamical_model_from_dir(
    release / "training",
    dim=52,
    device="cpu",
    edge_predictor_path=release / "preprocess/edge_classifier/chicken_heart_edge_model.pt",
)
```

The model takes 50 expression PCA coordinates and two spatial coordinates.

## Run the analyses

After downloading the matching aligned H5AD to `data/chicken_heart_aligned.h5ad`:

```bash
cytobridge workflow --config chicken_heart --step downstream \
  --aligned-h5ad data/chicken_heart_aligned.h5ad \
  --model-dir release_artifacts/chicken_heart_model_c72e592/training \
  --edge-predictor-path release_artifacts/chicken_heart_model_c72e592/preprocess/edge_classifier/chicken_heart_edge_model.pt \
  --output-dir outputs/chicken_heart_analysis --device cuda
```

This runs the standard downstream analyses. The collaborator's daily
interpolation and final paper layouts use additional code and inputs, described
in [the code inventory](../../reproduction/chicken_heart/README.md).

Do not substitute a newly fitted PCA representation for this model's aligned
input.
