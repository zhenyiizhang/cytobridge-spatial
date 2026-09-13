# Lineage analysis

Cell types are assigned by the trained classifier, and transitions count the same particles across time. The inputs here supply Figure 4c and Supplementary Figures S14 and S37.

## MOSTA

`mosta/fixed_particle_labels.csv.gz` contains labels for 50,000 particles at seven times. `mosta/cartilage_lineage.npz` contains the selected E15.0 cartilage-primordium particles, their E15.5 labels and coordinates, and the background cell coordinates. The complete destination counts are in `mosta/Figure4c_lineage_fractions.csv`.

To calculate these inputs from the trained model, run the **Figure 4: MOSTA** notebook through panel c. It simulates the fixed-particle trajectory, loads the classifier, predicts each particle's cell type, and counts the destinations. The **MOSTA: mouse organogenesis** notebook also writes the complete particle-label table for S14.

To redraw from the numerical inputs in this folder:

```python
from pathlib import Path
import json
from reproduction.mosta.main_figure import draw_cartilage
from reproduction.mosta.figures import PALETTE_FILE, draw_lineage

output = Path('outputs/mosta_lineage')
output.mkdir(parents=True, exist_ok=True)
draw_cartilage(output, json.loads(PALETTE_FILE.read_text()))
draw_lineage(Path('data/mosta/paper/shared'), output)
```

The PDF preserves vector text and connections. The dense scatter background uses a high-resolution image. Figure 4 keeps the other four panels and their approved annotations unchanged.

## Zebrafish

The five `zebrafish/seed_*` directories retain the four daughter-cell perturbation settings and their source-to-target counts. `plots/s33_*` files use historical filenames and supply the current S37.

```python
from pathlib import Path
from reproduction.zebrafish.plot_daughter_noise import collect, draw

source = Path('release_artifacts/lineage_classifier_predictions_20260913/zebrafish')
output = Path('outputs/zebrafish_lineage')
tables = collect([source / f'seed_{seed}' for seed in range(42, 47)], output)
draw(tables, output)
```

The **Zebrafish S31–S38** notebook calculates the trajectories and classifier labels before collecting these results. Cell composition and growth use the same values as before this lineage update.

## Calculation records

The `calculation_record.json` and `report.json` files record the original trajectories and classifier weights. Full state trajectories remain with the model-analysis data. The small files included here were calculated from those trajectories, not extracted from figure images. ARISTA Figure 5a and S21 are archived separately under the ARISTA release folder.
