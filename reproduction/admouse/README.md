# AD mouse figure calculations

The [AD figure guide](../../docs/tutorials/paper_figures/admouse_figures.md)
lists the inputs, dependencies, and commands in calculation order.
Run the commands from the source checkout.

## Gene programs and LR time courses

The input tables are included with the code. This command clusters gene
profiles, calculates the program curves and LR z-scores, and draws Figure
6c–e and the Spp1 module contrasts in S30:

```bash
python reproduction/admouse/draw_figures.py \
  --panels cd e s30 --output-dir outputs/admouse_programs
```

The four program gene lists are written to `outputs/admouse_programs/tables/`.
Use program 1 and program 4 with `go/enrich_program.R` for S27–S28.

## Populations and perturbations

Extract `admouse_population_data.zip` and `admouse_perturbation_data.zip`
from the [data release](https://github.com/zhenyiizhang/cytobridge-spatial/releases/tag/paper-data-20260905).

```bash
python reproduction/admouse/plot_population.py \
  --run-dir data/admouse/populations --output-dir outputs/admouse_population
```

```bash
python reproduction/admouse/draw_figures.py \
  --data-dir data/admouse --panels b f g --output-dir outputs/admouse_main
```

The first command counts the saved classifier labels and draws S26.
The second draws Figure 6b, recalculates the Trem2 cell-type composition,
and draws Figure 6f–g. To generate new populations from the 0.015 model,
use the [AD dataset notebook](../../docs/tutorials/dataset_workflows/admouse.ipynb).

## NicheNet and GO

For NicheNet, `prepare.py` reconstructs expression from 51 saved states,
`score.R` calculates ligand activities for 50 adjacent intervals, and
`plot_activity.R` and `plot_links.R` draw S29. The data are in
`admouse_analysis_data.zip` and `admouse_nichenet_data.zip`.
The NicheNet R functions and license are in `nichenet/nichenetr_R/`.

The GO script writes gene-ID mappings, the enrichment table, the fitted
enrichment object, package versions, and five plots. It uses clusterProfiler
and org.Mm.eg.db. See the guide for the tested versions and R dependencies.

## Original figure source

`final_figures/` contains the supplied scripts and small numerical tables.
Those scripts retain their original directory assumptions. The commands
above use explicit input and output directories and leave the source data
unchanged.
