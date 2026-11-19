# Main figures

Only the five retained PNG figures are in `figures/`.  Each PNG stem has one
corresponding script in `scripts/` and a same-stem directory in `data/`.

| Figure PNG | Script | Data directory |
|---|---|---|
| `ad_main_figureb.png` | `scripts/ad_main_figureb.py` | `data/ad_main_figureb/` |
| `ad_main_figurecd.png` | `scripts/ad_main_figurecd.py` | `data/ad_main_figurecd/` |
| `ad_main_figuree.png` | `scripts/ad_main_figuree.py` | `data/ad_main_figuree/` |
| `ad_main_figuref.png` | `scripts/ad_main_figuref.py` | `data/ad_main_figuref/` |
| `ad_main_figureg.png` | `scripts/ad_main_figureg.py` | `data/ad_main_figureg/` |

## Figure 6f

Extract `admouse_perturbation_data.zip` into the data directory, then run from
the repository root:

```sh
python reproduction/admouse/draw_figures.py --panels f --data-dir data/admouse --output-dir outputs/admouse_figure6f
```

This writes `ad_main_figuref.pdf`, `ad_main_figuref.png` and
`tables/whole_tissue_composition_endpoint.csv`. It reads saved particle states,
cell-type labels and edge arrays from `data/admouse/trem2/figure_source`.

The spatial maps highlight neurons and microglia. Other cell types are grey.
The stacked bars show whole-tissue cell-type proportions.
