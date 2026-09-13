# Drawing Figure 5a

Figure 5a uses the populations and communication matrices calculated in the
ARISTA dataset notebook. The plotting function reads `slice_data/time_*.h5ad`,
`all_time_communications.pkl` and `fixed_particle_lineage_labels_unsmoothed.npz` from that
notebook's population output directory. It calculates cell-type anchors and
lineage links, then draws the five spatial planes. The coordinates are the
simulated coordinates, without spatial warping.

Figure 5a and Supplementary Figure S21 use direct classifier predictions
without spatial label smoothing. Spatial maps retain their existing annotations.

From the repository directory, after running the ARISTA dataset notebook:

```bash
python -m reproduction.arista.main_figure \
  --data-dir outputs/arista/populations \
  --output-dir outputs/arista/figures/main5 \
  --panels a
```

This writes the interactive HTML, the exported PDF before adding labels,
the finished panel PDF and PNG, and the population counts. The panel uses
`stack_style.json` and `data/label_to_color.json`. The camera, curve separation
and display offset are set in `draw_stack` in `main_figure.py`.

## PDF export

The September 2026 manuscript export used Linux, Python 3.10, Plotly 6.5.2,
Kaleido 1.2.0 and Chrome for Testing 135.0.7011.0. Set `BROWSER_PATH` to that
Chrome executable before running the command above. The export scale is 3
on the 3507 by 2481 logical canvas. Other browser versions can change the
appearance of the 3D points and lines even with the same plotting parameters.

`pdf_layout.py` places the exported PDF directly below the labels in
`data/figure5a_labels.pdf`. It does not convert the PDF to a bitmap or resize
its image layer. Plotly's 3D layer is raster inside the PDF. The labels and
arrows remain vector objects. The PNG is produced only after PDF assembly.

To repeat only the layout step on a newly exported PDF:

```bash
python -m reproduction.arista.pdf_layout \
  --source-pdf outputs/arista/figures/main5/Figure5a_calculated_core.pdf \
  --output-pdf outputs/arista/figures/main5/Figure5a_spatiotemporal_map.pdf
```

Use the finished PDF for manuscript assembly. Do not paste a screenshot or
transform a PNG before assembling the figure.
