# Supplementary figure update, 6 September 2026

This archive contains the accepted style revisions for S2–S7, S25, S34, S36,
and S39–S46, together with the panel-level tables. The combined preview contains
17 figures in SI order.

The plotting code is in
[`reproduction/supplementary_figures`](../../reproduction/supplementary_figures/README.md).
Run it from the repository root:

```bash
python reproduction/supplementary_figures/plot_figures.py --output-dir outputs/supplementary_figures
```

The complete command was executed against the included numerical inputs on
6 September. It generated all 17 figures. Rendered output was compared with the
accepted figures. The calculations and panel layouts are retained.

S2 shows medians. S39a uses the original within-group permutation analysis. S41c
uses the top 100 LR pairs in every dataset. The top-10, top-50 and top-100
comparisons are retained in `tables/`. S42c/d use the same fitted checkpoint with
and without interaction during inference. S44 includes SpaTrack and retains the
Linear OT and random-interpolation controls.

S7 here is the accepted pre-existing experiment. A separate experiment with
integer translation and rotation magnitudes was requested on 6 September.
Those new results are not substituted into this archive before review.

The S10 comparison-region annotation is separate from this style set. Its source
is [`s10_highlights.tex`](../../reproduction/chicken_heart/s10_highlights.tex).
