# Downstream analyses

## Velocity

Interaction velocity is recomputed independently within each real time slice.
The workflow does not reuse historical cross-time interaction caches.

## Growth

Growth values use the pre-warp joint state. `growth_alpha=1.0` is explicit in
the formal workflows and is applied consistently to AnnData and dataframe
simulation paths.

## Sparse communication

Spatial radius pairs are constructed without a dense cell-by-cell matrix.
Edges retain sender/source to receiver/target direction, and attention is
aggregated into cell-type tables without changing the underlying model output.

## Ligand–receptor projection

Generated inverse-PCA log1p expression is clipped to non-negative values per
cell before scoring. A complex is eligible only when every required subunit is
present. The formal score uses the minimum across subunits; geometric mean is a
sensitivity option.

## Gene and module dynamics

Gene reconstruction uses the fitted PCA loadings and center. Genes with
inactive PCA loadings are reported separately instead of diluting module means.
Both non-negative expression and signed reconstruction diagnostics can be
retained, but formal expression summaries use the non-negative values.

## Figures

Computation produces tables first. Plotting helpers then apply the paper panel
order, palette, typography, legends, and optional display warp. Scientific
values are never recovered by reading rendered images.
