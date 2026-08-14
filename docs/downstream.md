# Downstream analyses

## Zebrafish interval-local simulation scope

The formal Zebrafish preset uses observed-anchored piecewise split-SDE. Real
observed slices remain unchanged, and each generated midpoint is a one-sided
forward simulation starting from the immediately preceding observed slice. It
is not conditioned on the following observed endpoint and therefore is not a
two-endpoint bridge. Communication and ligand–receptor tables combine real
observed stages with these interval-local generated stages; they do not
describe one lineage-continuous rollout or global t0 extrapolation.

The paper-specific S22 reproduction is deliberately different: it generates
one continuous global-t0 fixed-population state transport through `t=4` and
exports observed integer slices only as a separate reference figure. Velocity
drift, score-gradient correction, interaction forces, and stochastic diffusion
remain active. Learned growth-driven birth/extinction is disabled, so N remains
equal to the sampled t=0 cohort. The panel is not a cell-abundance forecast,
adjacent-anchor interpolation, or reconstruction of observed stages. S25 and
communication continue to use the interval-local contract above with learned
growth enabled; a global-t0 S22 bundle cannot be silently reused for those
analyses.

## Velocity

Interaction velocity is recomputed independently within each real time slice.
The workflow does not reuse historical cross-time interaction caches.
The plotted arrows are model-derived state derivatives. Direct 2D derivatives
are interpolated in their existing coordinates; only higher-dimensional state
derivatives pass through scVelo's transition projection. Unsupported grid
locations receive a common two-component mask so finite vector streamlines are
preserved in vector PDF output.

## Growth

Growth values use the pre-warp joint state. `growth_alpha=1.0` remains explicit
for the standard/interval-local workflows and is applied consistently to
AnnData and dataframe simulation paths. Paper S22 is the documented exception:
it hard-codes `growth_alpha=0.0` for fixed-population state transport, while S23
reports the trained growth head separately on observed states.

## Sparse communication

Spatial radius pairs are constructed without a dense cell-by-cell matrix.
Edges retain sender/source to receiver/target direction, and attention is
aggregated into cell-type tables without changing the underlying model output.

Each time point records the number of within-cutoff candidates, the number
retained by the configured edge prior, and their fraction. Under a learned
prior, `candidate_count > 0` with zero retained edges is a valid structural zero
when no candidate passes the frozen LR-informed learned edge-predictor
threshold. If `candidate_count == 0`, the status instead records that there
were no within-cutoff candidates. Neither case establishes the absence of all
biological communication. The formal acceptance check permits such individual
time points only when the sparse arrays are canonically empty and the
trajectory-wide communication output remains non-degenerate. A structural
zero also makes that time point's LR projection zero by construction; edges
must not be added and the frozen threshold must not be changed after the fact.

For AD main, distinguish the downstream reporting graph from the stochastic
interaction groups used by the dynamical model. Downstream constructs the full
radius candidate graph for the analyzed time-slice cohort, or for an explicit
seeded subsample, and applies the learned edge predictor in memory batches. The
predictor is supported by seven strict panel-covered LR pairs, so the result is
a panel-limited spatial-attention summary rather than global CCI inference.

## Ligand–receptor projection

Generated inverse-PCA log1p expression is clipped to non-negative values per
cell before scoring. A complex is eligible only when every required subunit is
present. The formal score uses the minimum across subunits; geometric mean is a
sensitivity option.

The AD panel contains seven complete pairs under this rule. The main workflow
uses their database-derived graph labels to fit its learned edge predictor and
also reports their downstream projection. The separately packaged all-spatial
condition removes this LR-informed gate as a no-LR-prior ablation.

## Gene and module dynamics

Gene reconstruction uses the fitted PCA loadings and center. Genes with
inactive PCA loadings are reported separately instead of diluting module means.
Both non-negative expression and signed reconstruction diagnostics can be
retained, but formal expression summaries use the non-negative values.

## Figures

Computation produces tables first. Plotting helpers then apply the paper panel
order, palette, typography, legends, and optional display warp. Scientific
values are never recovered by reading rendered images.
