# ARISTA package-native paper and reviewer scripts

This directory contains the canonical script snapshots used for the accepted
ARISTA rerun archived under
`release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1`.

The scientific values come from the corrected package-native run. The Figure 5
and Supplementary Figure S12--S17 renderers intentionally preserve the
submitted paper's visual grammar, palette, typography, panel placement, and
scVelo stream style. They do not substitute a newly designed plotting style.

## Scientific pipeline

The package implementation and workflow configuration live in the normal
package locations. In particular:

- `CytoBridge/pp/preprocess.py` implements count-layer validation, package HVG
  selection, latent-feature normalization, and label-blind spatial-outlier QC.
- `CytoBridge/pp/spatial_align.py` forwards the ARISTA fit-scope and spatial-QC
  policy and preserves fixed external spatial references.
- `CytoBridge/workflow_configs/arista.json` records the ARISTA workflow policy.
- `scripts/preprocess_pipeline.py` and
  `scripts/run_spatiotemporal_downstream.py` expose the preprocessing and
  downstream entry points.

The exact fitted configuration, six-stage history, checkpoints, run log, and
downstream outputs are in the archived `main_run` directory. The original
audit-branch commits were `b0a9faa`, `5e66c03`, and `1a2c7de`; these changes are
cherry-picked into the release branch.

## Figure scripts

- `render_figure5a_package_native_legacy_style.py` and
  `preview_figure5a_foreground_lift_html.py`: Figure 5a legacy renderer and the
  audited foreground-z lift. Reciprocal-edge curvature is display-only and
  defaults to 0.06 of the global planar diagonal; edge identities, directions,
  and weights are unchanged. `arista_helpers_focus_anchor.py` is the archived
  historical renderer dependency used by this entry point.
- `extract_figure5b_package_native_points_legacy_style.py` and
  `assemble_figure5b_original_style.py`: generated t=0.5 markers and Figure 5b.
- `build_figure5c_package_native_state.py` and
  `assemble_figure5c_original_style.py`: corrected Figure 5c velocity and
  per-cell cosine state in the submitted layout.
- `build_figure5d_package_native_gene_velocity_state.py` and
  `render_figure5d_corrected_gene_velocity_legacy_style.py`: corrected full
  gene velocity on the fresh package-native PCA geometry.
- `build_figure5e_package_native_state_server.py` and
  `assemble_figure5e_package_native_original_style.py`: Figure 5e growth and
  interaction state.
- `assemble_figure5_fullpage_from_accepted_panels.py`: deterministic assembly
  of the accepted Figure 5 page.
- `build_s12_package_native_warpk1_oldstyle.py`,
  `build_s13_s14_package_native_oldstyle.py`, its archived legacy-style helper
  `build_s12_s14_legacy_style_corrected.py`, and
  `build_s15_s17_strict_legacy_style.py`: accepted S12--S17 replacements.
  S13 can repeat one fixed anatomical injury locator across all nine panels via
  `--annotate-injury-all-panels`; the locator is not inferred from growth values.

## Figure 5c reviewer analysis

`server_analyze_figure5c_two_niche_timecourse.py` fixes the two physical niche
domains from the Figure 5c ROI and evaluates attention organization and LR
programs with cell-type-matched null regions. Pair-level LR axes are computed
by `server_analyze_figure5c_two_niche_lr_axes.py`. The accepted biology-first
figure is rendered by `plot_figure5c_two_niche_reviewer_figure_clean.py`.

The final figure, tables, caption, provenance, reviewer-response summary, and
the Chinese internal interpretation note are archived with the run. Earlier
hotspot/GO prototypes and rejected visual iterations are deliberately omitted.
