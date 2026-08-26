# MOSTA Supplementary Figure S11 provenance

## Source paths

- Corrected seed-42 component audit: `output/mosta_si_s11_component_decomposition_audit_20260826_v1/server_download/si-s11-component-decomposition-2b3c79e-20260826-v1`
- Sampling stability audit: `output/mosta_si_s11_msum_seed_stability_20260826_v1/server_download/si-s11-msum-seed-stability-2b3c79e-20260826-v1`
- Submitted style notebook: `output/jupyter-notebook/mosta-lr-interaction-timecourse-review.ipynb`
- Submitted SVG style oracle: `output/jupyter-notebook/figures/mosta-lr-interaction-timecourse-review/top_lr_small_multiples_dense.svg`
- Selection script: `source/select_s11_msum_stable_representative31.py`
- Render script: `source/render_s11_msum_exact_submitted_style.py`

## Numerical contract

The displayed values use the accepted MOSTA model (`2b3c79e` release), sealed 50k global-t0 fully generated states, inverse-PCA count-space expression at all seven half-step times, strict mouse ligand–receptor complexes, and package-native `M_sum`. The old notebook's per-time maximum normalization is not used. Clustering is package min–max normalization followed by average-linkage exact k=3 with peak-time ordering.

The choice of `M_sum` restores the submitted panel's total cell-type communication estimand. The rejected `M_per_source` candidate changed the estimand and amplified the global communication decline into synchronized min–max pulses. Seeds 42, 43, and 44 produced median per-pair profile correlations of 0.9993–0.9996 and ARI values of 0.86–0.93; every displayed pair kept the same peak-ordered cluster across all three seeds.

## Rebuild

1. Run `source/select_s11_msum_stable_representative31.py` with the sealed component and seed-stability roots.
2. Run `source/render_s11_msum_exact_submitted_style.py` with the generated selection and the pinned notebook/SVG style oracles.
3. Render the PDF with `pdftoppm -png -r 180 -singlefile`.
4. Validate using the CytoBridge scientific-figure bundle validator.

## SHA-256

- PDF: `e3aed96eacb8e14019a181d32f070158dfec19c283ed826cda7bf51056570484`
- SVG: `21f9290ef8d3622b2cd3a601dfc89c66930f1895cbe6e41959263fefb8a1ba67`
- Selection CSV: `80486a4d679b1ee2af14f7c7738719bbde52dcdd2d174fc4bc1f1eecfd144b2d`
- Corrected component manifest: `538f99871abcaec2b85e70fac7081d6f5dd1bd5998200621ffb64b5db2f64893`
- Seed-stability manifest: `729971c82a240e4336ddaefa6663273df9abf910aefe7d10f7561c2688d6e3ee`
- Style notebook: `255f96c8c572898f460cca33a1f7b6ea7bf385a5d2ae77b440d285f871c0e4e0`
- Style oracle SVG: `ab461c87c6b15353e7c71a1c582bb8a489b96b339d4cf21bf8494071e31ff010`
