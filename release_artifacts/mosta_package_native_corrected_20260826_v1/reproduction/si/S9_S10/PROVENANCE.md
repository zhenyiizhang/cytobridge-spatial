# MOSTA SI Figures S9-S10 provenance

## Scope

- Panels: S9a-b and S10a-d.
- Numerical truth: corrected MOSTA package-native results from the accepted server retraining and corrected downstream S8/S10 analyses.
- Style truth: the submitted SI S9/S10 panels and their historical MOSTA plotting grammar.
- No ARISTA data, labels, biological logic, numerical results, or dataset-specific styling were used.
- No rotation, stretch, warp, coordinate remapping, or image-level geometry manipulation was applied.

## Package and model authority

- Package release: `/data/cytobridge/projects/CytoBridge-ST-1104/software/cytobridge-release-2b3c79e-runtime`
- Package commit: `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`
- Package archive SHA-256: `06852992db0d8ebedd8f0baa19a3f539bcdd271dfc0c6fae9774dd553c8fe55e`
- Accepted training directory: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta/training`
- Finetune weight SHA-256: `d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5`
- Score-refine weight SHA-256: `d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a`
- Shared corrected computation: `si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`; 50,000 trajectories are initiated from the global t0 distribution, with no observed-time restart.

## S9/S10 numerical contract

- S9 queries are the exact corrected S8 Ward-k2 Brain programs: 883 genes in Pattern 1 and 1,117 genes in Pattern 2.
- S10 uses the exact corrected three-phase dynamic-programming segmentation of the peak-ordered 1,000 Brain profiles: boundaries `[0,483)`, `[483,782)`, `[782,1000)` and sizes 483/299/218.
- GO universe: the same 2,000 eligible Brain original-HVG symbols used by corrected S8; 1,725 map uniquely to Entrez IDs.
- Enrichment: server R 4.3.3, clusterProfiler 4.10.0, org.Mm.eg.db 3.18.0, ontology `ALL`, pooled BP/MF/CC, `minGSSize=5`, `maxGSSize=500`.
- Multiple testing: one Benjamini-Hochberg family across pooled BP/MF/CC terms for each query.
- Display: all and only `p.adjust < 0.05`, ranked by `p.adjust`, `pvalue`, descending Count, then Description; at most 20 terms; no manual term selection or filler.
- S9 Pattern 2 therefore contains 11 bars because only 11 pooled terms pass FDR 0.05.

## Independent audit

- Status: PASS; zero audit errors.
- S10 DP3 objective: `70.34855609421834`; assignments and profile order match exactly.
- Ordered z-score maximum absolute reconstruction error: `3.4468544828358816e-06`.
- Independently reproduced hypergeometric p-values and pooled BH adjusted p-values agree with clusterProfiler to numerical precision (maximum absolute errors below `2.4e-14`).
- Corrected late Phase 3 is led by cell-periphery/adhesion/ECM terms. `nervous system process` remains significant (`p.adjust=0.0003012730274171`) but is not the leading pooled GO-ALL term; the plot does not force the legacy synaptic interpretation.

## Rendering contract and QA

- S9 page: 7.31 x 11 in (526.32 x 792 pt), matching the submitted compact panel geometry.
- S10 page: 7.98 x 11 in (574.56 x 792 pt), matching the submitted compact panel geometry.
- GO grammar: horizontal Count bars, raw `p.adjust` color scale, submitted five-stop teal-to-coral palette `#264653/#2A9D8F/#E9C46A/#F4A261/#E76F51`, filename-form titles, light gray grid, and original panel arrangement.
- S10a grammar: peak-ordered viridis wave map, z-score limits -1.8 to 1.8, and original blue/orange/green phase strip.
- Corrected GO labels are longer than the legacy labels; S10b-c use the smallest necessary line wrapping and label-size adjustment within the unchanged submitted axes boxes to prevent collisions.
- PDF and SVG retain vector geometry; the final PDFs contain no embedded raster images and embed Arial/Arial Bold subsets.
- Both final PDFs were rasterized at 200 dpi and visually inspected after generation.

## Archived contents

- `figures/`: final PDF/SVG/PNG for S9 and S10.
- `qa/`: actual 200-dpi PDF renders used for visual QA.
- `numerical_inputs/clusterprofiler_server_run/`: complete immutable server clusterProfiler inputs, outputs, R source, session information, OrgDb metadata, and manifest.
- `numerical_inputs/shared_s10/`: corrected S10 profiles, assignments, prototypes, diagnostics, settings, and temporal-variable tables.
- `audit/`: independent numerical audit and query metrics.
- `source/`: server R computation, independent Python audit, and final exact-style renderer.
- `style_references/`: submitted S9/S10 compact panel references.
- `render_manifest.json`, `MANIFEST.json`, and `CHECKSUMS.sha256`: identities, contracts, and archive integrity.

## Source paths

- Final renderer: `source/render_mosta_s9_s10_clusterprofiler_exact_submitted_style.py`
- Server GO computation: `source/run_mosta_s9_s10_clusterprofiler.R`
- Independent numerical audit: `source/audit_mosta_s9_s10_clusterprofiler_and_dp3.py`
- Corrected GO run: `numerical_inputs/clusterprofiler_server_run/`
- Corrected wave inputs: `numerical_inputs/shared_s10/`

## Rebuild

Run the archived renderer with `numerical_inputs/shared_s10/` exposed as the `s10_developmental_wave` child of a shared-run directory, `numerical_inputs/clusterprofiler_server_run/` as the clusterProfiler run, and the two archived submitted JPEGs as style references. The exact absolute commands and source identities used for this archive are retained in `render_manifest.json`; numerical correctness must remain PASS in `audit/numerical_audit.json` before accepting a rebuild.
