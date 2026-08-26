# Figure provenance — Supplementary Figure S7

## Source paths

- Submitted SI style authority: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`, page 33.
- Submitted MOSTA plotting notebook snapshot: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s7_lineage_20260826_v2/source/mosta_lineage_sankey_3d.ipynb`.
- Submitted Sankey PDF style reference: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s7_lineage_20260826_v2/source/submitted_lineage_sankey_style_reference.pdf`.
- Submitted Sankey SVG style reference: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s7_lineage_20260826_v2/source/submitted_lineage_sankey_style_reference.svg`.
- Historical generic Plotly Sankey helper used by the submitted MOSTA notebook: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s7_lineage_20260826_v2/source/arista_helpers_style_authority.py`.
- Package workflow wrapper snapshot: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s7_lineage_20260826_v2/source/cytobridge_workflows_style_wrapper.py`.
- Original MOSTA categorical palette: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s7_lineage_20260826_v2/input/label_to_color.json`.
- Corrected persistent-particle labels: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/s7_lineage/fixed_particle_labels.csv.gz`.
- Corrected lineage contract, nodes, and edges: the adjacent files `lineage_contract.json`, `lineage_nodes.csv`, and `lineage_edges.csv` in that same immutable shared-compute directory.

## Experiment or command

Numerical truth was computed on the server in the immutable run:

`/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`

The calculation used package release commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`, the accepted MOSTA model, the accepted classifier with `k=10`, 50,000 initial particles, and seed 42. All displayed lineages use the same persistent particle IDs propagated from global t0. There is no observed-stage restart and no spatial warp.

Local numerical audit:

```bash
python output/mosta_si_s7_lineage_20260826_v2/source/audit_s7_latest_package_fixed_particle_lineage.py
```

The exact old Plotly style was rendered on the server because the local Plotly 5.9.0/Kaleido 1.2.0 pair is incompatible. The immutable rendering run is:

`/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s7-lineage-render-oldstyle-plotly652-kaleido021-arial-2b3c79e-seed42-20260826-v2`

It used Python 3.10.12, Plotly 6.5.2, and Kaleido 0.2.1. Fontconfig was pinned to the server's established paper-font cache, with SHA-verified Arial regular, bold, italic, and bold-italic files; `fc-match` resolved Arial directly to the pinned `Arial.ttf`. The renderer snapshot is `source/server_render_s7_exact_old_plotly_style_arial.py`.

## Manuscript protocol

- Geometry and layout are copied from the submitted MOSTA notebook: a seven-column Plotly Sankey at model times 0, 0.5, 1, 1.5, 2, 2.5, and 3.
- The old filter is retained exactly: for each interval and source label, keep the smallest descending set of outgoing flows whose cumulative source fraction reaches at least 0.8. This displays 343 of 1,380 corrected edges.
- No additional minimum-flow cutoff or normalization is applied.
- The title remains `Cell Fate Transitions`; the old bottom time axis, white background, true embedded Arial typography, node padding/thickness/borders, source-colored links at alpha 0.4, and the original MOSTA cell-type palette are retained.
- The canvas is the submitted 1600 × 1000 px geometry, exported to the same 1200 × 750 pt PDF page.
- No smoothing, relabeling, rotation, stretching, or coordinate warp is applied.
- The PDF and SVG preserve vector geometry; the PNG is a preview only.

The helper file retains its historical `arista_helpers.py` name because the submitted MOSTA notebook imported its generic `plot_sankey` function. No ARISTA data, labels, palette, model output, or biological-analysis logic is used. All numerical inputs and categorical colors in this figure are MOSTA-specific.

## Validation status

- Numerical audit: PASS.
- The table contains exactly 50,000 persistent particle IDs at each of seven times (350,000 rows total).
- The provided nodes and edges are reproduced exactly from the particle-label table.
- Source and target mass are conserved exactly, and each of the six intervals contains 50,000 transitions.
- The minimum displayed outgoing coverage under the submitted 80% rule is exactly 0.8; every source meets the rule.
- Same-label fractions across successive intervals are 0.74292, 0.68268, 0.67378, 0.68922, 0.73264, and 0.73486.
- Actual page-1 PDF was rendered at 180 dpi and visually compared with the submitted PDF reference: same page geometry, node/link visual grammar, palette, title, time axis, and no clipping or transform artifact.
- The final PDF contains no embedded raster images and embeds `ArialMT`, matching the submitted style-reference PDF. The earlier Liberation Sans fallback candidate is rejected and is not a deliverable.
- Server output checksums were verified locally against the sealed `SHA256SUMS.txt` file.

## Interpretation

The corrected Sankey preserves the submitted lineage-transition message without using disconnected stage-wise samples. Brain is predominantly self-maintaining, while Cartilage primordium diversifies late: at model time 2.5→3 its leading targets are Cartilage (27.15%), Cartilage primordium (19.73%), and Connective tissue (18.95%). This is a direct consequence of the corrected persistent-particle trajectories, not a display-driven reassignment.

## Rebuild

From `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104`, run the numerical audit and require `status: PASS`:

```bash
python output/mosta_si_s7_lineage_20260826_v2/source/audit_s7_latest_package_fixed_particle_lineage.py
```

Static Plotly rendering must use the recorded server environment and a new immutable server output directory. Do not overwrite the sealed run. Render the resulting PDF itself for visual QA rather than inspecting only the direct PNG export.

## Deliverables

- Interactive HTML: `figures/Figure_S7_MOSTA_latest_package_global_t0_fixed_particle_k10_exact_old_plotly_style_Arial.html`
- Vector PDF: `figures/Figure_S7_MOSTA_latest_package_global_t0_fixed_particle_k10_exact_old_plotly_style_Arial.pdf`
- Vector SVG: `figures/Figure_S7_MOSTA_latest_package_global_t0_fixed_particle_k10_exact_old_plotly_style_Arial.svg`
- Preview PNG: `figures/Figure_S7_MOSTA_latest_package_global_t0_fixed_particle_k10_exact_old_plotly_style_Arial.png`
- Numerical audit: `tables/s7_numerical_audit.json`
- Key-source interpretation table: `tables/s7_key_source_top_targets.csv`
- Server render manifest: `server_download/si-s7-lineage-render-oldstyle-plotly652-kaleido021-arial-2b3c79e-seed42-20260826-v2/manifest.json`

## SHA-256 identities

- Corrected persistent-particle table: `667d19659ffb7cab18caed14e543e5d03dc57c0697f48557728fe0bc9af003cb`
- Lineage contract: `0633c5b831fe4ae16bdfbe71a9c4acd6bacc1131f43f5984c549b678c4cb1cf3`
- Corrected edges: `e3b8e2be6c1a1434d7c841a986ef4182af8e60c555c30c3a9c36aede8a52fb13`
- Corrected nodes: `3aa567a993db4736790b8354ee9035a7a6a9a8f99bdebdcf7bf67acd8536fa7b`
- Numerical audit: `374ee1e84ef8b25050deffcd0f17f3595dc2aaf5ee7c381a129aa666eb31966c`
- Submitted notebook snapshot: `bb659054242ea1a5eed173fe917460591711f6f9de0bc9bfd4d0a48232c0efce`
- Generic historical style helper: `128be4a718d8ca43575e7b944a6f06c59fd9cd29cfc0b668a2ad5c19c9ff10d6`
- Package workflow wrapper: `889dd4a7e70b646535a3a606b8c58fbceec028aa8886e9e025c5ab2f97984539`
- Submitted PDF style reference: `9c540022dfaa3b0a9223db15c3d648da51b136a670ed5dd225b383ea4ac0ebf7`
- Submitted SVG style reference: `d8c8063a7f0e1ad240bfef0b309e486bda808496838f81b3dbb00d87f8dd1a53`
- MOSTA palette: `7e95e868e0a6ecd4a2ed13b57e6a8223e77e2302a0f9634ca30f41390c040b71`
- Arial regular: `525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9`
- Arial bold: `d72db21f9242aedd6b917d8549ad5921766b24d5f8d0becfda2ff4c620b3c2e0`
- Arial italic: `ce1d2f1ab89db45f9796100eee960f5702a40e84c225c2b48c3ec3e81d153f98`
- Arial bold italic: `374b0190a9844343110d8f8ed1818117a4591803d022bbb2bd189d63a681e731`
- Server renderer: `01ae2e333482fbde00261fda283c6486c2b1124c10cd854a9dd9ed0480eecd7e`
- Server render manifest: `9d69d1efcd7dc80676038864227eed27a9f904cee7e2bc14249d33e87ca7d0df`
- HTML: `e1c4bfcd9a39c138360318be4e71972314ab62fb406c5428c7d88e04d1c624a7`
- PDF: `2f8e498e4161324d7655142024ee14a88b96f6a105dba091783eb5a19950845c`
- SVG: `835f92d41db94c98bc9f86c6d57def3ecd3d6e63b9373b991fb6447fbf512542`
- PNG: `3d4b38d1e661769fcdc4b8ae95e9785d3a981faced9dc9279650d64f7ada6c1a`
- 180-dpi rendered PDF QA image: `9f0bc2174902f1e096a8b1d25c46f2157a78179ac4bab74a71f88b1177bfff77`
