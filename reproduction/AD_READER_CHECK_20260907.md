# AD numerical reproduction check — 7 September 2026

The new `admouse/calculate_programs.py` and `admouse/perturbations.py` were run
on the analysis server with the distributed AD aligned data, model and
classifier. No model training or scientific parameter change was made.
New results remain separate from the original paper inputs.

| Calculation | Comparison with paper inputs |
| --- | --- |
| Microglial profiles | 347 genes × 26 model times, maximum absolute z-score difference 1.33e-15 |
| LR time courses | 182 pair/time rows, maximum absolute score difference 8.55e-10 |
| Four gene programs | Membership counts 127, 63, 8 and 149 |
| Trem2 module contrasts | Maximum absolute difference below 4.9e-8 |
| Spp1 endpoint contrasts | All ten displayed rows compared, maximum absolute difference 8.95e-8 |

The Spp1 source clips reconstructed log-expression at zero and computes
per-cell scores using pooled first and second moments. The Trem2 source uses
unclipped reconstructed expression and the equivalent population mean/variance
calculation in float64. The implementation retains these respective choices.
An initial Spp1 calculation without clipping was rejected, not published.

The resulting Figure 6c–e, f–g and S30 plotting commands were run using the
new tables. The two website examples were copied from those generated plots,
not from an earlier figure image. Numerical input tests also cover the active
gene selection and the two normalization calculations.

Server run directory:
`runs/package-reader-repair-20260907/admouse/` under the study project.
Final Spp1 inputs are in `perturbations/spp1_clipped/`.
The earlier `perturbations/spp1/` directory is not used for publication.

The R GO and NicheNet stages retain their previously archived calculation
scripts. Their dependencies and input/output sequence were reviewed in this
pass. They were not rerun as part of this numerical check.
