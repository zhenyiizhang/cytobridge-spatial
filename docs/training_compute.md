# Training curves and compute cost

The authoritative matched matrix completed all 12 requested runs (four
datasets × full, no-LR-prior, and no-interaction) and their package downstream
chains. All 12 profiles and all four three-arm families pass acceptance
SHA-256
`c4f8e203e2da73fe78e28525516bbec192d3cbbd35d423dcd64080a0f83a10df`.
Missing historical telemetry remains `NA`; it is not inferred from file sizes,
GPU model, or neighboring runs.

The table reports the measurements in each matched
`training_run_summary.json`. Wall time covers `TrainingPipeline.train` only and
excludes downstream inference, evaluation, and AnnData serialization. CPU max
RSS is the process-lifetime high-water mark sampled after training; CUDA is the
largest per-stage peak allocation.

| Dataset | Arm | Retained epochs | Wall time (s) | Peak CPU RSS (MiB) | Peak CUDA allocation (MiB) | Downstream / acceptance |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Zebrafish | full learned prior | 5,252 / 5,252 | 1,239.226 | 2,174.1 | 577.8 | complete / PASS |
| Zebrafish | no LR prior (`all_spatial`) | 5,252 / 5,252 | 1,300.973 | 2,153.4 | 4,726.7 | complete / PASS |
| Zebrafish | no interaction | 5,252 / 5,252 | 805.898 | 2,040.4 | 207.9 | complete / PASS |
| MOSTA | full learned prior | 5,252 / 5,252 | 1,997.504 | 17,023.8 | 1,051.0 | complete / PASS |
| MOSTA | no LR prior (`all_spatial`) | 5,252 / 5,252 | 2,009.665 | 17,010.7 | 1,143.7 | complete / PASS |
| MOSTA | no interaction | 5,252 / 5,252 | 1,529.482 | 16,912.5 | 458.7 | complete / PASS |
| ARISTA | full learned prior | 5,252 / 5,252 | 1,310.566 | 3,310.8 | 919.3 | complete / PASS |
| ARISTA | no LR prior (`all_spatial`) | 5,252 / 5,252 | 1,289.641 | 3,295.9 | 3,040.4 | complete / PASS |
| ARISTA | no interaction | 5,252 / 5,252 | 829.866 | 3,180.7 | 321.4 | complete / PASS |
| AD mouse | full learned prior | 7,252 / 7,252 | 696.454 | 2,446.7 | 708.6 | complete / PASS |
| AD mouse | no LR prior (`all_spatial`) | 7,252 / 7,252 | 771.601 | 2,429.7 | 964.2 | complete / PASS |
| AD mouse | no interaction | 7,252 / 7,252 | 503.421 | 2,318.6 | 340.9 | complete / PASS |
| Chicken heart | full learned prior | 5,252 / 5,252 | 738.557 | 2,115.5 | 1,435.5 | complete / single-profile contract |

Neural-ODE and score-matching losses optimize different objectives and must not
be joined into one continuous numeric scale. Every matched run has a complete,
finite six-stage history; the earlier sparse MOSTA/ARISTA histories and the old
AD timers are legacy provenance, not the current matched evidence.

The no-interaction arm retains velocity, growth, and score but has no
communication or LR analysis by construction; those outputs are `NA`, not
failed or zero-valued measurements. The no-LR-prior arm retains interaction and
changes only its gate to `all_spatial`.

Separately instrumented Zebrafish OT:mass sensitivity fits took about 1,420–
1,455 seconds with about 2,234 MiB peak CPU RSS and 899–1,027 MiB peak CUDA
allocation. These legacy diagnostics are not substituted for the matched-grid
measurements above.

The original Heart Figure 3 run remains historical and is not used as current
evidence. The table instead records the current package-native chicken-heart
full fit, whose six-stage timing and memory telemetry are present in its
`training_run_summary.json`. It is outside the four-dataset matched three-arm
chain because no matched chicken-heart no-LR/no-interaction retraining was
requested.

The machine-readable matched table is available as
{download}`formal_training_compute_cost.csv
<data/formal_training_compute_cost.csv>`.
