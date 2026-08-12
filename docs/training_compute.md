# Training curves and compute cost

This page reports only measurements preserved by the completed runs. Missing
historical memory or per-epoch telemetry is shown as `NA`; no value is inferred
from file sizes, GPU model, or neighboring runs.

Retained training evidence differs by dataset. Missing epochs or historical
memory values are not reconstructed.

| Dataset | Planned epochs | Retained loss records | Wall time | Peak RSS / VRAM |
| --- | ---: | ---: | ---: | --- |
| Zebrafish | 5,252 | 5,252 complete | 1,400.264 s filesystem span | NA / NA |
| MOSTA | 5,252 | 127 exact sparse points | 1,830.191 s filesystem span | NA / NA |
| ARISTA | 5,252 | 127 exact sparse points | 1,308.484 s filesystem span | NA / NA |
| AD | 7,252 | 7,252 complete | 917 s recorded timer | NA / NA |
| Heart, original Figure 3 | 3,501 | unavailable | NA | NA / NA |

MOSTA and ARISTA retained losses every ten epochs for neural-ODE stages and the
final loss for each score-matching stage. These are valid observed points, not
complete per-epoch curves. Neural-ODE and score-matching losses optimize
different objectives and must not be joined into one continuous numeric scale.

The accepted formal runs did not record process-level peak RSS or CUDA peak
allocation. Total server memory, present GPU occupancy, or later replays are not
substitutes; the historical peak values remain `NA`.

Separately instrumented Zebrafish OT:mass sensitivity fits took about 1,420–
1,455 seconds with about 2,234 MiB peak CPU RSS and 899–1,027 MiB peak CUDA
allocation. The 344,603-cell MOSTA downstream replay took 1,470.846 seconds but
did not retain usable GPU-memory telemetry.

The original Heart Figure 3 run and later Heart-v2 benchmark runs are distinct
contexts. Heart-v2 metrics cannot be used as provenance for the original figure.

The compact source table for this page is shipped as
{download}`formal_training_compute_cost.csv
<data/formal_training_compute_cost.csv>`. It intentionally contains `NA` for
unrecorded peak memory instead of inferred values.
