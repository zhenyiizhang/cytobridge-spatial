# Historical artifact compatibility

The four formal alpha-expression `0.015` checkpoints can be loaded by the
current package without changing their fixed-batch inference. This was tested
on the private compute server against the exact source tree used for each
formal run, not against a second invocation of the current loader.

No model training, rollout, classifier fitting, or downstream reanalysis was
run for this check. Each historical loader and the current package loader read
the same aligned H5AD, model directory, edge predictor, and species-matched
ligand-receptor database. They then evaluated the same 32-cell, 52-dimensional
CPU input batch.

| Dataset | Weight checkpoint | Score checkpoint | Retained interaction edges | Maximum absolute difference |
| --- | --- | --- | ---: | ---: |
| Zebrafish | `Finetune/best_model.pth` | `Score_Refine/score_model.pth` | 62 | 0 |
| MOSTA | `Finetune/best_model.pth` | `Score_Refine/score_model.pth` | 9 | 0 |
| ARISTA | `Finetune/best_model.pth` | `Score_Refine/score_model.pth` | 258 | 0 |
| AD mouse | `Finetune/last_model.pth` | `Score_Refine/score_model.pth` | 7 | 0 |

The comparison covered velocity drift, growth, score potential, score
gradient, direct interaction output, grouped interaction output, interaction
edge indices, and their combined drift. The acceptance tolerance was
`atol=1e-6`, `rtol=1e-5`; all four datasets passed with exact CPU equality.
The AD mouse batch was anchored on a real positive edge from its formal LR
graph so the comparison exercised the learned edge gate rather than accepting
an empty interaction graph.

The loaded configs were also checked for input dimension, random seed,
training weight, spatial weight, interaction cutoff, edge threshold, and the
selected weight and score stages. The current artifact-reuse CLI plan reports
all four downstream inputs as ready. Its classifier policy is `k=10` for
Zebrafish, MOSTA, and ARISTA, and `k=1` for AD mouse.

This result has a narrow interpretation: the current package preserves the
formal checkpoint's model inference when it reuses that checkpoint and its
matched inputs. A corrected raw-H5AD run retrains preprocessing, the edge
predictor, and the main model, so it is not expected to reproduce an old
checkpoint bit for bit. Corrected classifier, communication, gene/LR dynamics,
and figure code may also intentionally change historical downstream labels or
summaries.

The executable check is
`scripts/verify_historical_artifact_compatibility.py`. Copy
`scripts/historical_artifact_compatibility.example.json`, replace its readable
artifact paths and expected scientific values, and pass that matrix to the
runner. Private server paths are deliberately not part of the public release.
