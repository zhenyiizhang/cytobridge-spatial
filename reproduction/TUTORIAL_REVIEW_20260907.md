# Numerical-input review — 7 September 2026

Maintenance record. This is not a public tutorial or a package-wide completion
statement. Apply [the tutorial review rules](TUTORIAL_REVIEW.md) before editing
the pages below.

## The recurring error

The Figure 4 page linked a general MOSTA tutorial but loaded precomputed
population, communication and lineage files. Its download instructions did
not explain how to calculate those files. The user had repeatedly requested
the calculation, not just an archive and a redraw command.

The repair must include the actual producer, its measured inputs and selected
model, its returned object or saved filename, and the plotting call that reads
that result. Adding another link does not complete this repair.

## Changes checked in this review

- Figure 4 now calculates populations, communication, LR scores, fixed-particle
  lineage and velocity fields from the measured inputs and trained model.
  The complete notebook ran in a fresh server kernel. No training was rerun.
  All seven population counts and labels matched the paper inputs. The 1,282
  selected lineage identities and destination labels also matched. LR score
  differences were below `9e-10`, and velocity-array differences below `2e-7`.
  The trajectories are not byte-identical: at E15, seven of 100,871 population
  cells differ in a spatial coordinate by more than 0.001, with maximum 0.00404.
  The separate lineage background has a maximum coordinate difference of
  0.0292; selected-lineage coordinates differ by less than `4.1e-6`.
  Do not report these fresh stochastic/numerical runs as exact array identity.
- Figure 5c/e, S2–S5 and S25 now pass the newly calculated objects to the
  accepted plotting functions. Their notebooks ran in fresh server kernels
  using the paper's saved numerical inputs. This is not a test of new training.
- S39, S43 and S44 now pass the selected input directories throughout plotting.
  S39's separate COMMOT directory is explicit. All eight cells across these
  three notebooks ran in fresh kernels, and their PNG outputs matched the
  accepted archive byte for byte. External methods were not rerun.
- S39 now has a collector connecting the actual analysis outputs to the plot
  input format. Its collector-to-plot route was executed separately.
- ARISTA's population generator now exports its returned states to `slice_data`
  and calculates communication from its model populations. The export and
  settings have focused tests. The complete GPU simulation was not rerun.
- Figure 4's generator matches the published cells in a regression test.
  ARISTA generator calls no longer restore the adjusted Figure 5b coordinates
  or the disconnected Figure 5c/e calculation.

## All 27 notebooks

Every row was reviewed in source together with the called implementation.
“Executed” below refers only to this review. Earlier execution records are not
treated as a new run. A connected default route may still have an untested
optional training or custom-data branch.

| Notebook under `docs/tutorials/` | Calculation route and next action |
| --- | --- |
| `paper_figures/main_figure_4.ipynb` | Executed from measured inputs and model weights through all five panels. New outputs stay separate from the paper inputs. |
| `paper_figures/main_figure_5.ipynb` | Executed from saved numerical inputs. c/e object continuity fixed. Test the new population producer and its connection to a/b with an actual model run. |
| `paper_figures/main_figure_2.ipynb` | a–d still use included artwork. Connect the existing numerical analysis to these panels. The separate e calculation does not yet feed this page. |
| `paper_figures/mosta_figures.ipynb` | Current calculation and explicit trajectory-file continuation are connected. Retire or update the older notebook generators so they cannot overwrite this route. |
| `paper_figures/arista_figures.ipynb` | S19 has explicit population inputs. S20 and S22–S24 still need the model-to-table continuation checked and completed. |
| `paper_figures/arista_local_domains.ipynb` | Executed from saved numerical inputs; calculated objects now reach S25. Provide the domain/control generation and six-file export from a new model run. |
| `paper_figures/agist_figures.ipynb` | Executed from saved numerical inputs; calculated objects now reach S2/S3. Connect newly generated simulation/training results to the compact input format. |
| `paper_figures/nonspatial_figures.ipynb` | Executed from saved numerical inputs; calculated objects now reach S4/S5. Complete the trained-model-to-plot-input conversion. |
| `paper_figures/chicken_heart_daily.ipynb` | Reads the stated upstream populations and calculates downstream quantities. Record the exact current paper panels for each displayed output. |
| `paper_figures/classifier_smoothing.ipynb` | S6 needs the actual held-out predictions, generated-frame and fixed-cohort sensitivity producers. The guide currently begins with undefined arrays. |
| `paper_figures/compute_cost.ipynb` | Recorded measurements are formatted correctly. The optional guide's final notebook step resets the input directory; fix it and explain the collector's hardware requirement. |
| `paper_figures/interaction_ablation.ipynb` | Selected files reach S42. Inference-on/off has a producer. Full/No-LR needs complete preparation/training/evaluation commands and portable benchmark paths. |
| `paper_figures/loto_benchmark.ipynb` | Selected files reach S45. Benchmark preparation still reads absolute server paths and needs explicit input/model configuration. |
| `paper_figures/loto_benchmark_summary.ipynb` | Executed from completed comparisons; explicit input reaches S44. The upstream benchmark preparation gaps also apply here. |
| `paper_figures/lr_complex_aggregation.ipynb` | S41 calculates agreement from LR scores. Its existing sensitivity runner needs the preceding downstream workflow command that produces `downstream/summary.json`. |
| `paper_figures/training_histories.ipynb` | S46 has training-history generation, collection and plotting commands using the selected directory. No fresh training run in this review. |
| `paper_figures/zebrafish_attention.ipynb` | Executed from saved analysis outputs; S39 collection and explicit COMMOT input fixed. Full model and external-method inference not rerun. |
| `paper_figures/spatial_communication.ipynb` | Executed from saved method comparisons; explicit inputs reach S43. Complete the external-method-output-to-summary commands. |
| `paper_figures/zebrafish_si_s31_s38.ipynb` | Default model-to-analysis route is connected. Propagate a reader's custom classifier selection to S33/S34/S37 as well as S31/S38. |
| `dataset_workflows/admouse.ipynb` | Training choice, model simulation, cell labels and S26 plotting are connected. The later AD page must consume these outputs for its gene/LR/perturbation analyses. |
| `dataset_workflows/arista.ipynb` | Simulation and growth values reach S20. Resolve its classifier selection against the paper-population route before declaring one common model-to-figure route. |
| `dataset_workflows/chicken_heart.ipynb` | Simulation/growth values reach S9 and save the stated H5ADs for later analysis. Clean up remaining internal wording. |
| `dataset_workflows/mosta.ipynb` | Simulation, classification and growth are calculated; explicit outputs continue to S11–S18. Check the older generator against the current notebook. |
| `dataset_workflows/zebrafish.ipynb` | Model growth evaluation reaches S32. Correct the final description of S31–S38, which now calculates results rather than only drawing saved inputs. |
| `data_preparation/synthetic_preprocessing.ipynb` | Self-contained count generation and preprocessing example; no paper figure or hidden result input. |
| `your_data.ipynb` | Raw-input preparation, configuration, training and downstream command are connected. No new full training test in this review. |
| `model_analysis.ipynb` | Model-derived velocity and growth use the stated inputs. No unrelated scientific figures are displayed. |

The linked Markdown routes were also read. Further work includes portable
S7/S8 input generation and the later AD gene/LR/perturbation calculations.
These entries must not be closed solely because a notebook or Sphinx build
finishes successfully. Do not replace paper figures to make them match an
incorrect tutorial.
