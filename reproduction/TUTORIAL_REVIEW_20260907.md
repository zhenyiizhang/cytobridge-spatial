# Numerical-input review — 7 September 2026

Maintenance record, not tutorial prose. Apply [the review rules](TUTORIAL_REVIEW.md).
This records actual execution, not a package-wide claim of complete reproduction.

## Repairs and execution

The recurring error was treating an available result table as an explanation of
its calculation. Figure 4 exposed this clearly. These repairs connect input,
model calculation, numerical output and plotting. Paper results were not changed.

- Figure 4 ran from measured inputs and weights through all panels. All seven
  population counts and labels and 1,282 selected lineage identities matched.
  LR differences were below 9e-10 and velocity differences below 2e-7.
  Stochastic coordinates were not identical: E15 maximum difference 0.00404,
  separate lineage background 0.0292, selected lineage below 4.1e-6.
- ARISTA's four notebooks ran in fresh kernels: 21 cells, 13 PNGs, no errors.
  Population counts and labels matched. S25 recovered the same 203/77 domain
  cells and 419/122 edges. Optional new training was not performed.
- AD used 53,615 starting cells and 26 simulated times. Recalculated gene
  profiles differed by at most 1.33e-15, LR by 8.55e-10, and Trem2/Spp1 module
  scores by less than 9e-8. Figure 6b–g/S30 producers and plots ran.
  R GO and NicheNet commands were reviewed but not executed in this pass.
- S2/S3 and S4/S5 notebooks ran in fresh kernels. S3 raw generation reproduced
  both original CSVs and reference NPZ arrays exactly. Five inference seeds ran
  with the original model architecture. S4b evaluated three LR models over five
  groupings each, maximum field difference 5.87e-8. S4/S5 now also have executed
  distribution, clone-fate, new-RNA direction, trajectory and LR-attribution
  producers, with adapters from their actual outputs to the panel tables.
  The new Weinreb/scNT sender-message values match the original tables within
  1.11e-8 and 2.34e-9, respectively. Details of the inference repeats are in
  the non-spatial execution report.
- The final default S4/S5 notebook ran all nine cells in a fresh kernel, with
  no saved-result override. All 12 model-dependent inputs came from these new
  calculations. The ten clone-fate seeds reproduced the original metrics.
  Both full figures are embedded as actual execution outputs.
- S3's original radial training runtime and the released model configuration
  are now distributed as simulation_training_code.zip (380,629 bytes). The
  six-stage trainer was executed with one epoch per stage on the original data,
  producing all six checkpoints. This checks training execution, not a new
  full 5,252-epoch reproduction. The original full schedule remains in the
  download. The guide connects generation, training, evaluation and the notebook.
- S4/S5 training implementations and five selected configurations are distributed
  in nonspatial_training_code.zip (1,720,687 bytes). Each model's six-stage
  training entry ran with one epoch per stage and 32 observed cells per time.
  The current launcher also takes the cutoff and LR threshold from the selected
  preparation and prior. Its LR training path was executed separately. These
  checks do not constitute new full-length model training.
- S6 trained five held-out classifiers and one trajectory classifier, generated
  growing/fixed-cohort trajectories and evaluated smoothing. S39 reran its
  analysis and 10,000-label spatial permutations. S41 reran four-dataset
  min/geometric-mean comparisons. S43 recalculated summaries from saved external
  outputs. Five sensitivity/table notebooks ran with explicit selected inputs.
- S7/S8 preparation ran on released heart data. All six transformed coordinate
  arrays matched the paper input exactly. Portable training/comparison commands
  were added; seven full training variants were not repeated in this pass.
- Portable LOTO configuration, heart preparation/verification, CytoBridge
  preflight and two linear-control predictions ran. The complete five-dataset,
  all-method training benchmark was not repeated.
- Selected Zebrafish classifiers now reach removal/daughter analyses. Older
  generators cannot restore obsolete cells. An additional non-spatial model/
  numerical-input archive was uploaded from the server to the data release.

## All 27 notebooks

“Executed” means execution in this review. Earlier saved outputs alone do not
count. Optional training and native external-method inference are separate.

| Notebook under docs/tutorials/ | Current route and verification |
| --- | --- |
| paper_figures/main_figure_4.ipynb | Measured data/model → populations, communication, LR, lineage and fields → all five panels. Executed. |
| paper_figures/main_figure_5.ipynb | Prerequisite dataset output → fresh c/d/e fields and a/b plots. Executed. Displayed e redraws original per-cell values; original random groups are missing, as detailed below. |
| paper_figures/main_figure_2.ipynb | September 7 check: a snapshots, b fields and e ten-seed W2 executed, with c/d references missing then. The September 12 update supplies both references and the original attention model. See `agist/FIGURE2_REPRODUCTION.md` for the subsequent model and notebook checks. |
| paper_figures/mosta_figures.ipynb | Explicit model/trajectory calculations → S11–S18. Older standalone generators retired. |
| paper_figures/arista_figures.ipynb | Dataset populations/growth → S19–21; new gene reconstruction, clustering, GO and LR → S22–24. All six figures executed and displayed. |
| paper_figures/arista_local_domains.ipynb | Same-run Figure 5c fields/attention and S23 LR → segmentation/permutations → six tables → S25. Executed. |
| paper_figures/agist_figures.ipynb | New clustering/model fields → S2. Original reference/weights → five-seed attraction evaluation → S3. Executed; original failed diagnostic outcomes retained internally. |
| paper_figures/nonspatial_figures.ipynb | Default nine-cell notebook calculates model fields, both-arm distributions, ten-seed clone fate, new-RNA direction, dense trajectories and LR attribution, then converts and plots S4/S5. Executed without saved-result overrides. The linked guide additionally provides preparation and original-model training with explicit downstream selection. |
| paper_figures/chicken_heart_daily.ipynb | Declared population/model inputs → paper velocity, composition, transition and interaction plots. Non-paper daily transition removed in earlier repair. |
| paper_figures/classifier_smoothing.ipynb | Classifiers → trajectories → smoothing evaluation → selected plotting inputs. Producers and notebook executed. |
| paper_figures/compute_cost.ipynb | Actual run summaries → table. Selected directory persists. New-hardware measurements are separate; no new training timing claimed. Executed. |
| paper_figures/interaction_ablation.ipynb | Explicit Full/No-LR and inference-on/off inputs reach S42. Inference producer connected. Fresh Full/No-LR training matrix not run here. |
| paper_figures/loto_benchmark.ipynb | Completed target summaries → S45. Portable new training route documented and heart preparation/control-tested. Full repeated benchmark not run here. |
| paper_figures/loto_benchmark_summary.ipynb | Explicit completed comparisons → S44. Selected-input plotting executed. Same new-run distinction as S45. |
| paper_figures/lr_complex_aggregation.ipynb | Explicit workflow summaries → four-dataset min/geometric-mean calculation → S41. Executed from original downstream states/communication. |
| paper_figures/training_histories.ipynb | Explicit logs → history collection → S46. No new full training here. |
| paper_figures/zebrafish_attention.ipynb | Named model/external outputs → analysis/JAM controls/spatial permutations → collector → S39. Executed. Native inference and full matched training not repeated. |
| paper_figures/spatial_communication.ipynb | Original external outputs/selection → new aggregate/molecular summaries → collector → S43. Executed. External methods themselves not rerun. |
| paper_figures/zebrafish_si_s31_s38.ipynb | Model simulation/perturbation → analysis → figures. Explicit classifier now propagates to removal/daughter analyses as well as populations. |
| dataset_workflows/admouse.ipynb | Inline training choice → selected model simulation/classification → S26. Later AD page now has real gene/LR/perturbation producers. |
| dataset_workflows/arista.ipynb | Optional inline training or downloaded model → populations/communication/growth → S20. Five cells executed with downloaded weights. Model selection saved for later pages. |
| dataset_workflows/chicken_heart.ipynb | Simulation/growth → S9, with named H5AD continuation. No new full training in this pass. |
| dataset_workflows/mosta.ipynb | Simulation/classification/growth → named S11–18 continuation. Canonical notebook preserved by generator repair. |
| dataset_workflows/zebrafish.ipynb | Model growth → S32. Final description correctly identifies the S31–38 calculation route. |
| data_preparation/synthetic_preprocessing.ipynb | Self-contained count generation/preprocessing. No paper-figure claim. |
| your_data.ipynb | Raw data → preparation/configuration → training/downstream calls. No new full training claimed here. |
| model_analysis.ipynb | Declared model/data → velocity/growth arrays. No unrelated scientific figures. |

Linked AD, alignment, simulation, benchmark and sensitivity Markdown routes
were reviewed alongside these notebooks.

## Original information still needed for exact reproduction

1. **Figure 2c/d:** original generator files `attn_matrix_time0.npy` and
   `g_values.npy` from `mosta_interaction_1017_tiaocan` were not found in the
   server project, released ZIPs or local figure archives. Three available
   velocity arrays do not replace attention or growth. The original renderer
   was located and ported; it can draw c/d with the matching arrays.
2. **Figure 2d source record:** the later `r = 0.96` label cites
   `simulation_gradients_np_gt.npy/pearson_r`, a 31,816 × 52 base-velocity
   array, not growth. That record cannot establish the growth correlation.
   Resolve against original growth, not by substituting another quantity.
   The manuscript figure and label were not changed here.
3. **Figure 5e:** original per-cell scores exist, but original random 1,024-cell
   groups/RNG state do not. Original values are redrawn and a real new model
   calculation is supplied separately. Five preselected seeds retained broad
   early/late patterns and all 14/14 matched growth-increase directions. Only
   120/177 original group means were within the five-draw range. This is not
   exact per-group reproduction or a confidence interval. No best seed chosen.

## Records and publication

Detailed execution commands/comparisons are in the author's workspace under
`output/notebook_input_review_20260907/`: `arista_reader_repair_20260907.md`,
`simulation_nonspatial_execution_review.md`, `sensitivity_reader_repair.md`
and `portable_benchmark_reader_repair.md`. AD is also summarized in
[AD_READER_CHECK_20260907.md](AD_READER_CHECK_20260907.md).

Automatic GitHub-to-Read-the-Docs deployment was repaired and verified on
commit `8462c54`. Commit `9b42bc6` also passed both Python CI jobs, the docs CI
job and Read-the-Docs build 34434652. The local full suite for the subsequent
source-download/non-spatial additions passed 1,855 tests with 12 skips. The
subsequent focused download, public-language, selected-input and numerical
analysis checks passed 395 tests with three optional-dependency skips. The
server's non-spatial numerical-analysis tests passed all 21 cases. Later changes
require their own push, build and page check.
Full-suite execution and publication are verified separately from scientific
input-to-figure execution.
