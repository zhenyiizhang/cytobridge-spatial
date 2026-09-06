# Tutorial review rules

These rules record the author's corrections to the September 2026 tutorials.
They apply to the website, notebooks, linked analysis scripts, and recorded
notebook outputs. They are maintenance instructions, not tutorial prose.

## What belongs in a tutorial

Use the current manuscript and SI as the figure reference. Record the figure
number and panels before adding an output. A plot from a collaborator's notebook
is not automatically a paper figure. Do not add diagnostic maps, exploratory
lineage plots, alternate heatmaps, or extra API illustrations merely because
the code produces them. A general API page may show arrays and small tables
without drawing another set of scientific figures.

Inspect hidden outputs too: a command called for one current figure must not
also display an outdated figure. Keep original research files recoverable,
but do not present them as the current reproduction tutorial.

## Resolve a disagreement before changing the paper

When the website and paper differ, first identify the figure actually embedded
in the current Word or SI and follow its own source record. Do not assume the
tutorial is correct, and do not replace an accepted paper figure just to match
the tutorial. Check each panel separately: in September 2026, S19 was already
unadjusted, whereas the embedded Figure 5b still used adjusted display coordinates.

Check the coordinates or quantities that the plotting function actually reads.
A filename, caption, successful notebook run, or visually similar preview is
not enough. Keep simulation output, display transforms, and training alignment
distinct. If the error is in the tutorial, correct its code, selected inputs,
download, and executed output together. Change the paper only when that change
is needed and authorized.

## Follow the calculation from input to figure

For every analysis, a reader must be able to answer:

1. Which downloaded file or preceding notebook output do I start with?
2. Which function or command calculates the scientific quantity?
3. What does it return or save, and what do its rows and columns represent?
4. Which following plotting call reads that exact result?
5. Which manuscript or SI panel does the output reproduce?

Loading final summary tables is not model evaluation. Reading a finished PDF
and exporting it again is not figure reproduction. Displaying a PNG that the
preceding numerical plotting call just created is fine.

Pass newly calculated objects or explicit result paths to the plotting step.
Do not calculate a table in one cell and silently reload an unrelated included
table in the next. Changing the documented input must change the calculation
and plot without editing package internals. Existing output files must not
silently suppress a requested recalculation.

## Put the steps in the order a reader runs them

Give each dataset one primary route. Keep training, model loading, simulation,
analysis, and plotting in a clear sequence on that route. Specialized reference
pages may explain details, but must not be necessary for reconstructing an
unstated sequence of prerequisites.

Explain whether the input already contains processed expression features and
aligned coordinates. Do not ask a reader to preprocess again before a function
that already does so. When training is optional, state the choice once, show
the complete training call, and make the following model-loading cell use the
selected model directory. Do not overwrite the downloaded paper model.

An explicitly selected classifier must also be loaded without retraining or
overwriting its checkpoint. Validate the feature order and count. A change in
cache metadata version is not permission to replace downloaded weights. Use
one declared classifier across a dataset's figures, and check its recorded
model-selection procedure before describing any full-data refit.

Use concrete names from the distributed files and configuration. A directory
link is not an executable command. Do not pack several commands, alternative
flags, and instructions into one prose sentence. Keep model settings in the
configuration or the calculation where they are used, not as unexplained
introductory qualifiers.

## Write for readers, not for the internal revision process

Remove references to collaborators' earlier displays, internal handovers,
source checkouts, corrected-versus-legacy histories, smoke runs, and release
checklists from the tutorial narrative. Explain what a function does using
its input and output. Preserve necessary scientific parameters and definitions.
Do not replace technical accuracy with vague language.

Do not add learning goals, exercises, or result interpretation unless the
author requests them. These tutorials explain how to reproduce the analysis.

## Verify three different things

**Execution:** run the current cells in a fresh kernel with the stated files.
Saved execution counts and embedded images are not sufficient. An optional
training step that was not rerun must not be reported as training-tested.

**Calculation continuity:** follow the actual arrays and files through the
called implementation. Where practical, change a temporary input and confirm
that the immediately following plot uses it. Do not modify accepted results
for this check.

**Paper correspondence:** inspect the rendered output against the current
paper panel. Check the populations, time points, model, summary statistic,
displayed groups, and layout. Successful execution does not establish this.

If any of these checks remains open, record it explicitly. Do not describe
the package as ready to release because the documentation builds or unit
tests pass. Do not count different cells producing the same apparent image
as independent verification.

## Before delivering

Read every edited page from beginning to end as a new reader. Inspect its
actual browser output and called code, not just a text diff. Check links and
make sure removed examples have also disappeared from recorded outputs.
Rebuild the website, verify the pushed GitHub commit, and check the published
page. Report local verification, publication, and remaining work separately.

Keep a dated page-by-page review outside the public tutorial navigation.
Record concrete failures and fixes, rather than a single package-wide pass.

## Corrections that prompted these rules

- MOSTA: a drawing wrapper did not explain or calculate its upstream analyses.
- Chicken heart: an internal alignment note and an extra daily transition plot
  were presented on a paper-reproduction page.
- General model analysis: additional spatial-velocity and growth illustrations
  did not correspond to the paper.
- AD: the reader was sent to a generic training page instead of being shown
  the dataset's training call and the model directory used next.
- Zebrafish: several overlapping entry pages made the analysis order unclear.
- S34 and S36: a working calculation still called an older plot style. Check
  the current SI image, not only the renderer's name or figure number.
- Custom-data configuration: changing a workflow setting must also update
  the training configuration used by the command. Test a reader's edit.
- Several SI notebooks: the displayed calculation was disconnected from a
  plotting wrapper that loaded its own defaults.
- ARISTA: the S19 tutorial selected spatially adjusted display coordinates,
  whereas the current SI used the original simulation. Verify the actual
  coordinate arrays as well as the figure filename and caption. All generated
  paper maps now use the simulated spatial coordinates directly.
