#!/usr/bin/env python3
"""Build reader-followable AGIST and non-spatial numerical figure tutorials."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/tutorials/paper_figures"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    path = DEST / name
    if path.is_file():
        previous = nbf.read(path, as_version=4)
        old_code = [cell for cell in previous.cells if cell.cell_type == "code"]
        new_code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if [cell.source for cell in old_code] == [cell.source for cell in new_code]:
            for old, new in zip(old_code, new_code):
                new.outputs = old.outputs
                new.execution_count = old.execution_count
    nbf.write(notebook, DEST / name)


SETUP = """
import os
import sys
import subprocess
from pathlib import Path

import CytoBridge as cb
from IPython.display import Image, display

project = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()
device = os.environ.get("CYTOBRIDGE_DEVICE", "cuda")
run_name = os.environ.get("CYTOBRIDGE_RUN_LABEL", "reader_run")
repo = Path(cb.__file__).resolve().parents[1]

def run(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                   cwd=repo, check=True)
"""


def agist():
    """Preserve the canonical, executed paper notebook."""
    book = nbf.read(DEST / "agist_figures.ipynb", as_version=4)
    write("agist_figures.ipynb", book.cells)


def main2():
    """Preserve the canonical, executed paper notebook."""
    book = nbf.read(DEST / "main_figure_2.ipynb", as_version=4)
    write("main_figure_2.ipynb", book.cells)


def attention_recovery():
    """Preserve the canonical, executed paper notebook."""
    book = nbf.read(DEST / "agist_attention_recovery.ipynb", as_version=4)
    write("agist_attention_recovery.ipynb", book.cells)


def nonspatial():
    """Preserve the canonical, executed paper notebook."""
    book = nbf.read(DEST / "nonspatial_figures.ipynb", as_version=4)
    write("nonspatial_figures.ipynb", book.cells)


def weinreb_matched():
    cells = [md("""
# Supplementary Figure S4: lineage-traced hematopoiesis

Recalculate the distribution errors and clone-fate agreement in S4c/d, then
draw the complete figure. Panels b–f use the same three LR-informed Full
models. The three No-interaction models were trained without the interaction
network using the corresponding training settings.

This walkthrough needs NumPy, pandas, SciPy and the figure dependencies.
It runs on a CPU from the per-simulation errors and clone probabilities
included with the repository. No finished image is used as a plotting input.
The final section shows how to repeat prediction with all six checkpoints.
"""), code("""
from pathlib import Path
import os
import shutil
import tempfile
import pandas as pd
import CytoBridge as cb
from IPython.display import Image, display
from reproduction.nonspatial.weinreb_matched import summarize
from reproduction.nonspatial.update_weinreb_s4 import update

repo = Path(cb.__file__).resolve().parents[1]
source = repo / "reproduction/nonspatial/larry_evaluation"
output = Path(tempfile.mkdtemp(prefix="cytobridge-s4-"))
print(output)
"""), md("""
## Recalculate the two comparisons

For S4c, each point represents one trained model, averaged over 50 prediction
runs. Bars show the mean across the three models. For S4d, each model was
used for ten predictions from the day-2 cells. We average clone-fate
probabilities across the three models before comparing them with the
observed descendants of 1,237 lineages. Averaging metric values instead
would answer a different question.
"""), code("""
paths = summarize(source / "raw", output / "summary")
display(pd.read_csv(paths["clone_fate"])[["metric", "full", "no_interaction"]].round(3))
display(pd.read_csv(paths["distribution"])[["time", "w1_full", "w1_no_interaction", "w2_full", "w2_no_interaction"]].round(3))
"""), md("""
## Draw the complete S4

The unchanged panels use the measured coordinates, model-derived velocity
grids, interaction scores and pathway tables in the figure input bundle.
Replace its c/d tables with the values calculated above and draw all panels.
"""), code("""
from reproduction.paper_figures import draw_supplementary

bundle = output / "plot_inputs"
shutil.copytree(repo / "CytoBridge/results/data/nonspatial_figures", bundle)
update(output / "summary", bundle)
figures = draw_supplementary([4], output / "figures", results_dir=bundle)
display(Image(filename=str(figures["s4"][1]), width=720))
"""), md("""
## Repeat prediction from the checkpoints

Download and extract the LARRY models from
[the S4 archive](https://github.com/zhenyiizhang/cytobridge-spatial/releases/tag/weinreb-s4-20261029).
It includes the six checkpoints, prepared cells, per-run results, training
code and checksums. Access requires repository permission while it is private.

Set `CYTOBRIDGE_S4_ARCHIVE` to the extracted folder to run the optional cell
below on a GPU. It repeats both evaluations, averages the results in the
same order, and draws S4 from the newly calculated values. Otherwise the
cell leaves the CPU reconstruction above unchanged.
"""), code("""
archive = os.environ.get("CYTOBRIDGE_S4_ARCHIVE")
if archive:
    from reproduction.nonspatial.weinreb_matched import evaluate
    evaluate(archive, output / "repeated", device=os.environ.get("CYTOBRIDGE_DEVICE", "cuda"))
    update(output / "repeated/summary", bundle)
    repeated = draw_supplementary([4], output / "repeated_figures", results_dir=bundle)
    display(Image(filename=str(repeated["s4"][1]), width=720))
"""), md("""
### Check the averaging

Choose one day and distance measure in `distribution_means_per_training_seed.csv`.
The mean of its three Full values should equal the corresponding blue bar.
The same calculation applies to No-interaction. The full per-seed table also
shows the variation between independent training runs.
"""), code("""
means = pd.read_csv(output / "summary/distribution_means_per_training_seed.csv")
means.query("time == 1 and metric == 'w2'").groupby("model").value.mean()
""")]
    write("weinreb_s4.ipynb", cells)


if __name__ == "__main__":
    agist()
    main2()
    attention_recovery()
    nonspatial()
    weinreb_matched()
