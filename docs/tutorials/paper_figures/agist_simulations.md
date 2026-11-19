# AGIST simulation and training

AGIST starts from 5,000 Brain cells sampled from the E12.5 MOSTA section. A
trained MOSTA model generates their subsequent states, population growth and
interactions. Four snapshots are then used to train a new CytoBridge model.
The Figure 2 notebook compares this model with the generator.

## Download the paper experiment

```bash
python -m CytoBridge.datasets agist --output-dir paper_run --kind agist_final.zip
```

The archive creates `paper_run/data/agist_final/`. It includes the initial
states, generated observations, generator and fitted-model checkpoints,
baseline predictions, and evaluation arrays. Use the
[Figure 2 notebook](main_figure_2.ipynb) to evaluate the supplied model and draw
Figure 2 and Supplementary Figures S2–S3.

## Generate new observations and train

Run these commands from the source-code folder. Choose a new output directory
so the supplied paper experiment remains available.

```bash
export CYTOBRIDGE_FIGURE2_RUN="$PWD/paper_run/outputs/agist_new"
export CYTOBRIDGE_CODE="$PWD/paper_run/data/agist_final/code"
export CYTOBRIDGE_FIGURE2_TEACHER="$PWD/paper_run/data/agist_final/teacher"
python reproduction/agist/experiment/run_cytobridge.py generate \
  --initial-states paper_run/data/agist_final/data/initial_states.npy --device cuda:0
python reproduction/agist/experiment/run_cytobridge.py train --device cuda:0
python reproduction/agist/experiment/run_cytobridge.py predict --device cuda:0
```

The initial expression features are already normalized and projected into the
shared PCA space. Generation uses these values directly. Training writes a new
six-stage model under `agist_new/cytobridge/model/`. Prediction runs ten
simulations from the same initial population.

For the baseline fits, install the stVCR and STORIES environments and set
`STVCR_SOURCE` and `STORIES_SOURCE` to their source directories. The versions
used in the paper are recorded in the experiment's `PROTOCOL.md`.

```bash
python reproduction/agist/experiment/run_baseline.py stvcr --device cuda:0
python reproduction/agist/experiment/run_baseline.py stories --device cuda:0
```

Return to the CytoBridge environment and evaluate the new run with the
calculation steps in the Figure 2 notebook, selecting `agist_new` as its run
directory. Simulation seeds and the full training schedule are specified in
`run_cytobridge.py`.
