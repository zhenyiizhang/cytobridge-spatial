# Paper datasets

Choose a dataset to follow its analysis from raw counts to a trained model and
downstream plots. Each notebook follows the same order: get the data, set the
paths, prepare the input, train, and open the results.

These notebooks contain actual training calls. Add the input files before
running them locally. The website does not execute GPU training, and it does
not display placeholder output as if a model had been trained.

| Dataset | Input and model availability |
| --- | --- |
| Zebrafish | Exact paper inputs and final model download pending |
| MOSTA | Model files in the source repository; aligned H5AD download pending |
| ARISTA | Model files in the source repository; aligned H5AD download pending |
| AD mouse | Exact paper inputs and final model download pending |
| Chicken heart | Final model files included; aligned H5AD and annotation inputs need a separate download |

See [Data and checkpoints](../../data_checkpoints.md) for the original study
downloads and the files needed for each starting point.

```{toctree}
:maxdepth: 1

zebrafish
mosta
arista
admouse
chicken_heart
```

For another experiment, use [Run CytoBridge on your data](../your_data.ipynb).
To draw from the paper's existing numerical results, use
[Paper figures](../paper_figures/index.md).
