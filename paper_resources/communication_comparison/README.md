# Cell–cell interaction comparisons: Supplementary Figure S11

The tables compare CytoBridge cell-type interaction scores with COMMOT and
CellAgentChat, and examine selected ligand–receptor axes with NicheNet.

To draw the figure from the numerical tables:

```bash
python -m reproduction.current_figures --figures 11 --output-dir outputs/communication_comparison
```

The [communication tutorial](../../docs/tutorials/paper_figures/spatial_communication.ipynb)
explains the calculations and the inputs for each method. The Python
comparison functions are in `CytoBridge/spatial_communication_consistency.py`.
