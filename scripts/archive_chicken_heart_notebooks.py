"""Prepare portable copies of the collaborator's four chicken-heart notebooks.

The numerical cells and selected plotting functions are copied unchanged.
Only filesystem setup and the classifier filename are made explicit.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import nbformat


NOTEBOOKS = (
    "formal_daily_piecewise_interpolation_celltypecorrected",
    "formal_daily_piecewise_replot_celltypecorrected",
    "formal_d10_velocity_detail_celltypecorrected",
    "formal_supplementary_replot_celltypecorrected",
)
FUNCTIONS = {
    "heart.py": {
        "HEART_LABEL_TO_COLOR", "set_seed", "plot_celltype_stackbar",
        "plot_segment_network_evolution", "plot_side_by_side_spatial_with_custom_colors",
        "prepare_data_for_side_by_side_2d", "_prepare_g_heatmap_matrix",
    },
    "heart_lineage_functions.py": {
        "prepare_multi_timepoint_adata", "plot_lineage_transition",
    },
}
IMPORTS = '''from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

HEART_REPO_ROOT = Path.cwd()
'''


def select_functions(source: str, names: set[str]) -> str:
    """Keep the original function bodies and their local helper functions."""
    tree = ast.parse(source)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    selected = set(names)
    while True:
        previous = selected.copy()
        for name in previous:
            for node in ast.walk(definitions[name]):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in definitions and node.id != "HEART_REPO_ROOT":
                        selected.add(node.id)
        if previous == selected:
            break
    bodies = [ast.get_source_segment(source, node) for node in tree.body
              if any(definitions.get(name) is node for name in selected)]
    imports = IMPORTS
    suffix = "\n"
    if "set_seed" in names:
        imports += "\nimport random\nimport torch\n"
        # The source helper sets this seed when imported. Retain that behavior
        # for the numerical cells that follow the notebook imports.
        suffix += "\nset_seed(42)\n"
    return '"""Plotting functions copied from the collaborator heart analysis."""\n' + imports + "\n\n" + "\n\n".join(bodies) + suffix


def portable_source(source: str) -> str:
    source = re.sub(
        r'CLUSTER_ROOT = Path\([^\n]+\)\nREPO_ROOT = CLUSTER_ROOT / "cytobridge-downstream"',
        'import os\nREPO_ROOT = Path(os.environ.get("CYTOBRIDGE_SOURCE_DIR", ".")).resolve()\n'
        'PROJECT_DIR = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()\n'
        'DATA_DIR = PROJECT_DIR / "data" / "chicken_heart"\n'
        'sys.path.insert(0, str(REPO_ROOT / "reproduction" / "chicken_heart"))',
        source,
    )
    source = source.replace('ANALYSIS_ROOT = REPO_ROOT / "chicken_heart_analysis"',
                            'ANALYSIS_ROOT = Path(os.environ.get("CYTOBRIDGE_HEART_OUTPUT_DIR", PROJECT_DIR / "outputs" / "chicken_heart_paper")).resolve()')
    source = re.sub(r'^PACKAGE_ROOT = .+$',
                    'import CytoBridge as cb\nPACKAGE_ROOT = Path(cb.__file__).resolve().parents[1]',
                    source, flags=re.MULTILINE)
    source = re.sub(r'^CELLTYPE_SHARE_ROOT = .+$', 'CELLTYPE_SHARE_ROOT = DATA_DIR', source, flags=re.MULTILINE)
    source = source.replace('HELPER_ROOT = REPO_ROOT / "downstream_helpers"',
                            'HELPER_ROOT = REPO_ROOT / "reproduction" / "chicken_heart" / "downstream_helpers"')
    replacements = {
        'DATA_HEART_DIR = REPO_ROOT / "data" / "heart"': 'DATA_HEART_DIR = DATA_DIR / "raw"',
        'ALIGNED_H5AD_PATH = ANALYSIS_ROOT / "chicken_heart_ot_retrained_20260823_c72e592" / "preprocess" / "chicken_heart_aligned.h5ad"': 'ALIGNED_H5AD_PATH = DATA_DIR / "aligned.h5ad"',
        'MODEL_DIR = ANALYSIS_ROOT / "chicken_heart_ot_retrained_20260823_c72e592" / "training"': 'MODEL_DIR = DATA_DIR / "model"',
        'EDGE_PREDICTOR_PATH = ANALYSIS_ROOT / "chicken_heart_ot_retrained_20260823_c72e592" / "preprocess" / "edge_classifier" / "chicken_heart_edge_model.pt"': 'EDGE_PREDICTOR_PATH = DATA_DIR / "edge_classifier" / "chicken_heart_edge_model.pt"',
        'WORKFLOW_CONFIG_PATH = PACKAGE_ROOT / "CytoBridge" / "workflow_configs" / "chicken_heart.json"': 'WORKFLOW_CONFIG_PATH = DATA_DIR / "workflow.json"',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = re.sub(
        r'CLASSIFIER_CACHE_PATHS = .*?CLASSIFIER_CACHE_PATH = CLASSIFIER_CACHE_PATHS\[0\]',
        'CLASSIFIER_CACHE_PATH = DATA_DIR / "classifier_cache" / "classifier_resmlp_432f09f20ff65c0d.pt"\n'
        'if not CLASSIFIER_CACHE_PATH.is_file():\n    raise FileNotFoundError(CLASSIFIER_CACHE_PATH)',
        source, flags=re.DOTALL,
    )
    if any(fragment in source for fragment in ("/lustre/", "CLUSTER_ROOT", "CLASSIFIER_CACHE_PATHS")):
        raise ValueError("Unconverted notebook setup")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    notebook_dir = args.output_dir / "notebooks"
    helper_dir = args.output_dir / "downstream_helpers"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    helper_dir.mkdir(parents=True, exist_ok=True)
    (helper_dir / "__init__.py").write_text('"""Plotting helpers for the archived chicken-heart notebooks."""\n')
    for filename, names in FUNCTIONS.items():
        source = (args.helpers / filename).read_text()
        (helper_dir / filename).write_text(select_functions(source, names))
    for stem in NOTEBOOKS:
        matches = list(args.notebooks.rglob(stem + ".ipynb"))
        if len(matches) != 1:
            raise ValueError(f"Expected one source notebook for {stem}, found {len(matches)}")
        nb = nbformat.read(matches[0], as_version=4)
        for i, cell in enumerate(nb.cells):
            cell.id = f"heart-{NOTEBOOKS.index(stem)}-{i:02d}"
            if cell.cell_type == "code":
                cell.source = portable_source(cell.source)
                cell.outputs = []
                cell.execution_count = None
        nb.metadata["cytobridge"] = {"source_model_commit": "c72e592", "runs_training": False}
        nbformat.write(nb, notebook_dir / (stem + ".ipynb"))


if __name__ == "__main__":
    main()
