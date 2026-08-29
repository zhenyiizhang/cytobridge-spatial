from __future__ import annotations

from pathlib import Path
import os
import sys

# SciPy is a base dependency.  Import it before autodoc activates its optional
# ``torch`` mock; recent SciPy versions probe ``torch.Tensor`` while loading.
import scipy.stats
from sphinx.ext.autodoc.mock import _MockObject


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / "_build" / ".matplotlib"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "docs" / "_build" / ".numba"))

from CytoBridge._version import __version__


# A few legacy modules evaluate annotations such as ``str | torch.device`` at
# import time.  Sphinx's optional-module proxy is intentionally not a type, so
# teach that proxy to collapse such unions to ``object`` during documentation
# imports.  Runtime annotations are unchanged.
def _mock_union_as_object(_mock: object, _other: object) -> type[object]:
    return object


_MockObject.__or__ = _mock_union_as_object
_MockObject.__ror__ = _mock_union_as_object


project = "CytoBridge"
author = "CytoBridge developers"
copyright = "2026, CytoBridge developers"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Read the Docs installs the base package and documentation tools, but it does
# not need CUDA, PyTorch, Scanpy, or the optional plotting stack merely to
# render API signatures.  Mock only those optional imports; the core metadata
# and NumPy/AnnData table contracts are imported normally.
autodoc_mock_imports = [
    "cellrank",
    "geomloss",
    "kaleido",
    "matplotlib",
    "ot",
    "phate",
    "PIL",
    "plotly",
    "qnorm",
    "scanpy",
    "scvelo",
    "seaborn",
    "torch",
    "torch_geometric",
    "torchdiffeq",
    "umap",
]
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
exclude_patterns = [
    "_build",
    "tutorials/paper_figures/outputs/**",
    "Thumbs.db",
    ".DS_Store",
    # Retained project records that are not part of the public documentation.
    "historical_artifact_compatibility.md",
    "release_notes.md",
    "scientific_contract.md",
    "zebrafish_clean_counts_workflow.md",
]

html_theme = "furo"
html_title = f"CytoBridge {release}"
html_logo = "_static/cytobridge_logo.svg"
html_static_path = ["_static"]
html_css_files = ["cytobridge.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/zhenyiizhang/cytobridge-spatial/",
    "source_branch": "release/cytobridge-reproducible-20260812",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#08786b",
        "color-brand-content": "#07584f",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7ed4c7",
        "color-brand-content": "#7ed4c7",
    },
}
