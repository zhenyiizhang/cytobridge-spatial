"""Keep the archived daily analysis portable and its calculations intact."""
import ast
import importlib.util
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("heart_archive", ROOT / "scripts/archive_chicken_heart_notebooks.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_helper_selection_keeps_function_bodies_without_legacy_imports():
    source = '''from obsolete_model import train
COLOUR = "blue"
def helper(x):
    return x ** 2
def plot(x):
    return helper(x), COLOUR
def old_training():
    return train()
'''
    selected = archive.select_functions(source, {"plot"})
    assert "obsolete_model" not in selected
    assert "old_training" not in selected
    old_functions = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n) for n in ast.parse(selected).body if isinstance(n, ast.FunctionDef)}
    assert new_functions == {name: old_functions[name] for name in ("helper", "plot")}
    assert 'COLOUR = "blue"' in selected


def test_archived_notebooks_have_portable_inputs_and_no_training():
    for stem in archive.NOTEBOOKS:
        notebook = nbformat.read(ROOT / "reproduction/chicken_heart/notebooks" / (stem + ".ipynb"), as_version=4)
        source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        assert "CYTOBRIDGE_PROJECT_DIR" in source
        assert 'MODEL_DIR = DATA_DIR / "model"' in source
        assert 'ALIGNED_H5AD_PATH = DATA_DIR / "aligned.h5ad"' in source
        assert "cb.tl.fit(" not in source
        assert not any(path in source for path in ("/lustre/", "/Users/", "/home/"))
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, stem, "exec")
        if "CLASSIFIER_CACHE_PATH" in source:
            assert "classifier_resmlp_432f09f20ff65c0d.pt" in source
            assert "CLASSIFIER_CACHE_PATHS[0]" not in source


def test_runner_exposes_calculation_order():
    spec = importlib.util.spec_from_file_location("heart_run", ROOT / "scripts/run_chicken_heart_daily_notebooks.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert list(runner.STEPS) == ["interpolation", "daily", "d10", "supplementary"]
    assert tuple(runner.STEPS.values()) == archive.NOTEBOOKS
