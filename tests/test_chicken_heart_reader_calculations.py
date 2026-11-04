"""Check that the heart tutorial calculates transitions and plots its own results."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/tutorials/paper_figures/chicken_heart_daily.ipynb"


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


def test_transition_calculation_uses_model_predictions(tmp_path):
    tree = ast.parse(notebook_code())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "calculate_transition")
    source = SimpleNamespace(X=np.zeros((3, 3)),
                             obs=pd.DataFrame({"celltype_prediction": ["A", "A", "B"]}))
    speed = [0.]
    calls = []

    def velocity(**kwargs):
        calls.append(kwargs["time_value"])
        return {"full": np.full_like(kwargs["data"], speed[0])}

    def classify(**kwargs):
        return np.where(kwargs["points"][:, 0] > .5, "B", "A")

    scope = dict(np=np, pd=pd, populations={"D4": source}, times={"D4": 0., "D7": 1.},
                 output=tmp_path, model=object(), cutoff=1., device="cpu",
                 classifier=SimpleNamespace(model=None, label_encoder=None, feature_dim=3,
                                             include_time_feature=False),
                 cb=SimpleNamespace(tl=SimpleNamespace(compute_velocity_components=velocity,
                                                        predict_labels_for_points=classify)))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    first = scope["calculate_transition"]("D4", "D7")
    assert len(calls) == 20
    assert np.allclose(first["A"], 1.)
    speed[0] = 1.
    second = scope["calculate_transition"]("D4", "D7")
    assert np.allclose(second["B"], 1.)
    assert np.allclose(second.sum(axis=1), 1.)
    counts = pd.read_csv(tmp_path / "transition_counts_D4_D7.csv", index_col=0)
    assert counts.values.sum() == 3
    np.testing.assert_array_equal(source.X, np.zeros((3, 3)))


def test_daily_notebook_connects_calculations_to_plotting():
    code = notebook_code()
    assert 'population_dir / "manifest.json"' in code
    assert '("communication_h5ad", "communication_slice_data", communication_populations)' in code
    assert "cb.tl.compute_velocity_components(" in code
    assert "velocity_by_time=velocity" in code
    assert "transition_matrices={pair: transitions[pair]" in code
    assert "cb.tl.compute_timepoint_communications(" in code
    assert "all_time_communications=communication" in code
    assert "draw_supplementary" not in code
    assert "Image(" not in code and "SVG(" not in code
    text = NOTEBOOK.read_text().lower()
    assert "dataset_workflows/chicken_heart.ipynb" in text
    assert "collaborator" not in text
    assert "earlier alignment" not in text


def test_tutorial_omits_extra_daily_transition_illustration():
    code = notebook_code()
    assert "daily_settings" not in code
    assert '"daily_transitions"' not in code
    assert '"daily": days' not in code
    assert '"cardiomyocyte_transitions"' in code
    assert '"epicardial_fibroblast_transitions"' in code


def test_model_notebook_is_not_scheduled_without_its_dataset():
    runner = ROOT / "scripts/execute_paper_notebooks.py"
    download_set = next(node.value for node in ast.parse(runner.read_text()).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "DOWNLOAD_NOTEBOOKS"
                                for t in node.targets))
    assert "chicken_heart_daily" in ast.literal_eval(download_set)


def test_each_lineage_panel_uses_only_its_selected_intervals(tmp_path):
    function = next(n for n in ast.parse(notebook_code()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "plot_transitions")
    population = SimpleNamespace(X=np.zeros((1, 3)), obs=pd.DataFrame({"celltype_prediction": ["A"]}))
    recorded = []
    matrices = {("D4", "D5"): "daily", ("D4", "D7"): "long interval", ("D7", "D10"): "later"}
    scope = dict(np=np, populations={day: population for day in ("D4", "D5", "D7", "D10")},
                 prepare_multi_timepoint_adata=lambda **kwargs: population,
                 plot_lineage_transition=lambda **kwargs: recorded.append(kwargs),
                 plt=SimpleNamespace(show=lambda: None), palette={"A": "black"},
                 transitions=matrices, output=tmp_path)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    scope["plot_transitions"](["D4", "D5"], "daily", {}, (10, 4))
    assert recorded[-1]["transition_matrices"] == {("D4", "D5"): "daily"}
    scope["plot_transitions"](["D4", "D7", "D10"], "myocardial", {}, (10, 4))
    assert set(recorded[-1]["transition_matrices"]) == {("D4", "D7"), ("D7", "D10")}
