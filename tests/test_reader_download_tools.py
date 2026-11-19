from pathlib import Path
import zipfile

import pandas as pd
import pytest

from scripts.build_data_downloads import build_archive, safe_name
from scripts.paper_figures.zebrafish_loss_weight.collect_alpha_metrics import collect


def test_download_archive_has_relative_real_inputs(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"model weights for an archive test")
    bundle = {
        "filename": "example_model.zip", "dataset": "example", "purpose": "model",
        "instructions": "Extract into the project folder.",
        "files": [{"source": str(source), "destination": "data/example/model/weights.bin"}],
    }
    report = build_archive(bundle, tmp_path, reserve_bytes=0)
    assert report["status"] == "ready_to_upload"
    with zipfile.ZipFile(tmp_path / "example_model.zip") as archive:
        assert archive.testzip() is None
        assert archive.read("data/example/model/weights.bin") == source.read_bytes()
    with pytest.raises(FileExistsError):
        build_archive(bundle, tmp_path, reserve_bytes=0)


@pytest.mark.parametrize("name", ["../data.h5ad", "/data.h5ad", ""])
def test_download_rejects_nonrelative_names(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_collects_actual_alpha_evaluations(tmp_path):
    reference = tmp_path / "reference.csv"
    alternative = tmp_path / "alternative.csv"
    pd.DataFrame({"time": [1, 2], "space": ["spatial", "spatial"], "w1": [0.2, 0.3]}).to_csv(reference, index=False)
    pd.DataFrame({"time": [1, 2], "space": ["spatial", "spatial"], "w1": [0.4, 0.5]}).to_csv(alternative, index=False)
    result = collect(reference, alternative)
    assert result["w1"].tolist() == [0.2, 0.3, 0.4, 0.5]
    assert result["model"].tolist() == ["alpha_express_0015"] * 2 + ["alpha_expr_005"] * 2
    pd.DataFrame({"time": [1], "space": ["spatial"], "w1": [0.4]}).to_csv(alternative, index=False)
    with pytest.raises(ValueError, match="same times and spaces"):
        collect(reference, alternative)
