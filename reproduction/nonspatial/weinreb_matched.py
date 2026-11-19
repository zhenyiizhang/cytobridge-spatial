"""S5c/d: three LR Full models and three No-interaction models.

Recalculate the panel inputs from saved per-repeat errors and per-clone
probabilities, or repeat inference with the six distributed checkpoints.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import rel_entr

SEEDS = (42, 43, 44)


def clone_metrics(predicted, observed):
    predicted = predicted.loc[observed.index, observed.columns]
    p, q = predicted.to_numpy(float), observed.to_numpy(float)
    for values in (p, q):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid clone-fate probabilities")
        np.testing.assert_allclose(values.sum(axis=1), 1., rtol=0, atol=1e-10)
    midpoint = (p + q) / 2
    js = np.clip(1 - .5 * (rel_entr(p, midpoint) + rel_entr(q, midpoint)).sum(axis=1) / np.log(2), 0, 1)
    pm = np.isclose(p, p.max(axis=1, keepdims=True), rtol=1e-12, atol=1e-15)
    qm = np.isclose(q, q.max(axis=1, keepdims=True), rtol=1e-12, atol=1e-15)
    return dict(clone_macro_tv_agreement=float((1 - .5 * np.abs(p-q).sum(axis=1)).mean()),
                clone_macro_js_similarity=float(js.mean()),
                clone_macro_dominant_fate_match=float((pm & qm).any(axis=1).mean()))


def summarize(raw_dir, output_dir):
    """Average simulations within each model and probabilities across models."""
    raw, output = Path(raw_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    errors = pd.read_csv(raw / "distribution_all_repeats.csv")
    keys = ["training_seed", "inference_seed", "model", "time", "metric"]
    if errors.duplicated(keys).any():
        raise ValueError("Repeated distribution record")
    assert set(errors.training_seed) == set(SEEDS)
    assert set(errors.model) == {"LR Full", "No-interaction"}
    assert set(errors.time) == {1., 2.} and set(errors.metric) == {"w1", "w2"}
    for _, group in errors.groupby(["training_seed", "model", "time", "metric"]):
        assert len(group) == 50 and set(group.inference_seed) == set(range(10000,10050))
    errors.to_csv(output / "distribution_all_repeats.csv", index=False)
    means = errors.groupby(["training_seed", "model", "time", "metric"], as_index=False).value.mean()
    means.to_csv(output / "distribution_means_per_training_seed.csv", index=False)
    means.groupby(["model", "time", "metric"]).value.agg(["mean", "std", "count"]).to_csv(output / "distribution_across_training_seeds.csv")
    observed = pd.read_csv(raw / "observed_fate_distributions.csv", index_col=0)
    assert len(observed) == 1237
    per_seed, ensemble, comparisons = [], [], []
    for name, arm in (("LR Full", "full"), ("No-interaction", "no_interaction")):
        frames = []
        for seed in SEEDS:
            pred = pd.read_csv(raw / f"{arm}_seed{seed}_probabilities.csv", index_col=0)
            assert set(pred.index) == set(observed.index) and set(pred.columns) == set(observed.columns)
            pred = pred.loc[observed.index, observed.columns]
            frames.append(pred)
            per_seed.append(dict(training_seed=seed, model=name, **clone_metrics(pred, observed)))
        average = sum(frames) / len(frames)
        average.to_csv(output / f"{name.replace(' ', '_')}_ensemble_probabilities.csv")
        ensemble.append(dict(model=name, **clone_metrics(average, observed)))
    pd.DataFrame(per_seed).to_csv(output / "clone_metrics_per_training_seed.csv", index=False)
    table = pd.DataFrame(ensemble).set_index("model")
    table.to_csv(output / "clone_metrics_probability_ensemble.csv")
    for metric in ("tv_agreement", "js_similarity", "dominant_fate_match"):
        full, no = [float(table.loc[arm, "clone_macro_" + metric]) for arm in ("LR Full", "No-interaction")]
        comparisons.append(dict(metric=metric, full=full, no_interaction=no,
            delta_no_interaction_minus_full=no-full, relative_change=no/full-1, higher_is_better=True))
    clone_path = output / "weinreb_clone_fate.csv"
    pd.DataFrame(comparisons).to_csv(clone_path, index=False)
    rows = []
    indexed = means.set_index(["training_seed", "model", "time", "metric"]).value
    for time in (1., 2.):
        row = dict(time=time, space="pca")
        for metric in ("w1", "w2"):
            for name, arm in (("LR Full", "full"), ("No-interaction", "no_interaction")):
                values = [float(indexed.loc[seed, name, time, metric]) for seed in SEEDS]
                row[f"{metric}_{arm}"] = np.mean(values)
                row.update({f"{metric}_{arm}_seed{seed}": value for seed, value in zip(SEEDS, values)})
            row[f"{metric}_relative_change"] = row[f"{metric}_no_interaction"] / row[f"{metric}_full"] - 1
        rows.append(row)
    dist_path = output / "weinreb_distribution.csv"
    pd.DataFrame(rows).to_csv(dist_path, index=False)
    return dict(distribution=dist_path, clone_fate=clone_path)


def evaluate(archive_dir, output_dir, *, device="cuda"):
    """Repeat 50 distribution simulations and ten clone simulations per model.

    archive_dir is the extracted weinreb-s4-20261029 release archive. No model
    is fitted by this function. Training code is included in that archive.
    """
    import anndata as ad
    from reproduction.nonspatial.model import load_state_model
    from reproduction.nonspatial.analyze import clone_fate
    from CytoBridge.tl.downstream import evaluate_model_distributions
    archive, output = Path(archive_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    data_path = archive / "data/Weinreb_prepared_50pc.h5ad"
    data = ad.read_h5ad(data_path)
    rows, truth = [], None
    for seed in SEEDS:
        full = archive / f"lr_full/lr_seed{seed}_full/model"
        no = archive / f"weinreb-lr-matched-ablation-20261029/seed{seed}/model"
        for name, directory, stage in (("LR Full", full, "Finetune"), ("No-interaction", no, "Finetune_no_interaction")):
            loaded = load_state_model(directory, device=device, stage=stage)
            for inference in range(10000, 10050):
                result = evaluate_model_distributions(data, loaded.model, time_points=[0.,1.,2.],
                    n_samples=2048, dt=.05, sigma=.1, include_score=True, interaction_m=16,
                    max_ot_points=1024, structure_max_points=2048, device=device,
                    time_key="time_point_processed", obsm_key="X_latent", concat_spatial=False,
                    random_seed=inference, noise_seed=inference, include_initial_time=False, verbose=False)
                frame = result.metrics.assign(training_seed=seed, inference_seed=inference, model=name)
                rows.append(frame.melt(id_vars=["training_seed","inference_seed","model","time","space"],
                                       value_vars=["w1","w2"], var_name="metric", value_name="value"))
            loaded.model.to("cpu")
            del loaded
        clone_dir = output / f"clone_seed{seed}"
        clone_fate(data_path, full, no, clone_dir, device=device, seeds=tuple(range(10)))
        for arm in ("full", "no_interaction"):
            observed = pd.read_csv(clone_dir / arm / "observed_fate_distributions.csv", index_col=0)
            if truth is None:
                truth = observed
                truth.to_csv(raw / "observed_fate_distributions.csv")
            pd.testing.assert_frame_equal(observed, truth)
            pred = pd.read_csv(clone_dir / arm / "predicted_fate_distributions.csv", index_col=0)
            pred.to_csv(raw / f"{arm}_seed{seed}_probabilities.csv")
    pd.concat(rows, ignore_index=True).to_csv(raw / "distribution_all_repeats.csv", index=False)
    return summarize(raw, output / "summary")
