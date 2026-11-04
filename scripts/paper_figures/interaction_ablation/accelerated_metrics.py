"""Evaluate the same POT sliced-W2 in float64 projection batches on the GPU."""

import math
import numpy as np
import ot
import torch


def sliced_w2(predicted, observed, weights, projections, *, device="cuda:0", batch_size=64):
    x = torch.as_tensor(predicted, dtype=torch.float64, device=device)
    y = torch.as_tensor(observed, dtype=torch.float64, device=device)
    a = torch.as_tensor(weights, dtype=torch.float64, device=device)
    b = torch.full((len(observed),), 1. / len(observed), dtype=torch.float64, device=device)
    total = 0.
    with torch.no_grad():
        for start in range(0, projections.shape[1], batch_size):
            directions = torch.as_tensor(projections[:, start:start + batch_size], dtype=torch.float64, device=device)
            distance = ot.sliced_wasserstein_distance(x, y, a=a, b=b, p=2, projections=directions)
            total += float(distance.item()) ** 2 * directions.shape[1]
    return math.sqrt(max(0., total / projections.shape[1]))


def metric_rows(core, *, dataset, arm, target, points, weights, truth, training, transform, device="cuda:0"):
    predicted_state = (points[:, 2:].astype(np.float64) - transform["state_center"]) / transform["state_scale"] / math.sqrt(50)
    observed_state = (truth["state"].astype(np.float64) - transform["state_center"]) / transform["state_scale"] / math.sqrt(50)
    predicted_spatial = (points[:, :2].astype(np.float64) - transform["spatial_center"]) / float(transform["spatial_rms"]) / math.sqrt(2)
    observed_spatial = (truth["spatial"].astype(np.float64) - transform["spatial_center"]) / float(transform["spatial_rms"]) / math.sqrt(2)
    spaces = {"joint": (np.concatenate((predicted_state, predicted_spatial), 1), np.concatenate((observed_state, observed_spatial), 1)),
              "state": (predicted_state, observed_state), "spatial": (predicted_spatial, observed_spatial)}
    raw_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    normalized = raw_weights / raw_weights.sum()
    source_n = int(np.count_nonzero(np.isclose(training["time"], np.min(training["time"]), rtol=0., atol=1e-8)))
    observed_mass = len(truth["state"]) / source_n
    predicted_mass = raw_weights.sum()
    tmv_absolute = abs(predicted_mass - observed_mass)
    rows = []
    for space, (predicted, observed) in spaces.items():
        for repeat in range(core.PROJECTION_REPEATS):
            seed = core._projection_seed(dataset, space, repeat)
            directions = np.random.RandomState(seed).randn(predicted.shape[1], core.N_PROJECTIONS)
            directions /= np.sqrt(np.sum(directions ** 2, axis=0, keepdims=True))
            distance = sliced_w2(predicted, observed, normalized, directions, device=device)
            rows.append({"dataset": dataset, "arm": arm, "target": float(target), "space": space,
                         "projection_repeat": repeat, "projection_seed": seed,
                         "projection_sha256": core._projection_sha256(predicted.shape[1], seed),
                         "n_projections": core.N_PROJECTIONS, "sliced_w2": distance,
                         "n_predicted": len(predicted), "n_observed": len(observed),
                         "predicted_mass": predicted_mass, "observed_mass_relative": observed_mass,
                         "tmv_absolute": tmv_absolute, "tmv": tmv_absolute / observed_mass,
                         "weights_semantics": "native_unnormalised_growth_mass"})
    return rows


if __name__ == "__main__":
    import json
    import time
    torch.set_num_threads(4)
    rows = []
    for dim, nx, ny in [(2, 311, 733), (50, 817, 1027), (52, 5000, 10000)]:
        rng = np.random.RandomState(42 + dim)
        x = rng.randn(nx, dim)
        y = rng.randn(ny, dim) + .15
        weights = rng.lognormal(size=nx); weights /= weights.sum()
        projections = rng.randn(dim, 1024)
        projections /= np.sqrt((projections ** 2).sum(0, keepdims=True))
        begin = time.monotonic()
        expected = ot.sliced_wasserstein_distance(x, y, a=weights, b=np.full(ny, 1/ny), p=2, projections=projections)
        cpu_seconds = time.monotonic() - begin
        begin = time.monotonic()
        observed = sliced_w2(x, y, weights, projections)
        gpu_seconds = time.monotonic() - begin
        np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-10)
        rows.append({"dimension": dim, "predicted_n": nx, "observed_n": ny,
                     "numpy_pot": float(expected), "gpu_batched_pot": observed,
                     "absolute_difference": abs(float(expected)-observed),
                     "cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds})
    print(json.dumps({"passed": True, "comparisons": rows}, indent=2))

