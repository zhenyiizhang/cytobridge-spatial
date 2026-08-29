#!/usr/bin/env bash
set -euo pipefail

project=/data/cytobridge/projects/CytoBridge-ST-1104
runtime="$project/software/cytobridge-release-2b3c79e-runtime"
python_bin="$project/envs/arista-api/bin/python"
runner=/tmp/server_compute_mosta_si_shared_20260825_v1.py
aligned="$project/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta/preprocess/mosta_aligned.h5ad"
model="$project/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta/training"
classifier="$project/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta/downstream/classifier_cache/classifier_resmlp_6d2d7acf7d0ed92d.pt"
output="$project/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1"
log=/tmp/mosta_si_shared_20260825_v1.log
package_commit=2b3c79eff3face7c4dd33de24d45384b9dbd8a84
input_sha=8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25

if [[ -e "$output" ]]; then
    echo "REFUSE_EXISTING output=$output" >&2
    exit 90
fi
if [[ ! -f "$runner" ]]; then
    echo "MISSING_RUNNER runner=$runner" >&2
    exit 91
fi
detected_commit=$(tr -d '[:space:]' <"$runtime/RELEASE_COMMIT")
if [[ "$detected_commit" != "$package_commit" ]]; then
    echo "RELEASE_COMMIT_MISMATCH detected=$detected_commit" >&2
    exit 92
fi

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$runtime" \
MPLCONFIGDIR=/tmp/mosta_si_shared_mpl_20260825_v1 \
NUMBA_CACHE_DIR=/tmp/mosta_si_shared_numba_20260825_v1 \
PYTHONUNBUFFERED=1 \
"$python_bin" "$runner" \
    --aligned-h5ad "$aligned" \
    --model-dir "$model" \
    --classifier-cache-path "$classifier" \
    --output-dir "$output" \
    --device cuda:0 \
    --package-commit "$package_commit" \
    --expected-input-sha256 "$input_sha" \
    --n-samples 50000 \
    --seed 42 >"$log" 2>&1

echo "COMPLETE output=$output log=$log"
