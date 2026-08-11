#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-7}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  printf 'FAILED reason=invalid_gpu_id gpu=%s time=%s\n' \
    "${GPU_ID}" "$(date --iso-8601=seconds)" >&2
  exit 64
fi
BASE="/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-api/corrected-clean-counts-alpha0015-v1"
CODE="${BASE}/software/cytobridge-spatial-5d49b85"
RUN="${BASE}/formal_5d49b85_alignfull"
PYTHON="/data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python"
SOURCE="/data/cytobridge/projects/CytoBridge-ST-1104/workspace/spatial_data/Mouse_embryo_all_stage.h5ad"
DATABASE="/data/cytobridge/projects/CytoBridge-ST-1104/workspace/database/CellChatDB.ligrec.mouse.csv"
EXPECTED_COMMIT="5d49b85398fe2aaefea3052fffa8935571bb0d5d"

mkdir -p \
  "${BASE}/locks" \
  "${RUN}/logs" \
  "${RUN}/status" \
  "${RUN}/cache/numba" \
  "${RUN}/cache/xdg" \
  "${RUN}/cache/matplotlib"
exec 9>"${BASE}/locks/gpu${GPU_ID}.lock"
if ! flock -n 9; then
  printf 'REFUSED reason=run_local_gpu_lock_busy gpu=%s time=%s\n' \
    "${GPU_ID}" "$(date --iso-8601=seconds)" >"${RUN}/status/formal.status"
  exit 75
fi

if [[ ! -f "${SOURCE}" || ! -f "${DATABASE}" || ! -x "${PYTHON}" ]]; then
  printf 'FAILED reason=missing_required_input time=%s\n' \
    "$(date --iso-8601=seconds)" >"${RUN}/status/formal.status"
  exit 66
fi

ACTUAL_COMMIT="$(git -C "${CODE}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  printf 'FAILED reason=code_commit_mismatch expected=%s actual=%s time=%s\n' \
    "${EXPECTED_COMMIT}" "${ACTUAL_COMMIT}" "$(date --iso-8601=seconds)" \
    >"${RUN}/status/formal.status"
  exit 65
fi
if [[ -n "$(git --no-optional-locks -C "${CODE}" status --porcelain --untracked-files=all)" ]]; then
  printf 'FAILED reason=dirty_code_snapshot commit=%s time=%s\n' \
    "${ACTUAL_COMMIT}" "$(date --iso-8601=seconds)" \
    >"${RUN}/status/formal.status"
  exit 65
fi

GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F',' -v wanted="${GPU_ID}" '{ idx=$1; gsub(/[[:space:]]/, "", idx); if (idx == wanted) { uuid=$2; gsub(/[[:space:]]/, "", uuid); print uuid } }')"
if [[ -z "${GPU_UUID}" ]]; then
  printf 'FAILED reason=unknown_gpu gpu=%s time=%s\n' \
    "${GPU_ID}" "$(date --iso-8601=seconds)" >"${RUN}/status/formal.status"
  exit 64
fi

BUSY_PIDS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F',' -v wanted="${GPU_UUID}" '{ uuid=$1; gsub(/[[:space:]]/, "", uuid); if (uuid == wanted) { pid=$2; gsub(/[[:space:]]/, "", pid); print pid } }')"
if [[ -n "${BUSY_PIDS}" ]]; then
  printf 'REFUSED reason=gpu_has_compute_process gpu=%s pids=%s time=%s\n' \
    "${GPU_ID}" "${BUSY_PIDS//$'\n'/,}" "$(date --iso-8601=seconds)" \
    >"${RUN}/status/formal.status"
  exit 75
fi

if pgrep -af "run_mosta_end_to_end.py" | grep -F -- "${RUN}" >/dev/null 2>&1; then
  printf 'REFUSED reason=duplicate_run_detected run=%s time=%s\n' \
    "${RUN}" "$(date --iso-8601=seconds)" >"${RUN}/status/formal.status"
  exit 75
fi
if [[ -e "${RUN}/run_manifest.json" \
   || -d "${RUN}/input_contract" \
   || -d "${RUN}/preprocess" \
   || -d "${RUN}/training" \
   || -d "${RUN}/evaluation" \
   || -d "${RUN}/downstream" ]]; then
  printf 'REFUSED reason=run_output_already_exists run=%s time=%s\n' \
    "${RUN}" "$(date --iso-8601=seconds)" >"${RUN}/status/formal.status"
  exit 73
fi

printf 'RUNNING pid=%s gpu=%s alignment_max_cells_per_timepoint=none commit=%s time=%s\n' \
  "$$" "${GPU_ID}" "${ACTUAL_COMMIT}" "$(date --iso-8601=seconds)" \
  >"${RUN}/status/formal.status"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CYTOBRIDGE_ASSIGNED_GPU="${GPU_ID}"
export PYTHONPATH="${CODE}"
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="${RUN}/cache/numba"
export XDG_CACHE_HOME="${RUN}/cache/xdg"
export MPLCONFIGDIR="${RUN}/cache/matplotlib"
export JUPYTER_PLATFORM_DIRS=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

set +e
"${PYTHON}" "${CODE}/scripts/run_mosta_end_to_end.py" \
  --h5ad-path "${SOURCE}" \
  --database-path "${DATABASE}" \
  --output-dir "${RUN}" \
  --profile full \
  --stage all \
  --device cuda \
  --random-seed 42 \
  --interaction-m 1024 \
  --evaluation-max-ot-points 1024 \
  --evaluation-dt 0.01 \
  >>"${RUN}/logs/formal_pipeline.log" 2>&1
RC=$?
set -e

if [[ "${RC}" -eq 0 ]]; then
  printf 'COMPLETE exit=0 gpu=%s alignment_max_cells_per_timepoint=none commit=%s time=%s\n' \
    "${GPU_ID}" "${ACTUAL_COMMIT}" "$(date --iso-8601=seconds)" \
    >"${RUN}/status/formal.status"
else
  printf 'FAILED exit=%s gpu=%s alignment_max_cells_per_timepoint=none commit=%s time=%s\n' \
    "${RC}" "${GPU_ID}" "${ACTUAL_COMMIT}" "$(date --iso-8601=seconds)" \
    >"${RUN}/status/formal.status"
fi
exit "${RC}"
