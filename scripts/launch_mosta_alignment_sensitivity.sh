#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-3}"
if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "GPU index must be a non-negative integer: ${GPU_INDEX}" >&2
  exit 64
fi
RUN_ROOT="/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-api/corrected-clean-counts-alpha0015-v1"
FORMAL_DIR="${RUN_ROOT}/formal_5d49b85_align20000"
SOFTWARE_DIR="${RUN_ROOT}/software"
PYTHON="/data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python"
SNAPSHOT="${SOFTWARE_DIR}/cytobridge-spatial-5d49b85"
EXPECTED_COMMIT="5d49b85398fe2aaefea3052fffa8935571bb0d5d"
SCRIPT="${SOFTWARE_DIR}/run_mosta_alignment_sensitivity.py"
INPUT_H5AD="${FORMAL_DIR}/preprocess/mosta_aligned.h5ad"
OUTPUT_DIR="${FORMAL_DIR}/alignment_sensitivity_full_fit"
LOG_DIR="${FORMAL_DIR}/logs"
STATUS_DIR="${FORMAL_DIR}/status"
LOG_FILE="${LOG_DIR}/alignment_sensitivity_full_fit.log"
STATUS_FILE="${STATUS_DIR}/alignment_sensitivity_full_fit.status"
LOCK_FILE="${STATUS_DIR}/alignment_sensitivity_full_fit.lock"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${RUN_ROOT}/locks"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another alignment-sensitivity process owns ${LOCK_FILE}" >&2
  exit 75
fi
exec 8>"${RUN_ROOT}/locks/gpu${GPU_INDEX}.lock"
if ! flock -n 8; then
  echo "Shared run-local GPU lock is busy for GPU ${GPU_INDEX}" >&2
  exit 75
fi

if [[ ! -s "${INPUT_H5AD}" ]]; then
  echo "Missing formal aligned H5AD: ${INPUT_H5AD}" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing sensitivity script: ${SCRIPT}" >&2
  exit 2
fi
if [[ ! -d "${SNAPSHOT}/.git" ]]; then
  echo "Missing immutable source snapshot: ${SNAPSHOT}" >&2
  exit 2
fi
ACTUAL_COMMIT="$(git -C "${SNAPSHOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Snapshot commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 65
fi
if [[ -n "$(git --no-optional-locks -C "${SNAPSHOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing dirty source snapshot: ${SNAPSHOT}" >&2
  exit 65
fi

GPU_UUID="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader; } | awk -F ', ' -v idx="${GPU_INDEX}" '$1 == idx {print $2}')"
if [[ -z "${GPU_UUID}" ]]; then
  echo "GPU index ${GPU_INDEX} was not found" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "${GPU_UUID}"; then
  echo "GPU ${GPU_INDEX} (${GPU_UUID}) already has a compute process" >&2
  exit 76
fi

if [[ -d "${OUTPUT_DIR}" ]] && find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 73
fi

SCRIPT_SHA256="$(sha256sum "${SCRIPT}" | awk '{print $1}')"
START_TIME="$(date --iso-8601=seconds)"
echo "RUNNING pid=$$ gpu=${GPU_INDEX} gpu_uuid=${GPU_UUID} commit=${ACTUAL_COMMIT} script_sha256=${SCRIPT_SHA256} time=${START_TIME}" > "${STATUS_FILE}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SNAPSHOT}"
export NUMBA_CACHE_DIR="${FORMAL_DIR}/cache/numba_alignment_sensitivity"
export MPLCONFIGDIR="${FORMAL_DIR}/cache/matplotlib_alignment_sensitivity"
export XDG_CACHE_HOME="${FORMAL_DIR}/cache/xdg_alignment_sensitivity"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

set +e
"${PYTHON}" "${SCRIPT}" \
  --aligned-h5ad "${INPUT_H5AD}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --knn-k 10 \
  --ot-sample-per-timepoint 1024 \
  --plot-sample-per-timepoint 2500 \
  >> "${LOG_FILE}" 2>&1
RC=$?
set -e

END_TIME="$(date --iso-8601=seconds)"
if [[ ${RC} -eq 0 ]]; then
  echo "COMPLETED rc=0 gpu=${GPU_INDEX} commit=${ACTUAL_COMMIT} script_sha256=${SCRIPT_SHA256} time=${END_TIME}" > "${STATUS_FILE}"
else
  echo "FAILED rc=${RC} gpu=${GPU_INDEX} commit=${ACTUAL_COMMIT} script_sha256=${SCRIPT_SHA256} time=${END_TIME}" > "${STATUS_FILE}"
fi
exit "${RC}"
