#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage: launch_mosta_paired_alpha.sh GPU_ID ALPHA_EXPRESS SHARED_RUN OUTPUT_RUN [SOURCE_H5AD] [LR_DATABASE]

Train and evaluate one MOSTA alpha-expression condition while reusing the
corrected preprocessing from SHARED_RUN/preprocess as read-only input.
EOF
  exit 64
fi

GPU_ID="$1"
ALPHA_EXPRESS="$2"
SHARED_RUN="$(cd "$3" && pwd)"
OUTPUT_RUN="$4"
SOURCE_H5AD="${5:-/data/cytobridge/projects/CytoBridge-ST-1104/workspace/spatial_data/Mouse_embryo_all_stage.h5ad}"
LR_DATABASE="${6:-/data/cytobridge/projects/CytoBridge-ST-1104/workspace/database/CellChatDB.ligrec.mouse.csv}"
ALPHA_SPATIAL="${ALPHA_SPATIAL:-10}"
PYTHON="${CYTOBRIDGE_PYTHON:-/data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python}"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_RUN="$(mkdir -p "$(dirname "$OUTPUT_RUN")" && cd "$(dirname "$OUTPUT_RUN")" && pwd)/$(basename "$OUTPUT_RUN")"
PREPROCESS_DIR="${SHARED_RUN}/preprocess"
# This must be shared by every run directory on the host.  /tmp is
# machine-global for the common server account; sites can choose a persistent
# project directory with CYTOBRIDGE_GPU_LOCK_DIR.
LOCK_DIR="${CYTOBRIDGE_GPU_LOCK_DIR:-/tmp/cytobridge-gpu-locks}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  printf 'FAILED reason=invalid_gpu_id gpu=%s\n' "$GPU_ID" >&2
  exit 64
fi
for required in "$SOURCE_H5AD" "$LR_DATABASE" "$PREPROCESS_DIR/mosta_aligned.h5ad" "$PREPROCESS_DIR/edge_classifier/mosta_edge_model.pt"; do
  if [[ ! -e "$required" ]]; then
    printf 'FAILED reason=missing_required_input path=%s\n' "$required" >&2
    exit 66
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  printf 'FAILED reason=missing_python path=%s\n' "$PYTHON" >&2
  exit 66
fi
COMMIT="$(git -C "$CODE" rev-parse HEAD)"
if [[ -n "$(git --no-optional-locks -C "$CODE" status --porcelain --untracked-files=all)" ]]; then
  printf 'FAILED reason=dirty_code_snapshot commit=%s\n' "$COMMIT" >&2
  exit 65
fi

# Atomically reserve the complete output directory.  A previous failed launch
# is kept as audit evidence and must use a new output path on retry.
if [[ -e "$OUTPUT_RUN" ]] || ! mkdir "$OUTPUT_RUN" 2>/dev/null; then
  printf 'REFUSED reason=run_output_already_exists run=%s\n' "$OUTPUT_RUN" >&2
  exit 73
fi

mkdir -p "$LOCK_DIR" "$OUTPUT_RUN/logs" "$OUTPUT_RUN/status" \
  "$OUTPUT_RUN/cache/numba" "$OUTPUT_RUN/cache/matplotlib" "$OUTPUT_RUN/cache/xdg"
exec 9>"$LOCK_DIR/gpu${GPU_ID}.lock"
if ! flock -n 9; then
  printf 'REFUSED reason=run_local_gpu_lock_busy gpu=%s time=%s\n' \
    "$GPU_ID" "$(date --iso-8601=seconds)" >"$OUTPUT_RUN/status/formal.status"
  exit 75
fi

GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F',' -v wanted="$GPU_ID" '{ idx=$1; gsub(/[[:space:]]/, "", idx); if (idx == wanted) { uuid=$2; gsub(/[[:space:]]/, "", uuid); print uuid } }')"
if [[ -z "$GPU_UUID" ]]; then
  printf 'FAILED reason=unknown_gpu gpu=%s time=%s\n' \
    "$GPU_ID" "$(date --iso-8601=seconds)" >"$OUTPUT_RUN/status/formal.status"
  exit 64
fi
BUSY_PIDS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F',' -v wanted="$GPU_UUID" '{ uuid=$1; gsub(/[[:space:]]/, "", uuid); if (uuid == wanted) { pid=$2; gsub(/[[:space:]]/, "", pid); print pid } }')"
if [[ -n "$BUSY_PIDS" ]]; then
  printf 'REFUSED reason=gpu_has_compute_process gpu=%s pids=%s time=%s\n' \
    "$GPU_ID" "${BUSY_PIDS//$'\n'/,}" "$(date --iso-8601=seconds)" \
    >"$OUTPUT_RUN/status/formal.status"
  exit 75
fi

printf 'RUNNING pid=%s gpu=%s alpha_spatial=%s alpha_express=%s shared_preprocess=%s commit=%s time=%s\n' \
  "$$" "$GPU_ID" "$ALPHA_SPATIAL" "$ALPHA_EXPRESS" "$PREPROCESS_DIR" "$COMMIT" \
  "$(date --iso-8601=seconds)" >"$OUTPUT_RUN/status/formal.status"

CHILD_PID=""
FINAL_STATUS_WRITTEN=0
finalize_interrupted_run() {
  local rc="$?"
  if [[ "$FINAL_STATUS_WRITTEN" -eq 0 ]]; then
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
      kill -TERM "$CHILD_PID" 2>/dev/null || true
      wait "$CHILD_PID" 2>/dev/null || true
    fi
    printf 'INTERRUPTED exit=%s gpu=%s alpha_spatial=%s alpha_express=%s shared_preprocess=%s commit=%s time=%s\n' \
      "$rc" "$GPU_ID" "$ALPHA_SPATIAL" "$ALPHA_EXPRESS" "$PREPROCESS_DIR" "$COMMIT" \
      "$(date --iso-8601=seconds)" >"$OUTPUT_RUN/status/formal.status"
  fi
}
trap finalize_interrupted_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CYTOBRIDGE_ASSIGNED_GPU="$GPU_ID"
export PYTHONPATH="$CODE"
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="$OUTPUT_RUN/cache/numba"
export MPLCONFIGDIR="$OUTPUT_RUN/cache/matplotlib"
export XDG_CACHE_HOME="$OUTPUT_RUN/cache/xdg"
export JUPYTER_PLATFORM_DIRS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

"$PYTHON" "$CODE/scripts/run_mosta_end_to_end.py" \
  --h5ad-path "$SOURCE_H5AD" \
  --database-path "$LR_DATABASE" \
  --output-dir "$OUTPUT_RUN" \
  --reuse-preprocess-dir "$PREPROCESS_DIR" \
  --profile full \
  --stage train-evaluate \
  --device cuda \
  --random-seed 42 \
  --alpha-spatial "$ALPHA_SPATIAL" \
  --alpha-express "$ALPHA_EXPRESS" \
  --interaction-m 1024 \
  --evaluation-max-ot-points 1024 \
  --evaluation-dt 0.01 \
  >>"$OUTPUT_RUN/logs/formal_pipeline.log" 2>&1 &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
RC=$?
set -e

if [[ "$RC" -eq 0 ]]; then
  printf 'COMPLETE exit=0 gpu=%s alpha_spatial=%s alpha_express=%s shared_preprocess=%s commit=%s time=%s\n' \
    "$GPU_ID" "$ALPHA_SPATIAL" "$ALPHA_EXPRESS" "$PREPROCESS_DIR" "$COMMIT" \
    "$(date --iso-8601=seconds)" >"$OUTPUT_RUN/status/formal.status"
else
  printf 'FAILED exit=%s gpu=%s alpha_spatial=%s alpha_express=%s shared_preprocess=%s commit=%s time=%s\n' \
    "$RC" "$GPU_ID" "$ALPHA_SPATIAL" "$ALPHA_EXPRESS" "$PREPROCESS_DIR" "$COMMIT" \
    "$(date --iso-8601=seconds)" >"$OUTPUT_RUN/status/formal.status"
fi
FINAL_STATUS_WRITTEN=1
exit "$RC"
