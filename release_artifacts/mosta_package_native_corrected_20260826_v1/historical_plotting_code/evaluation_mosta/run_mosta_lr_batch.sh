#!/usr/bin/env bash
set -euo pipefail

# Batch runner for MOSTA LR workflow:
# 1) Real timepoints: LR projection + per-LR plotting.
# 2) Interpolated timepoints: split-SDE->gene + LR projection + per-LR plotting.
#
# Usage examples:
#   bash evaluation/mosta/code/run_mosta_lr_batch.sh
#   bash evaluation/mosta/code/run_mosta_lr_batch.sh --only-interp --target-times "0,0.5,1,1.5,2,3"
#   bash evaluation/mosta/code/run_mosta_lr_batch.sh --lr-pairs "Wnt3a_Fzd7_Lrp6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="DeepRUOTv2"

RUN_REAL=1
RUN_INTERP=1

TARGET_TIME="0.5"
TARGET_TIMES="0,0.5,1,1.5,2,3"
PROJECT_TIME_KEYS=""
TS_POINTS="0.0,0.5,1.0,1.5,2.0,3.0"
START_TIME="0.0"
N_SAMPLES="20000"

LR_PAIRS="Wnt3a_Fzd7_Lrp6"
REAL_TIMES="E12.5,E13.5,E14.5,E15.5"

REAL_COMM_PKL="evaluation/mosta/data/all_timepoint_communications_merged.pkl"
INTERP_COMM_PKL="results/mosta_interp_0_3_0208_n_pc_12/mosta_all_time_communications.pkl"
REAL_H5AD="spatial_data/Mouse_embryo_all_stage.h5ad"
INTERP_DATA_CSV="evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv"
INTERP_CLASSIFIER_DATA_CSV=""
INTERP_CLASSIFIER_CACHE_DIR="results/mosta_interp_0_3_0208_n_pc_12"
INTERP_CLASSIFIER_N_PCS="12"
INTERP_CLASSIFIER_BEST_METRIC="bacc"
INTERP_CLASSIFIER_TRAIN_ON_FULL_DATA=1
INTERP_SPATIAL_WARP_TO_OBS_PIECEWISE=0
INTERP_SPATIAL_WARP_K="8"
INTERP_SPATIAL_WARP_EPS="1e-6"
TIME_KEY_MAP="0=E12.5,1=E13.5,2=E14.5,3=E15.5"
PCA_COMPONENTS_CSV="mosta_pca_components_with_gene_names.csv"
PCA_MEAN_CSV="mosta_pca_mean.csv"
LR_DB="database/CellChatDB.ligrec.mouse.csv"

BASE_OUT="results/mosta_lr_batch_0208"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --env NAME                      Conda env name (default: ${ENV_NAME})
  --only-real                     Run only real timepoints
  --only-interp                   Run only interpolated timepoint
  --target-time T                 Interp target time (default: ${TARGET_TIME})
  --target-times CSV              Interp target times CSV (default: ${TARGET_TIMES})
  --project-time-keys CSV         Time keys to project/plot (default: interpolated keys only)
  --ts-points CSV                 Interp ts points (default: ${TS_POINTS})
  --start-time T                  Interp start time (default: ${START_TIME})
  --n-samples N                   Interp initial sample size (default: ${N_SAMPLES})
  --lr-pairs CSV                  LR pairs (default: ${LR_PAIRS})
  --real-times CSV                Real time labels (default: ${REAL_TIMES})
  --data-csv PATH                 Interp data CSV (default: ${INTERP_DATA_CSV})
  --classifier-data-csv PATH      Optional classifier CSV when --data-csv has no Annotation
  --interp-classifier-cache-dir P Classifier cache dir (or run root containing classifier_cache/)
  --interp-classifier-n-pcs N     Classifier PCs for interpolation script (default: ${INTERP_CLASSIFIER_N_PCS})
  --interp-classifier-best-metric M  accuracy|bacc (default: ${INTERP_CLASSIFIER_BEST_METRIC})
  --interp-classifier-train-on-full-data / --no-interp-classifier-train-on-full-data
  --interp-pca-mean-file PATH      PCA mean CSV/TSV/NPY for interpolation backprojection (default: ${PCA_MEAN_CSV})
  --interp-spatial-warp-to-observed-piecewise / --no-interp-spatial-warp-to-observed-piecewise
  --interp-spatial-warp-k K       Piecewise warp kNN neighbors (default: ${INTERP_SPATIAL_WARP_K})
  --interp-spatial-warp-eps E     Piecewise warp IDW epsilon (default: ${INTERP_SPATIAL_WARP_EPS})
  --time-key-map MAP              comm->real mapping (default: ${TIME_KEY_MAP})
  --real-comm-pkl PATH            Real communication pickle
  --interp-comm-pkl PATH          Interp communication pickle (must contain target time key)
  --real-h5ad PATH                Real h5ad for real-time plotting/projection
  --lr-db PATH                    LR database CSV
  --base-out DIR                  Base output directory (default: ${BASE_OUT})
  -h, --help                      Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --only-real) RUN_REAL=1; RUN_INTERP=0; shift ;;
    --only-interp) RUN_REAL=0; RUN_INTERP=1; shift ;;
    --target-time) TARGET_TIME="$2"; shift 2 ;;
    --target-times) TARGET_TIMES="$2"; shift 2 ;;
    --project-time-keys) PROJECT_TIME_KEYS="$2"; shift 2 ;;
    --ts-points) TS_POINTS="$2"; shift 2 ;;
    --start-time) START_TIME="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --lr-pairs) LR_PAIRS="$2"; shift 2 ;;
    --real-times) REAL_TIMES="$2"; shift 2 ;;
    --data-csv) INTERP_DATA_CSV="$2"; shift 2 ;;
    --classifier-data-csv) INTERP_CLASSIFIER_DATA_CSV="$2"; shift 2 ;;
    --interp-classifier-cache-dir) INTERP_CLASSIFIER_CACHE_DIR="$2"; shift 2 ;;
    --interp-classifier-n-pcs) INTERP_CLASSIFIER_N_PCS="$2"; shift 2 ;;
    --interp-classifier-best-metric) INTERP_CLASSIFIER_BEST_METRIC="$2"; shift 2 ;;
    --interp-classifier-train-on-full-data) INTERP_CLASSIFIER_TRAIN_ON_FULL_DATA=1; shift ;;
    --no-interp-classifier-train-on-full-data) INTERP_CLASSIFIER_TRAIN_ON_FULL_DATA=0; shift ;;
    --interp-pca-mean-file) PCA_MEAN_CSV="$2"; shift 2 ;;
    --interp-spatial-warp-to-observed-piecewise) INTERP_SPATIAL_WARP_TO_OBS_PIECEWISE=1; shift ;;
    --no-interp-spatial-warp-to-observed-piecewise) INTERP_SPATIAL_WARP_TO_OBS_PIECEWISE=0; shift ;;
    --interp-spatial-warp-k) INTERP_SPATIAL_WARP_K="$2"; shift 2 ;;
    --interp-spatial-warp-eps) INTERP_SPATIAL_WARP_EPS="$2"; shift 2 ;;
    --time-key-map) TIME_KEY_MAP="$2"; shift 2 ;;
    --real-comm-pkl) REAL_COMM_PKL="$2"; shift 2 ;;
    --interp-comm-pkl) INTERP_COMM_PKL="$2"; shift 2 ;;
    --real-h5ad) REAL_H5AD="$2"; shift 2 ;;
    --lr-db) LR_DB="$2"; shift 2 ;;
    --base-out) BASE_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

run_py() {
  echo "+ conda run -n ${ENV_NAME} python $*"
  conda run -n "${ENV_NAME}" python "$@"
}

normalize_time_key() {
  python3 - "$1" <<'PY'
import sys
print(str(float(sys.argv[1])))
PY
}

format_time_token() {
  python3 - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}".replace(".", "p").replace("-", "n"))
PY
}

resolve_classifier_cache_dir() {
  local input_dir="$1"
  if [[ -d "${input_dir}/classifier_cache" ]]; then
    echo "${input_dir}/classifier_cache"
  else
    echo "${input_dir}"
  fi
}

map_time_key() {
  python3 - "$1" "$2" <<'PY'
import sys

def norm(s: str) -> str:
    raw = str(s).strip()
    try:
        return str(float(raw))
    except Exception:
        return raw

key = norm(sys.argv[1])
mapping = str(sys.argv[2]).strip()
kv = {}
if mapping:
    for part in mapping.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        src, dst = part.split("=", 1)
        src = norm(src)
        dst = dst.strip()
        if dst:
            kv[src] = dst
print(kv.get(key, key))
PY
}

IFS=',' read -r -a LR_PAIR_ARR <<< "${LR_PAIRS}"
IFS=',' read -r -a REAL_TIME_ARR <<< "${REAL_TIMES}"
if [[ -z "${TARGET_TIMES}" ]]; then
  TARGET_TIMES="${TARGET_TIME}"
fi
IFS=',' read -r -a TARGET_TIME_ARR <<< "${TARGET_TIMES}"

mkdir -p "${BASE_OUT}"

if [[ "${RUN_REAL}" == "1" ]]; then
  REAL_PROJ_DIR="${BASE_OUT}/real_projection"
  REAL_PLOT_DIR="${BASE_OUT}/real_plots"
  mkdir -p "${REAL_PROJ_DIR}" "${REAL_PLOT_DIR}"

  run_py evaluation/mosta/code/mosta_lr_projection_local.py \
    --communications-pkl "${REAL_COMM_PKL}" \
    --real-h5ad "${REAL_H5AD}" \
    --lr-db "${LR_DB}" \
    --time-keys "$(IFS=,; echo "${REAL_TIME_ARR[*]}")" \
    --annotation-col annotation \
    --x-space log1p \
    --save-key-mode mapped \
    --output-dir "${REAL_PROJ_DIR}"

  for tk in "${REAL_TIME_ARR[@]}"; do
    score_pkl="${REAL_PROJ_DIR}/lr_scores_${tk}.pkl"
    for pair in "${LR_PAIR_ARR[@]}"; do
      run_py evaluation/mosta/code/mosta_lr_single_timepoint_nature_local.py \
        --scores-pkl "${score_pkl}" \
        --adata-h5ad "${REAL_H5AD}" \
        --lr-db "${LR_DB}" \
        --time-key "${tk}" \
        --lr-pair "${pair}" \
        --annotation-col annotation \
        --x-space log1p \
        --output-dir "${REAL_PLOT_DIR}" \
        --invert-y
    done
  done
fi

if [[ "${RUN_INTERP}" == "1" ]]; then
  INTERP_CACHE_DIR_RESOLVED="$(resolve_classifier_cache_dir "${INTERP_CLASSIFIER_CACHE_DIR}")"
  INTERP_H5AD_ROOT="${BASE_OUT}/interp_h5ad_0208_n${N_SAMPLES}"
  INTERP_MAP_JSON="${INTERP_H5AD_ROOT}/interp_map.json"
  INTERP_MAP_TSV="${INTERP_H5AD_ROOT}/interp_map.tsv"
  INTERP_PROJ_DIR="${BASE_OUT}/interp_projection_0208_n${N_SAMPLES}"
  INTERP_PLOT_DIR="${BASE_OUT}/interp_plots_0208_n${N_SAMPLES}"
  mkdir -p "${INTERP_H5AD_ROOT}" "${INTERP_PROJ_DIR}" "${INTERP_PLOT_DIR}"
  : > "${INTERP_MAP_TSV}"

  INTERP_TIME_KEYS=()
  for target in "${TARGET_TIME_ARR[@]}"; do
    t="$(echo "${target}" | tr -d '[:space:]')"
    if [[ -z "${t}" ]]; then
      continue
    fi
    t_key="$(normalize_time_key "${t}")"
    t_tok="$(format_time_token "${t}")"
    INTERP_DIR="${INTERP_H5AD_ROOT}/t${t_tok}"
    INTERP_H5AD="${INTERP_DIR}/adata_t${t_tok}_with_genes.h5ad"
    mkdir -p "${INTERP_DIR}"

    interp_cmd=(
      evaluation/mosta/code/mosta_interp_sde_to_gene_local.py
      --data-csv "${INTERP_DATA_CSV}"
      --annotation-col Annotation
      --pca-components-csv "${PCA_COMPONENTS_CSV}"
      --pca-mean-file "${PCA_MEAN_CSV}"
      --target-time "${t}"
      --ts-points "${TS_POINTS}"
      --start-time "${START_TIME}"
      --n-samples "${N_SAMPLES}"
      --classifier-n-pcs "${INTERP_CLASSIFIER_N_PCS}"
      --classifier-best-metric "${INTERP_CLASSIFIER_BEST_METRIC}"
      --classifier-cache-dir "${INTERP_CACHE_DIR_RESOLVED}"
      --output-x-space count
      --count-transform clip
      --output-dir "${INTERP_DIR}"
    )
    if [[ "${INTERP_CLASSIFIER_TRAIN_ON_FULL_DATA}" == "1" ]]; then
      interp_cmd+=(--classifier-train-on-full-data)
    else
      interp_cmd+=(--no-classifier-train-on-full-data)
    fi
    if [[ "${INTERP_SPATIAL_WARP_TO_OBS_PIECEWISE}" == "1" ]]; then
      interp_cmd+=(
        --spatial-warp-to-observed-piecewise
        --spatial-warp-k "${INTERP_SPATIAL_WARP_K}"
        --spatial-warp-eps "${INTERP_SPATIAL_WARP_EPS}"
      )
    fi
    if [[ -n "${INTERP_CLASSIFIER_DATA_CSV}" ]]; then
      interp_cmd+=(--classifier-data-csv "${INTERP_CLASSIFIER_DATA_CSV}")
    fi
    run_py "${interp_cmd[@]}"

    printf "%s\t%s\n" "${t_key}" "${INTERP_H5AD}" >> "${INTERP_MAP_TSV}"
    INTERP_TIME_KEYS+=("${t_key}")
  done

  if [[ ${#INTERP_TIME_KEYS[@]} -eq 0 ]]; then
    echo "No valid target times found in --target-times/--target-time" >&2
    exit 1
  fi

  PROJ_TIME_KEYS_ARR=()
  if [[ -n "${PROJECT_TIME_KEYS}" ]]; then
    IFS=',' read -r -a PROJ_TIME_KEYS_RAW_ARR <<< "${PROJECT_TIME_KEYS}"
    for tk_raw in "${PROJ_TIME_KEYS_RAW_ARR[@]}"; do
      tk="$(echo "${tk_raw}" | tr -d '[:space:]')"
      if [[ -z "${tk}" ]]; then
        continue
      fi
      PROJ_TIME_KEYS_ARR+=("$(normalize_time_key "${tk}")")
    done
  else
    PROJ_TIME_KEYS_ARR=("${INTERP_TIME_KEYS[@]}")
  fi
  if [[ ${#PROJ_TIME_KEYS_ARR[@]} -eq 0 ]]; then
    echo "No valid projection time keys found." >&2
    exit 1
  fi

  python3 - "${INTERP_MAP_TSV}" "${INTERP_MAP_JSON}" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
mapping = {}
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        k, v = line.split("\t", 1)
        mapping[str(k)] = str(v)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
print("Saved interp map:", dst)
PY

  PROJ_TIME_KEYS_CSV="$(IFS=,; echo "${PROJ_TIME_KEYS_ARR[*]}")"
  proj_cmd=(
    evaluation/mosta/code/mosta_lr_projection_local.py
    --communications-pkl "${INTERP_COMM_PKL}"
    --lr-db "${LR_DB}"
    --interp-map-json "${INTERP_MAP_JSON}"
    --time-keys "${PROJ_TIME_KEYS_CSV}"
    --annotation-col Annotation
    --x-space count
    --save-key-mode comm
    --output-dir "${INTERP_PROJ_DIR}"
  )
  if [[ -n "${TIME_KEY_MAP}" ]]; then
    proj_cmd+=(--time-key-map "${TIME_KEY_MAP}")
  fi
  run_py "${proj_cmd[@]}"

  for tk in "${PROJ_TIME_KEYS_ARR[@]}"; do
    score_pkl="${INTERP_PROJ_DIR}/lr_scores_${tk}.pkl"
    interp_adata_h5ad="$(awk -F $'\t' -v k="${tk}" '$1==k{print $2}' "${INTERP_MAP_TSV}" | head -n 1)"
    if [[ -n "${interp_adata_h5ad}" ]]; then
      adata_h5ad="${interp_adata_h5ad}"
      plot_annotation_col="Annotation"
      plot_time_key="${tk}"
      plot_time_filter_flag="--no-filter-time"
      plot_invert_flag="--no-invert-y"
    else
      adata_h5ad="${REAL_H5AD}"
      plot_annotation_col="annotation"
      plot_time_key="$(map_time_key "${tk}" "${TIME_KEY_MAP}")"
      plot_time_filter_flag=""
      plot_invert_flag="--no-invert-y"
    fi
    for pair in "${LR_PAIR_ARR[@]}"; do
      plot_cmd=(
        evaluation/mosta/code/mosta_lr_single_timepoint_nature_local.py
        --scores-pkl "${score_pkl}"
        --adata-h5ad "${adata_h5ad}"
        --lr-db "${LR_DB}"
        --lr-pair "${pair}"
        --time-key "${plot_time_key}"
        --annotation-col "${plot_annotation_col}"
        --x-space count
        --output-dir "${INTERP_PLOT_DIR}"
        "${plot_invert_flag}"
      )
      if [[ -n "${plot_time_filter_flag}" ]]; then
        plot_cmd+=("${plot_time_filter_flag}")
      fi
      run_py "${plot_cmd[@]}"
    done
  done
fi

echo "Done. Outputs under: ${BASE_OUT}"
