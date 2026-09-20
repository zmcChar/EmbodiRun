#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
CONFIG="${MICRODUCK_CONFIG:-$HERE/config.env}"
if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
PYTHON="${MICRODUCK_PYTHON:-python3}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${MICRODUCK_CPUS:-4}" MKL_NUM_THREADS="${MICRODUCK_CPUS:-4}" OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO/integrations/microduck_vln/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${MICRODUCK_RUNTIME_OVERLAY:-}" ]]; then
  [[ -d "$MICRODUCK_RUNTIME_OVERLAY" ]] || { echo "Missing overlay: $MICRODUCK_RUNTIME_OVERLAY" >&2; exit 2; }
  export PYTHONPATH="$MICRODUCK_RUNTIME_OVERLAY:$PYTHONPATH"
fi
for ARG in "$@"; do
  case "$ARG" in
    --help|-h|--write-manifest) exec "$PYTHON" "$HERE/run_demo.py" "$@" ;;
  esac
done
SLURM_OPTIONS=()
if [[ -n "${MICRODUCK_PARTITION:-}" ]]; then SLURM_OPTIONS+=(--partition="$MICRODUCK_PARTITION"); fi
case "${MICRODUCK_SLURM:-off}" in
  auto)
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      command -v srun >/dev/null || { echo 'srun unavailable; set MICRODUCK_SLURM=off on a GPU host.' >&2; exit 2; }
      exec srun "${SLURM_OPTIONS[@]}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${MICRODUCK_CPUS:-4}" --gres=gpu:1 --time="${MICRODUCK_TIME:-01:00:00}" \
        --job-name=microduck-vln --export=ALL bash "$HERE/start_demo.sh" "$@"
    fi ;;
  off) ;;
  *) echo 'MICRODUCK_SLURM must be auto or off' >&2; exit 2 ;;
esac
exec "$PYTHON" "$HERE/run_demo.py" "$@"
