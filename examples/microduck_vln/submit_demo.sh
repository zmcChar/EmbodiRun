#!/usr/bin/env bash
# Submit a persistent Slurm job; its lifetime is independent of this SSH session.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
CONFIG="${MICRODUCK_CONFIG:-$HERE/config.env}"
if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
mkdir -p "$REPO/artifacts/microduck_vln/slurm"
printf -v COMMAND '%q ' bash "$HERE/start_demo.sh" "$@"
SLURM_OPTIONS=()
if [[ -n "${MICRODUCK_PARTITION:-}" ]]; then SLURM_OPTIONS+=(--partition="$MICRODUCK_PARTITION"); fi
exec sbatch --parsable "${SLURM_OPTIONS[@]}" --nodes=1 --ntasks=1 \
  --cpus-per-task="${MICRODUCK_CPUS:-4}" --gres=gpu:1 --time="${MICRODUCK_TIME:-01:00:00}" \
  --job-name=microduck-vln --export=ALL --chdir="$REPO" \
  --output="$REPO/artifacts/microduck_vln/slurm/%j.log" --wrap="$COMMAND"
