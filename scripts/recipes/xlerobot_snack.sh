#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
recipe_script="$repo_root/recipes/xlerobot/snack_delivery/run.py"
default_config="$repo_root/recipes/xlerobot/snack_delivery/config.example.json"
recipe_python="${XLR_SNACK_PYTHON:-$repo_root/.venv-xlerobot-snack/bin/python}"
command_name="${1:-}"

if [[ "$command_name" == "setup" ]]; then
  shift
  exec "$repo_root/scripts/recipes/xlerobot_snack_setup.sh" "$@"
fi

case "$command_name" in
  plan|dry-run|hardware|services) ;;
  *)
    cat >&2 <<'USAGE'
usage:
  scripts/recipes/xlerobot_snack.sh setup [--minimal]
  scripts/recipes/xlerobot_snack.sh plan
  scripts/recipes/xlerobot_snack.sh dry-run
  scripts/recipes/xlerobot_snack.sh services
  scripts/recipes/xlerobot_snack.sh hardware

Set XLR_SNACK_CONFIG to a deployment config and XLR_SNACK_OUTPUT to choose
the output directory. Hardware mode assumes the owner and Control services
are already running; this script never starts a second serial owner.
USAGE
    exit 2
    ;;
esac

if [[ ! -x "$recipe_python" ]]; then
  recipe_python="${XLR_SNACK_PYTHON:-$(command -v python3)}"
fi
if [[ ! -x "$recipe_python" ]]; then
  echo "Python is unavailable; run '$0 setup' first" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$command_name" == "services" ]]; then
  shift
  exec "$recipe_python" "$repo_root/recipes/xlerobot/snack_delivery/services.py" \
    --deployment "${XLR_SNACK_DEPLOYMENT:-$repo_root/recipes/xlerobot/snack_delivery/deployment.example.yaml}" \
    --state-dir "${XLR_SNACK_SERVICES:-$repo_root/artifacts/recipes/xlerobot-services}" "$@"
fi
config_path="${XLR_SNACK_CONFIG:-$default_config}"
if [[ ! -f "$config_path" ]]; then
  echo "recipe config does not exist: $config_path" >&2
  exit 1
fi
run_stamp="$(date +%Y%m%d-%H%M%S)"
output_path="${XLR_SNACK_OUTPUT:-$repo_root/artifacts/recipes/xlerobot-snack-$run_stamp}"

args=("$recipe_script" --mode "$command_name" --config "$config_path" --output "$output_path")
if [[ "$command_name" == "dry-run" ]]; then
  args+=(--auto-confirm)
fi
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$recipe_python" "${args[@]}"
