#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script instead of sourcing it" >&2
  return 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/.." && pwd)
runtime_root=${GO2_NAV_RUNTIME_ROOT:-"$repository_root/.go2-nav-runtime"}
environment_path=${GO2_NAV_ENV_PATH:-"$runtime_root/env"}
python_command=${GO2_NAV_PYTHON:-python3.10}

mkdir -p -- "$runtime_root"
if [[ ! -x "$environment_path/bin/python" ]]; then
  "$python_command" -m venv "$environment_path"
fi

PYTHONNOUSERSITE=1 "$environment_path/bin/python" -m pip install --upgrade "pip>=25,<27"
PYTHONNOUSERSITE=1 "$environment_path/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
PYTHONNOUSERSITE=1 "$environment_path/bin/python" -m pip install -e "$repository_root[go2-nav]"

PYTHONNOUSERSITE=1 "$environment_path/bin/python" - <<'PY'
import torch
import transformers

print(f"python environment ready: torch={torch.__version__} transformers={transformers.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
PY
