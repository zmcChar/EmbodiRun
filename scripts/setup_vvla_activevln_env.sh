#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script instead of sourcing it" >&2
  return 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/.." && pwd)
vvla_source="$repository_root/third_party/vvla"
runtime_root=${VVLA_ACTIVEVLN_RUNTIME_ROOT:-"$repository_root/.vvla-activevln-runtime"}
environment_path=${VVLA_ACTIVEVLN_ENV_PATH:-"$runtime_root/env"}
python_command=${VVLA_ACTIVEVLN_PYTHON:-python3.10}
vvla_commit=6f65961c0222bbbd86ca3b09b93f13cee3ac70c8
environment_marker="$environment_path/.embodied-runtime-vvla-activevln"
marker_value="embodied-runtime-vvla-activevln:$vvla_commit:python3.10:torch2.7.1-cu128"

if [[ ! -f "$vvla_source/vvla/__init__.py" ]]; then
  echo "VVLA submodule is missing; run: git submodule update --init third_party/vvla" >&2
  exit 2
fi
actual_commit=$(git -C "$vvla_source" rev-parse HEAD)
if [[ "$actual_commit" != "$vvla_commit" ]]; then
  echo "VVLA submodule is not at the pinned revision: $actual_commit" >&2
  exit 2
fi
if [[ -n $(git -C "$vvla_source" status --porcelain --untracked-files=all) ]]; then
  echo "refusing to use a modified VVLA submodule" >&2
  exit 2
fi

if [[ -e "$environment_path" ]]; then
  if [[ ! -x "$environment_path/bin/python" || ! -f "$environment_marker" ]]; then
    echo "refusing to modify an existing unmarked environment: $environment_path" >&2
    exit 2
  fi
  if [[ $(<"$environment_marker") != "$marker_value" ]]; then
    echo "existing VVLA environment marker does not match this setup" >&2
    exit 2
  fi
else
  mkdir -p -- "$runtime_root"
  if command -v "$python_command" >/dev/null 2>&1; then
    "$python_command" -m venv "$environment_path"
  elif command -v uv >/dev/null 2>&1; then
    uv venv --seed --python 3.10 "$environment_path"
  else
    echo "Python 3.10 or uv is required" >&2
    exit 2
  fi
  printf '%s\n' "$marker_value" >"$environment_marker"
fi

environment_python="$environment_path/bin/python"
if ! PYTHONNOUSERSITE=1 "$environment_python" -m pip --version >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$environment_python" pip
  else
    "$environment_python" -m ensurepip --upgrade
  fi
fi
PYTHONNOUSERSITE=1 "$environment_python" -m pip install --upgrade "pip>=25,<27"
PYTHONNOUSERSITE=1 "$environment_python" -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
PYTHONNOUSERSITE=1 "$environment_python" -m pip install -e "$vvla_source[activevln]"
PYTHONNOUSERSITE=1 "$environment_python" -m pip install -e "$repository_root[vvla_activevln]"

PYTHONNOUSERSITE=1 "$environment_python" - "$vvla_source" "$vvla_commit" <<'PY'
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import vvla

source = Path(sys.argv[1]).resolve()
expected = sys.argv[2]
actual = subprocess.run(
    ["git", "-C", str(source), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

assert actual == expected
assert Path(vvla.__file__).resolve().is_relative_to(source)
assert transformers.__version__ == "4.51.3"
assert torch.__version__.split("+", 1)[0] == "2.7.1"
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    assert torch.cuda.is_bf16_supported()
    print(f"CUDA ready: {torch.cuda.get_device_name(0)} capability={capability}")
else:
    print("environment ready; CUDA is not visible")
print(f"VVLA ready: {source}@{actual}")
PY
