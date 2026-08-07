#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script instead of sourcing it" >&2
  return 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/.." && pwd)
runtime_root=${NAVILA_NAV_RUNTIME_ROOT:-"$repository_root/.navila-navigation-runtime"}
environment_path=${NAVILA_NAV_ENV_PATH:-"$runtime_root/env"}
source_path=${NAVILA_NAV_SOURCE_PATH:-"$runtime_root/source/NaVILA"}
python_command=${NAVILA_NAV_PYTHON:-python3.10}
conda_command=${NAVILA_NAV_CONDA:-}

navila_repository=https://github.com/AnjieCheng/NaVILA.git
navila_commit=76b98f233dd0fff05dfcd69435eec6740febff9d
s2wrapper_repository=https://github.com/bfshi/scaling_on_scales
s2wrapper_commit=9c008a37540e761f53574b488979db6e49a64312
environment_marker="$environment_path/.embodied-runtime-navila-navigation"
marker_value="embodied-runtime-navila-navigation:$navila_commit"

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "NaVILA's pinned CUDA environment requires Linux x86_64" >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to install the pinned NaVILA source" >&2
  exit 2
fi
if [[ -e "$environment_path" ]]; then
  if [[ ! -d "$environment_path" || ! -f "$environment_marker" ]]; then
    echo "refusing to modify an existing unmarked environment: $environment_path" >&2
    exit 2
  fi
  if [[ $(<"$environment_marker") != "$marker_value" ]]; then
    echo "existing NaVILA environment marker does not match the pinned source" >&2
    exit 2
  fi
  if [[ ! -x "$environment_path/bin/python" ]]; then
    echo "marked NaVILA environment is incomplete: $environment_path" >&2
    exit 2
  fi
else
  mkdir -p -- "$(dirname -- "$environment_path")"
  if command -v "$python_command" >/dev/null 2>&1 && \
    [[ $("$python_command" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') == 3.10 ]] && \
    "$python_command" -m ensurepip --version >/dev/null 2>&1; then
    "$python_command" -m venv "$environment_path"
  else
    if [[ -z "$conda_command" ]]; then
      for candidate in \
        "$(command -v conda 2>/dev/null || true)" \
        "$HOME/.local/miniforge3/bin/conda" \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/miniconda3/bin/conda"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
          conda_command="$candidate"
          break
        fi
      done
    fi
    if [[ -z "$conda_command" || ! -x "$conda_command" ]]; then
      echo "Python 3.10 with ensurepip or a Conda executable is required" >&2
      exit 2
    fi
    "$conda_command" create --yes --prefix "$environment_path" python=3.10 pip
  fi
  printf '%s\n' "$marker_value" >"$environment_marker"
fi

temporary_source_root=""
cleanup() {
  if [[ -n "$temporary_source_root" && -d "$temporary_source_root" ]]; then
    rm -rf -- "$temporary_source_root"
  fi
}
trap cleanup EXIT

if [[ -e "$source_path" ]]; then
  if [[ ! -d "$source_path/.git" ]]; then
    echo "refusing to modify an existing non-NaVILA source path: $source_path" >&2
    exit 2
  fi
  source_origin=$(git -C "$source_path" remote get-url origin)
  source_revision=$(git -C "$source_path" rev-parse HEAD)
  if [[ "$source_origin" != "$navila_repository" || "$source_revision" != "$navila_commit" ]]; then
    echo "existing NaVILA source is not the pinned official revision: $source_path" >&2
    exit 2
  fi
  if [[ -n $(git -C "$source_path" status --porcelain --untracked-files=all) ]]; then
    echo "refusing to use a modified NaVILA source tree: $source_path" >&2
    exit 2
  fi
else
  source_parent=$(dirname -- "$source_path")
  mkdir -p -- "$source_parent"
  temporary_source_root=$(mktemp -d "$source_parent/.navila-source.XXXXXX")
  git clone --filter=blob:none --no-checkout "$navila_repository" "$temporary_source_root/NaVILA"
  git -C "$temporary_source_root/NaVILA" checkout --detach "$navila_commit"
  mv -- "$temporary_source_root/NaVILA" "$source_path"
  rmdir -- "$temporary_source_root"
  temporary_source_root=""
fi

environment_python="$environment_path/bin/python"
PYTHONNOUSERSITE=1 "$environment_python" -m pip install --upgrade "pip>=25,<27"
PYTHONNOUSERSITE=1 DS_BUILD_OPS=0 "$environment_python" -m pip install \
  torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
PYTHONNOUSERSITE=1 "$environment_python" -m pip install \
  transformers==4.37.2 \
  tokenizers==0.15.2 \
  sentencepiece==0.1.99 \
  accelerate==0.27.2 \
  numpy==1.26.0 \
  datasets==2.16.1 \
  requests \
  einops==0.6.1 \
  timm==0.9.12 \
  opencv-python-headless==4.8.0.74 \
  decord==0.6.0 \
  deepspeed==0.9.5 \
  loguru==0.7.3
if ! PYTHONNOUSERSITE=1 "$environment_python" - "$s2wrapper_commit" <<'PY'
import importlib
import importlib.metadata
import json
import sys

expected_commit = sys.argv[1]
try:
    distribution = importlib.metadata.distribution("s2wrapper")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    importlib.import_module("s2wrapper")
except (ImportError, importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if commit == expected_commit else 1)
PY
then
  PYTHONNOUSERSITE=1 "$environment_python" -m pip install \
    "s2wrapper @ git+$s2wrapper_repository@$s2wrapper_commit"
fi
PYTHONNOUSERSITE=1 "$environment_python" -m pip install -e "$repository_root"

overlay_path="$source_path/llava/train/transformers_replace"
if [[ ! -d "$overlay_path" ]]; then
  echo "pinned NaVILA source is missing its Transformers overlay" >&2
  exit 2
fi
site_packages=$(PYTHONNOUSERSITE=1 "$environment_python" -c \
  'import site; print(site.getsitepackages()[0])')
environment_resolved=$(PYTHONNOUSERSITE=1 "$environment_python" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$environment_path")
site_packages_resolved=$(PYTHONNOUSERSITE=1 "$environment_python" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$site_packages")
case "$site_packages_resolved/" in
  "$environment_resolved/"*) ;;
  *)
    echo "refusing to overlay Transformers outside the dedicated NaVILA environment" >&2
    exit 2
    ;;
esac
PYTHONNOUSERSITE=1 "$environment_python" \
  "$repository_root/scripts/prepare_navila_eager_overlay.py" \
  --source "$overlay_path" \
  --transformers-target "$site_packages_resolved/transformers"

PYTHONNOUSERSITE=1 PYTHONPATH="$repository_root/scripts" "$environment_python" - \
  "$overlay_path" "$site_packages_resolved/transformers" <<'PY'
import sys
from pathlib import Path

from prepare_navila_eager_overlay import eager_modeling_llama

source = Path(sys.argv[1])
target = Path(sys.argv[2])
llama = Path("models/llama/modeling_llama.py")
for source_file in source.rglob("*.py"):
    relative = source_file.relative_to(source)
    expected = source_file.read_text(encoding="utf-8")
    if relative == llama:
        expected = eager_modeling_llama(expected)
    assert (target / relative).read_text(encoding="utf-8") == expected, relative
PY

PYTHONNOUSERSITE=1 PYTHONPATH="$source_path" "$environment_python" - \
  "$environment_resolved" "$source_path" "$navila_commit" <<'PY'
import subprocess
import sys
from pathlib import Path

import flash_attn
import deepspeed
import deepspeed.comm as deepspeed_dist
import llava
import llava.model.builder
import torch
import transformers
from llava.train.sequence_parallel import globals as sequence_parallel_globals

environment = Path(sys.argv[1])
source = Path(sys.argv[2]).resolve()
expected_revision = sys.argv[3]
transformers_path = Path(transformers.__file__).resolve()
llava_path = Path(llava.__file__).resolve()
source_revision = subprocess.run(
    ["git", "-C", str(source), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

assert transformers_path.is_relative_to(environment)
assert llava_path.is_relative_to(source)
assert source_revision == expected_revision
assert torch.__version__.split("+", 1)[0] == "2.3.0"
assert transformers.__version__ == "4.37.2"
assert deepspeed.__version__ == "0.9.5"
assert flash_attn.__version__ == "0+navila-eager-disabled"
assert sequence_parallel_globals.dist is deepspeed_dist
assert not torch.distributed.is_initialized()
assert not deepspeed_dist.is_initialized()

print(f"NaVILA source ready: {source}@{source_revision}")
print(
    "isolated environment ready: "
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"deepspeed={deepspeed.__version__} attention=eager "
    f"flash_guard={flash_attn.__version__}"
)
print(f"cuda available: {torch.cuda.is_available()}")
PY
