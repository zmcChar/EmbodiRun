#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PROMPT=""
BACKEND="${GO2_NAV_BACKEND:-streamvln}"
ROBOT_HOST="${GO2_ROBOT_HOST:-}"
SSH_PORT="${GO2_SSH_PORT:-22}"
SSH_USER="${GO2_SSH_USER:-}"
SSH_CONNECT_TIMEOUT_S="${GO2_SSH_CONNECT_TIMEOUT_S:-5}"
DEPLOY_ATTEMPTS="${GO2_DEPLOY_ATTEMPTS:-3}"
PASSWORD_ENV_NAME="${GO2_PASSWORD_ENV_NAME:-GO2_SSH_PASSWORD}"
REMOTE_ROOT="${GO2_REMOTE_ROOT:-/home/unitree/robotics/rlinf-go2-agent}"
ROBOT_PYTHON="${GO2_ROBOT_PYTHON:-/home/unitree/miniforge3/envs/unitree/bin/python}"
DDS_INTERFACE="${GO2_DDS_INTERFACE:-eth0}"
STATE_TOPIC="${GO2_STATE_TOPIC:-rt/lf/sportmodestate}"
CYCLONEDDS_LIB_DIR="${GO2_CYCLONEDDS_LIB_DIR:-/home/unitree/cyclonedds/install/lib}"

SECRETS_FILE="${GO2_NAV_SECRETS_FILE:-$RUNTIME_ROOT/run/go2-test.env}"
HOST_PYTHON_OVERRIDE="${GO2_NAV_PYTHON:-}"
DEPLOY_PYTHON="${GO2_DEPLOY_PYTHON:-python3}"
STREAMVLN_ROOT="${STREAMVLN_ROOT:-$RUNTIME_ROOT/src/StreamVLN}"
STREAMVLN_MODEL_PATH="${STREAMVLN_MODEL_PATH:-$RUNTIME_ROOT/checkpoints/streamvln-real-world}"
NAVILA_RUNTIME_ROOT="${NAVILA_NAV_RUNTIME_ROOT:-$SCRIPT_DIR/.navila-navigation-runtime}"
NAVILA_ROOT="${NAVILA_ROOT:-${NAVILA_NAV_SOURCE_PATH:-$NAVILA_RUNTIME_ROOT/source/NaVILA}}"
NAVILA_MODEL_PATH="${NAVILA_MODEL_PATH:-$RUNTIME_ROOT/checkpoints/navila-llama3-8b-8f}"
MODEL_PATH_OVERRIDE=""
CUDA_DEVICE_INDEX="${GO2_NAV_CUDA_DEVICE_INDEX:-1}"
CUDA_MEMORY_FRACTION="${GO2_NAV_CUDA_MEMORY_FRACTION:-0.44}"

CAMERA_PORT="${GO2_CAMERA_PORT:-8765}"
CONTROL_PORT="${GO2_CONTROL_PORT:-8080}"
CAMERA_SERIAL="${GO2_CAMERA_SERIAL:-}"
DEPTH_SCALE="${GO2_DEPTH_SCALE:-}"
MAX_RUNTIME_S="${GO2_NAV_MAX_RUNTIME_S:-120}"
CONTROL_TIMEOUT_S="${GO2_NAV_CONTROL_TIMEOUT_S:-3.0}"
MAX_VX="${GO2_NAV_MAX_VX:-0.25}"
MAX_VY="${GO2_NAV_MAX_VY:-0.25}"
MAX_YAW_RATE="${GO2_NAV_MAX_YAW_RATE:-0.50}"
EXECUTE=1

usage() {
  cat <<'EOF'
Usage:
  bash nav.sh --backend streamvln --prompt "Find the tripod and stop in front of it." \
    --robot-host HOST --ssh-user USER \
    --camera-serial SERIAL --depth-scale METERS_PER_UNIT

Options:
  --prompt TEXT               Navigation instruction (required)
  --backend NAME              streamvln (default) or navila
  --robot-host HOST           Go2 IP/hostname
  --ssh-port PORT             Go2 SSH port
  --ssh-user USER             Go2 SSH username
  --deploy-attempts COUNT     SSH deployment attempts
  --password-env NAME         Environment variable containing the SSH password
  --secrets-file PATH         File containing GO2_API_TOKEN/GO2_CAMERA_TOKEN
  --remote-root PATH          Dog-side installation root
  --camera-serial SERIAL      RealSense device serial (required)
  --depth-scale SCALE         Verified meters per raw depth unit (required)
  --streamvln-root PATH       Official StreamVLN repository
  --navila-root PATH          Official NaVILA repository
  --model-path PATH           Checkpoint for the selected backend
  --cuda-device-index INDEX   Physical GPU exposed to the process
  --max-runtime-s SECONDS     Navigation timeout
  --control-timeout-s SEC     Timeout for one control HTTP request
  --max-vx MPS               Forward/backward speed bound
  --max-vy MPS               Lateral speed bound
  --max-yaw-rate RAD_S        Yaw-rate bound
  --dry-run                   Run camera and model inference without motion
  -h, --help                  Show this help

Environment variables with matching GO2_* names may also override defaults.
EOF
}

die() {
  printf 'nav.sh: %s\n' "$*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local value="${2-}"
  [[ -n "$value" ]] || die "$option requires a value"
}

while (($#)); do
  case "$1" in
    --prompt)
      require_value "$1" "${2-}"
      PROMPT="$2"
      shift 2
      ;;
    --backend)
      require_value "$1" "${2-}"
      BACKEND="$2"
      shift 2
      ;;
    --robot-host)
      require_value "$1" "${2-}"
      ROBOT_HOST="$2"
      shift 2
      ;;
    --ssh-port)
      require_value "$1" "${2-}"
      SSH_PORT="$2"
      shift 2
      ;;
    --ssh-user)
      require_value "$1" "${2-}"
      SSH_USER="$2"
      shift 2
      ;;
    --deploy-attempts)
      require_value "$1" "${2-}"
      DEPLOY_ATTEMPTS="$2"
      shift 2
      ;;
    --password-env)
      require_value "$1" "${2-}"
      PASSWORD_ENV_NAME="$2"
      shift 2
      ;;
    --secrets-file)
      require_value "$1" "${2-}"
      SECRETS_FILE="$2"
      shift 2
      ;;
    --remote-root)
      require_value "$1" "${2-}"
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --camera-serial)
      require_value "$1" "${2-}"
      CAMERA_SERIAL="$2"
      shift 2
      ;;
    --depth-scale)
      require_value "$1" "${2-}"
      DEPTH_SCALE="$2"
      shift 2
      ;;
    --streamvln-root)
      require_value "$1" "${2-}"
      STREAMVLN_ROOT="$2"
      shift 2
      ;;
    --navila-root)
      require_value "$1" "${2-}"
      NAVILA_ROOT="$2"
      shift 2
      ;;
    --model-path)
      require_value "$1" "${2-}"
      MODEL_PATH_OVERRIDE="$2"
      shift 2
      ;;
    --cuda-device-index)
      require_value "$1" "${2-}"
      CUDA_DEVICE_INDEX="$2"
      shift 2
      ;;
    --max-runtime-s)
      require_value "$1" "${2-}"
      MAX_RUNTIME_S="$2"
      shift 2
      ;;
    --control-timeout-s)
      require_value "$1" "${2-}"
      CONTROL_TIMEOUT_S="$2"
      shift 2
      ;;
    --max-vx)
      require_value "$1" "${2-}"
      MAX_VX="$2"
      shift 2
      ;;
    --max-vy)
      require_value "$1" "${2-}"
      MAX_VY="$2"
      shift 2
      ;;
    --max-yaw-rate)
      require_value "$1" "${2-}"
      MAX_YAW_RATE="$2"
      shift 2
      ;;
    --dry-run)
      EXECUTE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "$BACKEND" in
  streamvln)
    MODEL_ROOT="$STREAMVLN_ROOT"
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-$STREAMVLN_MODEL_PATH}"
    HOST_PYTHON="${HOST_PYTHON_OVERRIDE:-$RUNTIME_ROOT/env/bin/python}"
    MODEL_ROOT_OPTION="--streamvln-root"
    ;;
  navila)
    MODEL_ROOT="$NAVILA_ROOT"
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-$NAVILA_MODEL_PATH}"
    HOST_PYTHON="${HOST_PYTHON_OVERRIDE:-${NAVILA_NAV_ENV_PATH:-$NAVILA_RUNTIME_ROOT/env}/bin/python}"
    MODEL_ROOT_OPTION="--navila-root"
    ;;
  *)
    die "--backend must be streamvln or navila"
    ;;
esac

[[ -n "${PROMPT//[[:space:]]/}" ]] || die '--prompt is required'
[[ -n "${ROBOT_HOST//[[:space:]]/}" ]] || \
  die '--robot-host is required (or set GO2_ROBOT_HOST)'
[[ -n "${SSH_USER//[[:space:]]/}" ]] || \
  die '--ssh-user is required (or set GO2_SSH_USER)'
[[ -n "${CAMERA_SERIAL//[[:space:]]/}" ]] || \
  die '--camera-serial is required (or set GO2_CAMERA_SERIAL)'
if ! [[ "$DEPTH_SCALE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  die '--depth-scale must be a positive finite decimal'
fi
awk -v value="$DEPTH_SCALE" 'BEGIN { exit !(value > 0 && value < 1) }' || \
  die '--depth-scale must be finite, greater than zero, and less than one'
[[ "$PASSWORD_ENV_NAME" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die 'invalid --password-env name'
[[ "$DEPLOY_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || die '--deploy-attempts must be positive'
[[ -r "$SECRETS_FILE" ]] || die "cannot read secrets file: $SECRETS_FILE"
[[ -x "$HOST_PYTHON" ]] || die "navigation Python is not executable: $HOST_PYTHON"
[[ -d "$MODEL_ROOT" ]] || die "$BACKEND repository does not exist: $MODEL_ROOT"
[[ -d "$MODEL_PATH" ]] || die "$BACKEND checkpoint does not exist: $MODEL_PATH"
command -v "$DEPLOY_PYTHON" >/dev/null || die "deployment Python was not found: $DEPLOY_PYTHON"
command -v curl >/dev/null || die 'curl is required'

set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

[[ -n "${GO2_API_TOKEN:-}" ]] || die 'GO2_API_TOKEN is missing from the secrets file'
[[ -n "${GO2_CAMERA_TOKEN:-}" ]] || die 'GO2_CAMERA_TOKEN is missing from the secrets file'

if [[ -z "${!PASSWORD_ENV_NAME:-}" ]]; then
  [[ -t 0 ]] || die "$PASSWORD_ENV_NAME is unset and no interactive terminal is available"
  read -rsp "Go2 SSH password for ${SSH_USER}@${ROBOT_HOST}: " PASSWORD_VALUE
  printf '\n'
  printf -v "$PASSWORD_ENV_NAME" '%s' "$PASSWORD_VALUE"
  export "$PASSWORD_ENV_NAME"
  unset PASSWORD_VALUE
fi

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DEPLOY=(
  "$DEPLOY_PYTHON" "$SCRIPT_DIR/examples/go2_agent_deploy.py"
  --host "$ROBOT_HOST"
  --ssh-port "$SSH_PORT"
  --connect-timeout "$SSH_CONNECT_TIMEOUT_S"
  --username "$SSH_USER"
  --password-env "$PASSWORD_ENV_NAME"
  --remote-root "$REMOTE_ROOT"
)

run_deploy() {
  local attempt
  for ((attempt = 1; attempt <= DEPLOY_ATTEMPTS; attempt++)); do
    if "${DEPLOY[@]}" "$@"; then
      return 0
    fi
    if ((attempt < DEPLOY_ATTEMPTS)); then
      printf 'Go2 SSH attempt %d/%d failed; retrying...\n' \
        "$attempt" "$DEPLOY_ATTEMPTS" >&2
      sleep 1
    fi
  done
  return 1
}

camera_ready() {
  curl -fsS --max-time 3 \
    -H "Authorization: Bearer $GO2_CAMERA_TOKEN" \
    "http://${ROBOT_HOST}:${CAMERA_PORT}/health" >/dev/null 2>&1
}

control_ready() {
  local expected_ready="$1"
  local payload
  payload="$(curl -fsS --max-time 3 \
    -H "Authorization: Bearer $GO2_API_TOKEN" \
    "http://${ROBOT_HOST}:${CONTROL_PORT}/health" 2>/dev/null)" || return 1
  if [[ "$expected_ready" == "true" ]]; then
    [[ "$payload" == *'"operator_motion_ready":true'* ]]
  else
    return 0
  fi
}

wait_until() {
  local description="$1"
  local check_function="$2"
  local check_argument="${3-}"
  local attempt
  for attempt in {1..30}; do
    if "$check_function" "$check_argument"; then
      return 0
    fi
    sleep 0.25
  done
  die "$description did not become ready"
}

printf 'Stopping installed Go2 services before refreshing agent code...\n'
run_deploy stop --service all >/dev/null

printf 'Installing the current repository-owned Go2 agent...\n'
run_deploy install \
  --package-root "$SCRIPT_DIR/src/embodied_runtime" >/dev/null

printf 'Starting Go2 camera service...\n'
run_deploy start \
  --service camera \
  --python "$ROBOT_PYTHON" \
  --camera-backend realsense \
  --realsense-serial "$CAMERA_SERIAL" \
  --depth-scale "$DEPTH_SCALE" \
  --camera-port "$CAMERA_PORT" \
  --camera-width 640 \
  --camera-height 360 \
  --camera-fps 15 \
  --jpeg-fps 5 \
  --camera-token-env GO2_CAMERA_TOKEN >/dev/null
wait_until 'camera service' camera_ready

if ((EXECUTE)); then
  printf 'Re-arming Go2 control service...\n'
  run_deploy start \
    --service control \
    --python "$ROBOT_PYTHON" \
    --control-mode live \
    --interface "$DDS_INTERFACE" \
    --state-topic "$STATE_TOPIC" \
    --cyclonedds-lib-dir "$CYCLONEDDS_LIB_DIR" \
    --control-port "$CONTROL_PORT" \
    --api-token-env GO2_API_TOKEN \
    --operator-ready >/dev/null
  wait_until 'armed control service' control_ready true
else
  printf 'Starting read-only Go2 control service...\n'
  run_deploy start \
    --service control \
    --python "$ROBOT_PYTHON" \
    --control-mode live \
    --interface "$DDS_INTERFACE" \
    --state-topic "$STATE_TOPIC" \
    --cyclonedds-lib-dir "$CYCLONEDDS_LIB_DIR" \
    --control-port "$CONTROL_PORT" \
    --api-token-env GO2_API_TOKEN >/dev/null
  wait_until 'control service' control_ready false
fi

printf 'Starting %s navigation: %s\n' "$BACKEND" "$PROMPT"

NAVIGATION=(
  "$HOST_PYTHON" -m embodied_runtime.apps.navigate
  --config "$SCRIPT_DIR/configs/go2_navigation.toml"
  --backend "$BACKEND"
  --session-mode reactive
  "$MODEL_ROOT_OPTION" "$MODEL_ROOT"
  --model-path "$MODEL_PATH"
  --device cuda:0
  --cuda-memory-fraction "$CUDA_MEMORY_FRACTION"
  --robot-host "$ROBOT_HOST"
  --camera-port "$CAMERA_PORT"
  --control-port "$CONTROL_PORT"
  --control-timeout-s "$CONTROL_TIMEOUT_S"
  --instruction "$PROMPT"
  --max-runtime-s "$MAX_RUNTIME_S"
  --max-vx "$MAX_VX"
  --max-vy "$MAX_VY"
  --max-yaw-rate "$MAX_YAW_RATE"
)

if ((EXECUTE)); then
  NAVIGATION+=(--execute)
fi

PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_INDEX" \
exec "${NAVIGATION[@]}"
