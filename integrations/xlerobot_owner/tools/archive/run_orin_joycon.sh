#!/usr/bin/env bash
# Run on the Mac. Only the Joy-Con program is started; the robot server is reused.
set -euo pipefail
joycon_mode="${1:-monitor}"
case "$joycon_mode" in
  list|monitor|control) ;;
  *) echo 'Usage: bash scripts/run_orin_joycon.sh [list|monitor|control]' >&2; exit 2 ;;
esac
exec ssh -t -o BatchMode=yes -o ConnectTimeout=8 -o IdentitiesOnly=yes \
  -i /Users/user/.ssh/agx_orin_20260914_ed25519 root@192.168.123.4 \
  "cd /home/user/rlinf-orin-bench-20260914/quest-teleop-src && PYTHONNOUSERSITE=1 PYTHONPATH=. /home/user/rlinf-orin-bench-20260914/.sdk-venv/bin/python -m embodirun_xlerobot_owner.joycon_teleop --mode $joycon_mode --token-file robot-token"
