#!/bin/bash
# A transient user service survives closure of an interactive execution session.
set -euo pipefail
task_lane="$(cd "$(dirname "$0")/.." && pwd)"
test ! -e "$task_lane/results/B5-controller.pid"
systemd-run --user --unit=burgers-tree-b5-continuation \
  --property="WorkingDirectory=$task_lane" \
  --property="StandardOutput=append:$task_lane/results/B5-controller.log" \
  --property="StandardError=append:$task_lane/results/B5-controller.log" \
  --property=RuntimeMaxSec=172800 \
  /home/tahmid/Dev/.venv/bin/python -u "$task_lane/cluster/finish_b5.py"
systemctl --user show burgers-tree-b5-continuation.service \
  --property=ActiveState --property=SubState --property=MainPID
