#!/bin/bash
# The exact pinned launches of this lane (DESIGN §7 / §A1). One job per call.
# Usage: launch.sh bank_a | bank_b | dev32 | dev64 | dev96
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SBATCH_EXTRA="--constraint=${GPU_CONSTRAINT:-a100-80G}"   # H100 override: GPU_CONSTRAINT=h100-80G
case "$1" in
  bank_a|bank_b) "$HERE/submit.sh" "$1" bank "configs/$1.json" gpu:1 160G 06:40:00 ;;
  dev96) "$HERE/submit.sh" dev96 rom configs/dev96.json gpu:1 160G 00:45:00 ;;
  dev64) "$HERE/submit.sh" dev64 rom configs/dev64.json gpu:1 120G 00:20:00 ;;
  dev32) "$HERE/submit.sh" dev32 rom configs/dev32.json gpu:1 96G 00:15:00 ;;
  *) echo "unknown job $1"; exit 1 ;;
esac
