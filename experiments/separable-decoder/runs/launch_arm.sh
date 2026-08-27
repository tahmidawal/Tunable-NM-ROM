#!/bin/bash
# detached single-arm launcher: launch_arm.sh <arm> <mfactor>
H="$(cd "$(dirname "$0")" && pwd)"
RUN="$H/cd_${TAGPFX:-}${1}_m$((64 * $2))"
mkdir -p "$RUN"
setsid nohup bash -c "REC_W_OVR=$REC_W_OVR TAGPFX=$TAGPFX bash '$H/cd_minipilot.sh' '$1' '$2'; echo \$? > '$RUN/rc'" >/dev/null 2>&1 < /dev/null &
disown
echo "launched $1 m$((64 * $2))"
