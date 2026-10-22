#!/bin/bash
# imp-fronts operator steps after an arm's training and after its selection job (DESIGN B4, B5; Amendments 4, 9, 12, 19).
# Arm b (schedule S3, bank rom/inputs/banks/sb) or arm c (schedule S3C, bank rom/inputs/banks/sc, arch 'sine_even').
# (Steps below are written for b; for c read sc / e2sc / polc / SEL-c / schedule S3C.)
# CPU only. Never commits, stages, pushes or submits; prints what to commit next.
#
#   bash rom/after_train.sh <b|c> bank <pulled trcov output dir> <pulled TR-cov job log (.out)>
#       1. register_arm.py --seed-dir <dir> 1 b --job-log <log> -> rom/inputs/banks/sb/{ckpt.pkl, rotation_R512.npz,
#          BANK.json} (production run_record with seed 1, log ending ALL-DONE, no EXHAUSTED.json, SEED-COMPLETE,
#          OUTPUTS.sha256 verified; refuses if sb exists)
#       2. mkconfigs_sel.py b --r384             -> rom/configs/e2sb.json (R384 group identical to e2s1's)
#       then commit (see the printed list) and run cluster/rom/stage_rom.sh sel-b
#
#   bash rom/after_train.sh <b|c> post-sel <pulled SEL-b r2acc output dir>   e.g. runs/rom/SEL-b/output/r2acc_e2sb
#       3. select2d_r384.py <dir> b               -> rom/checks/select2d-b.json (an incomplete job persists an explicit
#                                                    'incomplete selection' outcome: selected = null, fib4181 start)
#       4. mkconfigs_pol.py b --schedule S3       -> rom/configs/polb.json
#       5. testonce.py register configs/polb.json -> rom/registrations/<key>.json (the job claims it, not this script)
#       then commit (see the printed list) and run cluster/rom/stage_rom.sh pol-b
set -euo pipefail
ROM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/tahmid/Dev/.venv/bin/python
ARM=${1:?usage: after_train.sh <b|c> bank|post-sel <dir> [job log]}; shift
case "$ARM" in b) SCHED=S3;; c) SCHED=S3C;; *) echo "arm must be b or c" >&2; exit 2;; esac
cmd=${1:?usage: after_train.sh <b|c> bank|post-sel <dir>}; src=${2:?missing dir}
src="$(cd "$src" && pwd)"
cd "$ROM"
case "$cmd" in
  bank)
    [ ! -e inputs/banks/s$ARM ] || { echo "inputs/banks/s$ARM exists: refusing (an arm bank is registered once)" >&2; exit 3; }
    joblog=${3:?bank needs the training job log (.out) as the third argument}
    "$PY" -B register_arm.py --seed-dir "$src" 1 "$ARM" --job-log "$joblog"
    "$PY" -B -c "import sys; sys.path.insert(0, '.'); import bankguard as BG; a = BG.verify({'dir': 'inputs/banks/s$ARM'}, BG.HERE, expect_label='$ARM')['arch']; print('arch', a); assert a == {'b': 'bc16poly', 'c': 'sine_even'}['$ARM'], a"
    "$PY" -B mkconfigs_sel.py "$ARM" --r384
    "$PY" -B -c "import json; c=json.load(open('configs/e2s$ARM.json')); assert [g['Rp'] for g in c['groups']]==[384] and c['attempt']=='e2s$ARM'"
    cat <<EOF
NEXT: commit, in one commit (ckpt.pkl is ~36 MB, below the 50 MB mirror limit):
  experiments/imp-fronts/rom/inputs/banks/s$ARM/BANK.json
  experiments/imp-fronts/rom/inputs/banks/s$ARM/ckpt.pkl
  experiments/imp-fronts/rom/inputs/banks/s$ARM/rotation_R512.npz
  experiments/imp-fronts/rom/configs/e2s$ARM.json
then: bash experiments/imp-fronts/cluster/rom/stage_rom.sh sel-$ARM   (admission: resources.py --admit select_$ARM 0.4)
EOF
    ;;
  post-sel)
    [ -e inputs/banks/s$ARM/BANK.json ] || { echo "arm bank s$ARM not registered" >&2; exit 3; }
    [ ! -e configs/pol$ARM.json ] || { echo "configs/pol$ARM.json exists: refusing" >&2; exit 3; }
    JAX_PLATFORMS=cpu "$PY" -B select2d_r384.py "$src" "$ARM"
    "$PY" -B mkconfigs_pol.py "$ARM" --schedule "$SCHED"
    out=$("$PY" -B testonce.py register "configs/pol$ARM.json")
    echo "$out"
    key=$(echo "$out" | awk '/^registered/ {print $2}')
    cat <<EOF
NEXT: commit, in one commit:
  experiments/imp-fronts/rom/checks/select2d-$ARM.json
  experiments/imp-fronts/rom/configs/pol$ARM.json
  experiments/imp-fronts/rom/registrations/$key.json
(do not change policy_sm.py, fpolicy.py, schedule.py or pol$ARM.json afterwards: the registration key hashes them)
then: bash experiments/imp-fronts/cluster/rom/stage_rom.sh pol-$ARM   (admission: resources.py --admit policy_$ARM 0.5)
EOF
    ;;
  *) echo "unknown command $cmd" >&2; exit 2;;
esac
