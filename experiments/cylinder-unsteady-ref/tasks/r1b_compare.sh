# R1b comparisons (DESIGN-G5 4.4, G5-A5): run inside a ref-lane allocation (NGSolve on PYTHONPATH).
set -uo pipefail
A=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r1b-study-a/results
B=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r1b-study-b/results
mkdir -p results/compare
c() { "$PY" compare.py --a "$1" --b "$2" --out "results/compare/$3.json" || echo "comparison $3 invalid/failed"; }
c $A/time_p4_dt800  $A/time_p4_dt1600 compare_800_1600
c $A/time_p4_dt1600 $A/time_p4_dt3200 compare_1600_3200
c $A/time_p4_dt3200 $B/time_p4_dt6400 compare_3200_6400
c $A/time_p4_dt800  $B/startup16_p4_dt800  startup_800
c $A/time_p4_dt1600 $B/startup16_p4_dt1600 startup_1600
c $A/time_p4_dt3200 $A/startup16_p4_dt3200 startup_3200
c $A/time_p4_dt1600 $B/roundoff_p4_dt1600_threads8 roundoff_1600
"$PY" r1_gates.py --dir results/compare --out results/compare/R1BC-GATES.json
