# R1c (DESIGN-G5 4.4, G5-A3/A5): P3/P2 at dt0=1/3200 from the original P3 S*, vs the R1b P4/P3 run.
set -uo pipefail
"$PY" transient.py --config configs/space_p3/space_p3_dt3200.json --out results/space_p3_dt3200 || exit 1
mkdir -p results/compare
cp /cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r1b-compare01/results/compare/*.json results/compare/
"$PY" compare.py --a results/space_p3_dt3200 --b /cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r1b-study-a2/results/time_p4_dt3200 --out results/compare/space_3200.json
"$PY" r1_gates.py --dir results/compare --out results/compare/R1BC-GATES.json
