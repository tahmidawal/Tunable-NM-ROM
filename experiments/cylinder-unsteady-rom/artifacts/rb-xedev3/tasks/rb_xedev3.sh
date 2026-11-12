# Second rebuttal push, A9/A9b DEVELOPMENT role, long Re 95 window at the two coarse steps only (wall-time split of rb-xedev).
source tasks/stream_frozen.env
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
ROLE=dev
M=tasks/manifest_xdev.json
CASES="re95L:95"
LONG="re95L:$NS/r1-record-re95/results/record_re95_p4:$NS/rb-xl2/results/cont_re95_p4"
FLAGS=""
STAGES="rom"
DTS="0.0025 0.00125"
SBANK=$SARM/results/train/trained_bank_eval.npz
source tasks/rb_xe_common.sh
