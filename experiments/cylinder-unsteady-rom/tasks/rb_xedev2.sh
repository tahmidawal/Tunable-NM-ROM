# Second rebuttal push, A9/A9b DEVELOPMENT role, operator block only (wall-time split of rb-xedev; nothing else differs).
source tasks/stream_frozen.env
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
ROLE=dev
M=tasks/manifest_xdev.json
CASES="re85:85 re115:115 re95L:95"
LONG="re95L:$NS/r1-record-re95/results/record_re95_p4:$NS/rb-xl2/results/cont_re95_p4"
FLAGS=""
STAGES="ops"
SBANK=$SARM/results/train/trained_bank_eval.npz
source tasks/rb_xe_common.sh
