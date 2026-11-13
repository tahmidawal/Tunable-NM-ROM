# Second rebuttal push, PROTOCOL.md A9/A9b: DEVELOPMENT role of the extra-case evaluation (Re 85, Re 115, long Re 95).
source tasks/stream_frozen.env
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
ROLE=dev
M=tasks/manifest_xdev.json
CASES="re85:85 re115:115 re95L:95"
LONG="re95L:$NS/r1-record-re95/results/record_re95_p4:$NS/rb-xl2/results/cont_re95_p4"
FLAGS=""
STAGES=${STAGES:-"merge rom ops"}
SBANK=$SARM/results/train/trained_bank_eval.npz
source tasks/rb_xe_common.sh
