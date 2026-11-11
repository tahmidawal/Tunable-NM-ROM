# Second rebuttal push, PROTOCOL.md A9/A9b: FINAL-CONFIRMATION role of the extra-case evaluation (Re 80, Re 120, long
# Re 105).  Submitted once, only after the development role completed and PROTOCOL.md records that nothing changed.
source tasks/stream_frozen.env
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
ROLE=conf
M=tasks/manifest_xconf.json
CASES="re80:80 re120:120 re105L:105"
LONG="re105L:$NS/r1-record-re105/results/record_re105_p4:$NS/rb-xl2/results/cont_re105_p4"
FLAGS="--allow-heldout"
STAGES=${STAGES:-"merge rom ops"}
SBANK=$SARM/results/train/trained_bank_eval.npz
source tasks/rb_xe_common.sh
