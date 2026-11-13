# Rebuttal push arm a1: r16, original architecture/features, long training
RANK=16
TRAIN_ARGS="--adam-steps 20000 --lbfgs-seconds 1800 --ft plain --ft-seconds 1200"
ARMS='[{"name":"pod_r8","kind":"pod","r":8},{"name":"pod_r16","kind":"pod","r":16},{"name":"bank_r16","kind":"bank","r":16,"key":"trained"},{"name":"bank_p8","kind":"bank","r":8,"key":"trained"}]'
source tasks/rb_common.sh
