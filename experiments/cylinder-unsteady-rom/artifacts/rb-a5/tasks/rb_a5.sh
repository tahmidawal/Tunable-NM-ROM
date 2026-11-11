# Rebuttal push arm a5: r24, stream-function (divergence-free) construction
RANK=24
TRAIN_ARGS="--adam-steps 20000 --lbfgs-seconds 1800 --width 256 --depth 4 --fx 12,16,24,32,48 --stream --ft plain --ft-seconds 1200"
ARMS='[{"name":"pod_r8","kind":"pod","r":8},{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"bank_r24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"},{"name":"bank_p8","kind":"bank","r":8,"key":"trained"}]'
source tasks/rb_common.sh
