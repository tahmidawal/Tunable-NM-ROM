# Second rebuttal push arm s1 (PROTOCOL.md A8): r32 stream-function bank, the a8b recipe plus --stream
RANK=32
EX=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/rb-export32/results/export
TRAIN_ARGS="--adam-steps 20000 --lbfgs-seconds 1800 --width 256 --depth 4 --fx 12,16,24,32,48 --stream --ft plain --ft-seconds 1200"
ARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
source tasks/rb_common.sh
