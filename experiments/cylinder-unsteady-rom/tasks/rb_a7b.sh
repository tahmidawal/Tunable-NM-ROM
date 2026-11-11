# Rebuttal push arm a7 (resubmitted as a7b after the queue reshuffle): r24, as a3 then fine-tune with weight 4 on x in [0.12,1.3]
RANK=24
TRAIN_ARGS="--adam-steps 20000 --lbfgs-seconds 1800 --width 256 --depth 4 --fx 12,16,24,32,48 --ft plain --ft-seconds 1200 --ft-box-weight 4"
ARMS='[{"name":"pod_r8","kind":"pod","r":8},{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"bank_r24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"},{"name":"bank_p8","kind":"bank","r":8,"key":"trained"}]'
source tasks/rb_common.sh
