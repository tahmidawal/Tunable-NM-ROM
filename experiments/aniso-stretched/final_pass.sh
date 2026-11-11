#!/bin/bash
# One-time held-out pass (cohort "test") for the dev-selected configurations, run alone on the machine.
# 3D comparators on the GPU and on one CPU core, 3D ROM arms (online pinned to one core), then 2D comparators.
cd "$(dirname "$0")"; source /etc/profile.d/jax-mem.sh
export JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=_pydeps; PY=/home/tahmid/Dev/.venv/bin/python
DTYPES=float32 jaxrun $PY -u sweep3d.py time test runs/t3/test_gpu.jsonl 2:12,4:6,4:12 fd,jacobi 1e-2,1e-3 > runs/t3/test_gpu.log 2>&1
JAX_PLATFORMS=cpu OPFORM=dense BACKEND_LABEL=cpu_jax_1core_dense DTYPES=float32 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 6 \
  $PY -u sweep3d.py time test runs/t3/test_cpu1_dense.jsonl 4:4,4:6,4:12 fd 1e-2,1e-3 > runs/t3/test_cpu1_dense.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RANKS=$POD_RANKS jaxrun taskset -c 6 $PY -u rom3d.py pod 4 12 2048 test runs/rom3/pod_p4N12_2048_test.json > runs/rom3/pod_p4N12_2048_test.log 2>&1
RANKS=$B128_RANKS jaxrun taskset -c 6 $PY -u rom3d.py bank 4 12 2048 test runs/rom3/bankgal_R128_test.json runs/rom3/bank_R128.npz > runs/rom3/bankgal_R128_test.log 2>&1
RANKS=$B256_RANKS jaxrun taskset -c 6 $PY -u rom3d.py bank 4 12 2048 test runs/rom3/bankgal_R256_test.json runs/rom3/bank_R256.npz > runs/rom3/bankgal_R256_test.log 2>&1
C=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
taskset -c 6 $PY time_solvers.py A test $C 48:0:0,96:0:0 runs/time/test_A_quiet.json direct,frozen_lu,amg,cg,jacobi,fd > runs/time/test_A_quiet.log 2>&1
taskset -c 6 $PY hq_sweep.py A test runs/time/hq_test_A_quiet.json 3:4:0:0,6:3:0:0,6:6:0:0,4:16:0:1.5 > runs/time/hq_test_A_quiet.log 2>&1
taskset -c 6 $PY time_solvers.py B test $C 48:0:1.5,64:0:1.5 runs/time/test_B_quiet.json direct,frozen_lu,amg,cg,jacobi > runs/time/test_B_quiet.log 2>&1
taskset -c 6 $PY hq_sweep.py B test runs/time/hq_test_B_quiet.json 4:4:1.5:1.5,4:6:1.5:1.5,6:6:0:1.5,4:12:1.5:1.5 > runs/time/hq_test_B_quiet.log 2>&1
# dev curves re-timed pinned to one core (errors identical to the dev runs; used for the figure only)
unset RANKS
jaxrun taskset -c 6 $PY -u rom3d.py pod 4 12 2048 dev runs/rom3/pod_p4N12_2048_dev_quiet.json > runs/rom3/pod_p4N12_2048_dev_quiet.log 2>&1
jaxrun taskset -c 6 $PY -u rom3d.py bank 4 12 2048 dev runs/rom3/bankgal_R256_dev_quiet.json runs/rom3/bank_R256.npz > runs/rom3/bankgal_R256_dev_quiet.log 2>&1
echo final pass done > runs/final_pass.done
