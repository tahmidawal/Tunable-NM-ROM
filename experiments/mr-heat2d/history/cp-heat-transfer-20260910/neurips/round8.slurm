#!/bin/bash -l
#SBATCH --job-name=hopt8
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=/cluster/home/tawal01/nmrom/slurm-logs/hopt8-%j.out
#SBATCH --error=/cluster/home/tawal01/nmrom/slurm-logs/hopt8-%j.err
#
# Round 8, two independent phases.
#
# Phase A -- Heat-2D reduced-Gram.  The 2D solver builds a full (N^2 x k+1)
# Jacobian every Gauss-Newton iteration.  Because the CP decoder is affine in a
# rank-256 feature vector (2D; 3D uses 512) and the heat operator is linear with affine kappa
# dependence, J^T J / J^T r / ||r||^2 can be formed EXACTLY from kappa-free
# rank x rank Grams, so the online solve never touches a grid-sized array.
# Verified exact off-cluster: J^T J agrees to 9.3e-14 with identical
# Gauss-Newton counts on 12 random configurations (test_red2d.py), and the
# closed-form MLP Jacobian agrees with jax.jacfwd to 7.4e-8 relative.
# The reduced Grams MUST be f64 (X64=1): in f32 J^T J is off by 7% and J^T r by
# 51%, and GN counts flip on 4 of 12 configs.  Same lesson as GRAM64.
# X64 is a global flag, so h2opt pins every dtype on the FOM path and on the
# dense ROM path to f32; the X64=0 control below is what proves that worked.
#
# Phase B -- Heat-3D counted loop + CUDA graph capture, together.  Round 5 got
# ~8% from graph capture alone and noted the reason it was not more: a
# data-dependent while_loop cannot be captured.  The counted (fori_loop) variants
# have a static trip count, so the two should compose.  h3d_n64_fast has cap 2
# and h3d_n32 has cap 3, so for those two the counted loop is free.
# The graph flag is set for the FOM as well -- rule 1 -- and each setting runs in
# its own process on the same node.
#
# The FOM is untouched in both phases: same operator, CG tol 1e-6, 1000-iter
# cap, 50 steps, jitted with kappa as a runtime argument.

set -eo pipefail   # never -u: the sourced module script dies under it
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load modtree/deprecated
module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
export PYTHONPATH="/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:$PYTHONPATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
WORK=/cluster/tufts/paralab/tawal01/heatopt; cd "$WORK"; mkdir -p results

echo "=== hopt8 on $(hostname) job=$SLURM_JOB_ID at $(date) ==="
echo "co-resident on $SLURMD_NODENAME: $(squeue -h -w $SLURMD_NODENAME -t R -o '%i' | tr '\n' ' ')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
BACKEND=$(python3 -c "import jax; print(jax.default_backend())"); echo "jax_backend=$BACKEND"
[ "$BACKEND" = "gpu" ] || { echo "FATAL: jax_backend=$BACKEND"; exit 42; }

# f64 throughput of THIS device, relative to f32.  The local GB10 runs f64 at
# roughly 1/3 of f32 on this workload and that is what made the reduced path
# look slow there; A100/H100/H200 are nominally 1:2.  Recorded, not assumed.
python3 - <<'PY' || true
import time, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
out = []
for dt in (jnp.float32, jnp.float64):
    a = jnp.ones((2048, 2048), dt); b = jnp.ones((2048, 2048), dt)
    f = jax.jit(lambda x, y: x @ y)
    for _ in range(3): jax.block_until_ready(f(a, b))
    t0 = time.perf_counter()
    for _ in range(20): jax.block_until_ready(f(a, b))
    out.append((time.perf_counter() - t0) / 20)
print(f'device f32 {out[0]*1e3:.3f} ms  f64 {out[1]*1e3:.3f} ms  '
      f'f64/f32 = {out[1]/out[0]:.2f}x  (2048^3 matmul)')
PY

RC=0

########################### Phase A -- Heat-2D reduced ###########################
# X64=0 control FIRST, on every cell: these are the numbers round 6 measured, and
# they must not move when X64=1 is switched on in the runs that follow.
for CELL in h2d_n64_acc h2d_n64_fast h2d_n128_acc h2d_n128_fast; do
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  echo ""; echo "########## r8A $CELL  X64=0 CONTROL ##########"
  X64=0 python3 -u h2opt.py --cell "$CELL" \
    --variants 'base;enc_jit;v4w' --repeats 9 --warmup 3 \
    --out "$WORK/results/r8A_x64off_${CELL}_${STAMP}_${SLURM_JOB_ID}.json" || RC=1
done

for CELL in h2d_n64_acc h2d_n64_fast h2d_n128_acc h2d_n128_fast; do
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  echo ""; echo "########## r8A $CELL  X64=1 reduced-Gram ##########"
  X64=1 python3 -u h2opt.py --cell "$CELL" \
    --variants 'base;enc_jit;v4w;red;red_m;red64;red_jf' \
    --repeats 9 --warmup 3 \
    --out "$WORK/results/r8A_red_${CELL}_${STAMP}_${SLURM_JOB_ID}.json" || RC=1
done

# The reduced online solve is mesh-free, so graph capture should help it more
# than anything else measured so far.  Applied to the FOM in the same process.
for CELL in h2d_n128_acc h2d_n128_fast; do
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  echo ""; echo "########## r8A $CELL  X64=1 reduced + graph capture ##########"
  XLA_FLAGS="--xla_gpu_graph_level=3" X64=1 python3 -u h2opt.py --cell "$CELL" \
    --variants 'base;enc_jit;v4w;red;red_m' --repeats 9 --warmup 3 \
    --out "$WORK/results/r8A_redgraph_${CELL}_${STAMP}_${SLURM_JOB_ID}.json" || RC=1
done

################### Phase B -- Heat-3D counted loop + graph capture ###############
for CELL in h3d_n64_fast h3d_n32 h3d_n128_acc; do
  for F in "" "--xla_gpu_graph_level=3"; do
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    TAG=$([ -z "$F" ] && echo nograph || echo graph)
    echo ""; echo "########## r8B $CELL $TAG flag='$F' ##########"
    XLA_FLAGS="$F" python3 -u h3opt.py --cell "$CELL" \
      --variants 'base:eager;v5w;v5c' --repeats 9 --warmup 3 \
      --out "$WORK/results/r8B_${TAG}_${CELL}_${STAMP}_${SLURM_JOB_ID}.json" || RC=1
  done
done

echo ""; echo "=== hopt8 done $(date) rc=$RC ==="
exit $RC
