"""Poisson: error and cost against the LM iteration cap, from z = 0 and from the nearest code.
Same frozen checkpoint, bank, head, tests, tolerance and 12 sources as pabl01. Local diagnostic."""
import json, sys, time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-head-ablation')
for rel in ('experiments/head-ablation', 'experiments/mr-burgers2d', 'experiments/multiresolution-poisson'):
    sys.path.insert(0, str(ROOT / rel))
import core as C, arms as A, sep_common as sc, poisson_ablation as PA  # noqa
RUN = ROOT / 'experiments/multiresolution-poisson/runs/correction_accuracy10'
CFG = json.load(open(ROOT / 'experiments/head-ablation/config-poisson-ablation.json'))
OUT = Path(sys.argv[1]); MESHES = [int(x) for x in sys.argv[2].split(',')]
CAPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 300]

def make_kernel(head, B, bank, n, trust, budget, gtol, policy):
    lm = A.make_stationary_lm(lambda z, fm: B @ head(z) - fm, budget, trust, gtol, 'gj')
    @jax.jit
    def kernel(source, S, I, J, W, predictions, codes):
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        index = jnp.argmin(jnp.sum((predictions - fm[None, :]) ** 2, axis=1))
        z0 = codes[index] if policy == 'nearest' else jnp.zeros((codes.shape[1],), dtype=codes.dtype)
        z, rn, it, reason, gn = lm(z0, (fm,), 0.)
        return jnp.pad((bank @ head(z)).reshape(n - 1, n - 1), 1), z, rn, it, reason, gn
    return kernel

def main():
    assert jax.default_backend() == 'gpu'
    print('jax_backend=gpu', flush=True)
    params, codes, _ = sc.load_pkl(RUN / 'checkpoints/r128_joint.pkl'); codes = np.asarray(codes)
    head = lambda z: sc.head(params, z)
    draws = np.concatenate((C.source_params(CFG['eval_seed'], CFG['eval_count']), C.source_params(CFG['fresh_seed'], CFG['fresh_count'])))
    gtol, trust = CFG['stationarity_tolerance'], PA.radius(codes)
    out = dict(caps=CAPS, rows=[])
    for n in MESHES:
        ops = C.assemble(params, codes, n, CFG['requested_modes'], 300)
        B, bank, lam = ops['B'], ops['bank'], jnp.asarray(C.eigenvalues(n))
        pred = jax.jit(jax.vmap(lambda z: B @ head(z)))(jnp.asarray(codes))
        srcs = [jnp.asarray(C.full_source(n, q)) for q in draws]
        truths = [np.asarray(C.dst_solve(s, lam)) for s in srcs]
        for policy in ('zero', 'nearest'):
            for cap in CAPS:
                kern = make_kernel(head, B, bank, n, trust, cap, gtol, policy)
                errs, its, ms, reasons = [], [], [], []
                for src, truth in zip(srcs, truths):
                    args = (src, ops['S'], ops['I'], ops['J'], ops['W'], pred, jnp.asarray(codes))
                    jax.block_until_ready(kern(*args))
                    reps = []
                    for _ in range(3):
                        t1 = time.perf_counter(); res = kern(*args); jax.block_until_ready(res); reps.append((time.perf_counter() - t1) * 1e3)
                    field, z, rn, it, reason, gn = jax.device_get(res)
                    errs.append(float(C.relative(field, truth))); its.append(int(it)); ms.append(float(np.median(reps))); reasons.append(int(reason))
                    out['rows'].append(dict(mesh=n, start=policy, cap=cap, error=errs[-1], iterations=its[-1], reason=reasons[-1], ms=ms[-1]))
                print(f"mesh {n} {policy:8s} cap {cap:3d}: worst {max(errs)*100:8.4f}%  median {np.median(errs)*100:7.4f}%  "
                      f"iters med {np.median(its):.0f}  budget-exits {sum(r == 0 for r in reasons)}/12  ms {np.median(ms):.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=1)); print('DONE', flush=True)
main()
