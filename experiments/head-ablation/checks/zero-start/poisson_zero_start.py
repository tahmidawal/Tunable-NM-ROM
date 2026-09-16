"""Poisson: what happens if the latent solve starts from z = 0 every time?

Same frozen checkpoint, bank, head, test modes, budget, tolerance and sources as the
head-ablation Poisson job (pabl01). Only the starting point of the LM solve changes.
Local diagnostic; writes a JSON and prints a table. No repository file is touched.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-head-ablation')
for rel in ('experiments/head-ablation', 'experiments/mr-burgers2d', 'experiments/multiresolution-poisson'):
    sys.path.insert(0, str(ROOT / rel))
import core as C              # noqa: E402
import pilot as P             # noqa: E402
import arms as A              # noqa: E402
import sep_common as sc       # noqa: E402
import poisson_ablation as PA  # noqa: E402

RUN = ROOT / 'experiments/multiresolution-poisson/runs/correction_accuracy10'
CFG = json.load(open(ROOT / 'experiments/head-ablation/config-poisson-ablation.json'))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
MESHES = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else CFG['intervals']


def make_kernel(head, B, bank, n, trust, budget, gtol, policy):
    lm = A.make_stationary_lm(lambda z, fm: B @ head(z) - fm, budget, trust, gtol, 'gj')

    @jax.jit
    def kernel(source, S, I, J, W, predictions, codes):
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        index = jnp.argmin(jnp.sum((predictions - fm[None, :]) ** 2, axis=1))
        if policy == 'nearest':
            z0 = codes[index]
        elif policy == 'mean':
            z0 = jnp.mean(codes, axis=0)
        else:
            z0 = jnp.zeros((codes.shape[1],), dtype=codes.dtype)
        z, rn, it, reason, gn = lm(z0, (fm,), 0.)
        field = jnp.pad((bank @ head(z)).reshape(n - 1, n - 1), 1)
        return field, z, rn, it, reason, gn, jnp.linalg.norm(fm)
    return kernel


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    params, codes, _ = sc.load_pkl(RUN / 'checkpoints/r128_joint.pkl')
    codes = np.asarray(codes)
    K = codes.shape[1]
    head = lambda z: sc.head(params, z)
    draws = np.concatenate((C.source_params(CFG['eval_seed'], CFG['eval_count']), C.source_params(CFG['fresh_seed'], CFG['fresh_count'])))
    budget, gtol = CFG['lm_budget'], CFG['stationarity_tolerance']
    trust = PA.radius(codes)
    policies = [('nearest_code', 'nearest', trust), ('mean_code', 'mean', trust),
                ('zero', 'zero', trust), ('zero_no_trust', 'zero', np.inf)]
    out = dict(checkpoint=str(RUN / 'checkpoints/r128_joint.pkl'), K=K, budget=budget, gtol=gtol,
               trust_radius=float(trust), code_norms=dict(mean=float(np.linalg.norm(codes, axis=1).mean()),
                                                          max=float(np.linalg.norm(codes, axis=1).max()),
                                                          mean_code_norm=float(np.linalg.norm(codes.mean(0)))),
               rows=[])
    for n in MESHES:
        t0 = time.perf_counter()
        ops = C.assemble(params, codes, n, CFG['requested_modes'], budget)
        B, bank = ops['B'], ops['bank']
        lam = jnp.asarray(C.eigenvalues(n))
        pred = jax.jit(jax.vmap(lambda z: B @ head(z)))(jnp.asarray(codes))
        print(f'mesh {n}: assembled in {time.perf_counter() - t0:.1f}s, M={int(B.shape[0])}', flush=True)
        for name, policy, tr in policies:
            kern = make_kernel(head, B, bank, n, tr, budget, gtol, policy)
            for ci, q in enumerate(draws):
                src = jnp.asarray(C.full_source(n, q))
                truth = np.asarray(C.dst_solve(src, lam))
                res = kern(src, ops['S'], ops['I'], ops['J'], ops['W'], pred, jnp.asarray(codes))
                jax.block_until_ready(res)
                t1 = time.perf_counter()
                res = kern(src, ops['S'], ops['I'], ops['J'], ops['W'], pred, jnp.asarray(codes))
                jax.block_until_ready(res)
                ms = (time.perf_counter() - t1) * 1e3
                field, z, rn, it, reason, gn, fmn = jax.device_get(res)
                out['rows'].append(dict(mesh=n, start=name, case=ci, same_grid_error=float(C.relative(field, truth)),
                                        iterations=int(it), reason=int(reason), stationarity=float(gn),
                                        relative_residual=float(rn) / max(float(fmn), 1e-300),
                                        latent_norm=float(np.linalg.norm(z)), ms=ms))
            sel = [r for r in out['rows'] if r['mesh'] == n and r['start'] == name]
            errs = [r['same_grid_error'] for r in sel]
            print(f"  {name:14s} worst {max(errs) * 100:8.4f}%  median {np.median(errs) * 100:8.4f}%  "
                  f"iters med/max {np.median([r['iterations'] for r in sel]):.0f}/{max(r['iterations'] for r in sel)}  "
                  f"reasons {sorted(set(r['reason'] for r in sel))}  stationary {sum(r['stationarity'] <= gtol for r in sel)}/{len(sel)}  "
                  f"ms {np.median([r['ms'] for r in sel]):.2f}", flush=True)
    if OUT:
        OUT.write_text(json.dumps(out, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
