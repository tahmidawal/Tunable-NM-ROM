"""Exploration (local GB10, NOT a result): held-out rho of uniform sub-lattice rules at q=256, M=1088
on the INITIAL-FIT states w0 of training trajectories 8..47 (disjoint from the rule-fit trajectories
0..7 and from every evaluation cohort), at L = 256 / 512 / 1024.

hires-burgers found lat64's rho_max always attained at w0 (trajectory 15) and rising as the mesh
coarsens; this decides which lattice arms are worth a cluster job before any job is submitted.

    PYTHONPATH=... jaxrun python w0_lattice_rho.py 256 512 1024
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import topfix as TF
import ladder as LD
import hops as H
import hfast as HF

ROOT = Path(__file__).resolve().parents[3]
ck = pickle.load(open(ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl', 'rb'))
params = jax.tree_util.tree_map(jnp.asarray, LD.host(ck['params']))
Zold = np.asarray(ck['Z_tr'])
K = Zold.shape[1]
R = int(np.asarray(ck['params']['h_lin']).shape[1])
q, M = 256, 1088
C = jnp.asarray(np.load(ROOT / 'experiments/b-panel/inputs/directions_qtd02.npz')['C'])[:, :q]
bank = A.CoordBank(params, K, R)
head = TF.corrected_head(params, C, K)
Zsub = Zold[::max(1, len(Zold) // 8192)]
cold = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1), 48)[0]
train = e.params_draw(0, 128)
traj = list(range(8, 48))
out = {}
for L in [int(x) for x in sys.argv[1:]]:
    t0 = time.perf_counter()
    G = bank.on_grid(L)
    kx, ky, lam = H.modes_lean(L, M)
    sx, sy = (jnp.asarray(a) for a in H.sine_tables(L, kx, ky))
    Am = H.project_bank((G,), sx, sy, L)
    data = dict(A=Am, lam=jnp.asarray(lam), G=(G,))
    tab = HF.build_tables(params, C, K, data, cold)
    _, parts = HF.make_query(params, C, K, q, L, .005, 1., 'eq', parts=True)
    init = parts['initialize']
    W0 = []
    for i in traj:
        ph = train[i]
        W0.append(np.asarray(init(jnp.asarray(e.initial(L, ph)), data, cold, tab)[0]))
    hv = jax.jit(jax.vmap(head))
    states = np.asarray(hv(jnp.asarray(np.stack(W0))))
    tg = H.dense_targets((G,), states, sx, sy, L, chunk=8)
    res = {}
    for s in (32, 64, 128, 256):
        if s >= L or L % s:
            continue
        ij, w = H.lattice_rule(L, s)
        ops, ij, w = H.rule_ops(bank, L, kx, ky, ij, w)
        r = H.rho(ops['Pq'], H.sampled_advection(ops['G5'], states, L), tg)
        res[f'lat{s}'] = dict(m=int(len(ij)), rho_max=float(r.max()), argmax_traj=traj[int(r.argmax())],
                              rho_median=float(np.median(r)), over_bar=int(np.sum(r > .116)),
                              top4=[float(x) for x in np.sort(r)[-4:]])
        del ops
    out[L] = dict(seconds=time.perf_counter() - t0, rules=res)
    print(L, json.dumps(out[L]), flush=True)
    del G, Am, data, tab
    jax.clear_caches()
(Path(__file__).parent / f"w0_lattice_rho_{'_'.join(sys.argv[1:])}.json").write_text(json.dumps(out, indent=1) + '\n')
