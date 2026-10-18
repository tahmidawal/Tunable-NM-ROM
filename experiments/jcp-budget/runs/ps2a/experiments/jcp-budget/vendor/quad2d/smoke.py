"""Local smoke (GB10, tiny): evaluator validation before any GPU job (DESIGN.md section 7).

  S1  my mesh-path linear query == bkfast.make_linear_query (the Table-1 text) on fields, iterations, exits
  S2  continuum target converged: Gauss p^2 vs Gauss p'^2 tested advection, relative difference
  S3  gradient form vs flux form at a converged Gauss rule (validates the forward-mode bank gradients)
  S4  mesh rule 'all interior nodes' (lat L) == dense Phi^T a_h(G c)
  S5  one rollout per evaluator kind (dense, mesh lat64, point gauss64, flux gauss64): finite, error vs truth
Writes checks/smoke-<L>.json.
"""
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import qcore as Q

L = int(sys.argv[1]) if len(sys.argv) > 1 else 128
setting = sys.argv[2] if len(sys.argv) > 2 else 'fast'
out = {}
t0 = time.perf_counter()
mdl = Q.Model()
st = Q.SETTINGS[setting]
Rp, M = st['Rp'], st['M']
edges = [0, 128, 384, 512]
Grot = Q.build_rotated_bank(mdl, L, edges)
ops = Q.operators(Grot, edges, L, M)
G = Q.BK.prefix(Grot, edges, Rp)
Arot = ops['Arot'][:, :Rp]
ph = Q.cohort('dev6')
case = 0
u0 = jnp.asarray(Q.e.initial(L, ph[case]))
nu = float(ph[case, 4])
print('setup', round(time.perf_counter() - t0, 1), flush=True)

# truth
fom = Q.e.make_fom(L, .005)[0]
tr = np.asarray(fom(u0, nu, 1e-6, 1e-8)[0])
n0 = np.linalg.norm(np.asarray(u0))
err = lambda f: max(np.linalg.norm(a - b) / n0 for a, b in zip(np.asarray(f)[1:], tr[1:]))

# a state population: the lat64 rollout's internal states (computed in S1)
trust, Hall = mdl.trust_linear(Rp)
Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
codes = np.asarray(hv(jnp.asarray(Zsub))) @ mdl.Lrot[:Rp].T
cold = Q.build_cold_linear(mdl, Rp, codes)
ij, w = Q.mesh_rule('lat64', L)
dm = dict(A=Arot, lam=ops['lam'], G=G, **Q.mesh_data(mdl, Rp, ij, w, L, ops['kx'], ops['ky']))

# S1 parity with bkfast
fq_bk = Q.BK.make_linear_query(Rp, L, .005, trust, step_budget=600, gtol=1e-3, ridge=1e-10, solver='chol',
                               clip=True, lam_carry=True, predictor='quad')
vb = fq_bk(u0, nu, dm, cold, {})
fq_me, dec = Q.make_linear_query('mesh', Rp, L, .005, trust)
vm = fq_me(u0, nu, dm, cold)
fb, fm = np.asarray(vb[0]), np.asarray(vm['fields'])
out['S1_parity_bkfast'] = dict(fields_rel=float(np.linalg.norm(fb - fm) / np.linalg.norm(fb)),
                               iterations_identical=bool(np.array_equal(np.asarray(vb[1]), np.asarray(vm['it']))),
                               reasons_identical=bool(np.array_equal(np.asarray(vb[3]), np.asarray(vm['reason']))),
                               err_lat64_pct=100 * err(fm))
print('S1', out['S1_parity_bkfast'], flush=True)
states = np.asarray(vm['internal'])[1:]

# S2 continuum target convergence, S3 point vs flux
vals = {}
for p in (96, 128, 160, 200):
    X, wq = Q.offmesh_rule(f'gauss{p}')
    for form in ('point', 'flux'):
        d = Q.offmesh_data(mdl, Rp, X, wq, L, ops['kx'], ops['ky'], form)
        f = jax.jit(jax.vmap(Q.tested_value(form, L), in_axes=(0, None)))
        vals[(p, form)] = np.asarray(f(jnp.asarray(states), d))
rel = lambda a, b: float(np.max(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))
out['S2_gauss_convergence_point'] = {f'{p}_vs_200': rel(vals[(p, 'point')], vals[(200, 'point')]) for p in (96, 128, 160)}
out['S3_point_vs_flux'] = {str(p): rel(vals[(p, 'point')], vals[(p, 'flux')]) for p in (96, 128, 160, 200)}
print('S2', out['S2_gauss_convergence_point'], 'S3', out['S3_point_vs_flux'], flush=True)

# S4 all-node mesh rule == dense
ija, wa = Q.mesh_rule(f'lat{L}', L)
da = Q.mesh_data(mdl, Rp, ija, wa, L, ops['kx'], ops['ky'])
dd = dict(G=G, sx=ops['sx'], sy=ops['sy'])
fa = jax.jit(jax.vmap(Q.tested_value('mesh', L), in_axes=(0, None)))(jnp.asarray(states[:10]), da)
fd = jax.jit(jax.vmap(Q.tested_value('dense', L), in_axes=(0, None)))(jnp.asarray(states[:10]), dd)
out['S4_allnodes_vs_dense'] = rel(np.asarray(fa), np.asarray(fd))
out['S4b_mesh_vs_continuum_rho_max'] = rel(np.asarray(fd), vals[(200, 'point')][:10])
print('S4', out['S4_allnodes_vs_dense'], out['S4b_mesh_vs_continuum_rho_max'], flush=True)

# S5 rollouts
X64, w64 = Q.offmesh_rule('gauss64')
for kind, data in (('dense', dict(A=Arot, lam=ops['lam'], G=G, sx=ops['sx'], sy=ops['sy'])),
                   ('point', dict(A=Arot, lam=ops['lam'], G=G, **Q.offmesh_data(mdl, Rp, X64, w64, L, ops['kx'], ops['ky'], 'point'))),
                   ('flux', dict(A=Arot, lam=ops['lam'], G=G, **Q.offmesh_data(mdl, Rp, X64, w64, L, ops['kx'], ops['ky'], 'flux')))):
    t1 = time.perf_counter()
    fq, _ = Q.make_linear_query(kind, Rp, L, .005, trust, tangent_chunks=4 if kind == 'dense' else 1)
    v = fq(u0, nu, data, cold)
    f = np.asarray(v['fields'])
    out[f'S5_{kind}'] = dict(finite=bool(np.isfinite(f).all()), err_pct=100 * err(f),
                             vs_lat64_pct=100 * max(np.linalg.norm(a - b) / n0 for a, b in zip(f[1:], fm[1:])),
                             iters=int(np.sum(np.asarray(v['it']))), reasons=sorted(set(np.asarray(v['reason']).tolist())),
                             seconds=time.perf_counter() - t1)
    print('S5', kind, out[f'S5_{kind}'], flush=True)
out['L'] = L
out['setting'] = setting
out['backend'] = jax.default_backend()
json.dump(out, open(Q.HERE / 'checks' / f'smoke-{setting}-{L}.json', 'w'), indent=1)
print('SMOKE DONE', round(time.perf_counter() - t0, 1))
