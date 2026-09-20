"""Exploration (local, 256^2, development case 0 only): held-out-style rho of a uniform
sub-lattice quadrature against the NNLS eqtop rule, on states the EQ query actually visits.
Not a result; decides whether the lattice rule is worth a cluster arm."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import engines as e, arms as A, topfix as TF, ladder as LD

ROOT = Path(__file__).resolve().parents[3]
ck = pickle.load(open(ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl', 'rb'))
params = jax.tree_util.tree_map(jnp.asarray, LD.host(ck['params']))
Zold = np.asarray(ck['Z_tr']); K = Zold.shape[1]; R = int(np.asarray(ck['params']['h_lin']).shape[1])
L, dt = 256, .005
q = int(sys.argv[1]) if len(sys.argv) > 1 else 256
bank = A.CoordBank(params, K, R); G = bank.on_grid(L)
C = jnp.asarray(np.load(ROOT / 'experiments/b-panel/inputs/directions_qtd02.npz')['C'])[:, :q]
rf = {0: 'rule_q0_m1024_qrg304_reachable.npz', 128: 'rule_q128_m2319_bet101_rhow64.npz', 256: 'rule_q256_m2560_bet101_fs64.npz'}[q]
z = np.load(ROOT / 'experiments/b-panel/inputs/rules-eqtop' / rf)
nodes, weights = z['nodes'].astype(int), z['weights']
trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
Zsub = Zold[::max(1, len(Zold) // 8192)]
head = TF.corrected_head(params, C, K)
cold, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1), 48)
out = {}
phys = np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))

def rule_ops(pos, w, Phi):
    ij = np.stack(np.unravel_index(pos, (L - 1, L - 1)), 1) + 1
    return dict(G5=bank.stencil(ij, L), Pq=jnp.asarray(Phi[pos] * w[:, None]))

def lattice(s):
    k = np.arange(1, s) * (L // s)            # interior lattice lines, grid index
    ii, jj = np.meshgrid(k, k, indexing='ij')
    pos = ((ii - 1) * (L - 1) + (jj - 1)).ravel()
    return pos, np.full(len(pos), float((L // s) ** 2))

def rho(ops, P, states):
    def one(c):
        full = P.T @ e.spatial(G @ c, L)[0]
        us = jnp.einsum('msr,r->ms', ops['G5'], c)
        cc, xp, xm, yp, ym = [us[:, i] for i in range(5)]
        a = cc * L * (jnp.where(cc > 0, cc - xm, xp - cc) + jnp.where(cc > 0, cc - ym, yp - cc))
        return jnp.linalg.norm(ops['Pq'].T @ a - full) / jnp.linalg.norm(full)
    return np.asarray(jax.jit(jax.vmap(one))(states))

for M in sorted({4 * (K + q), 2 * (K + q)}, reverse=True):
    Phi, lam, _ = e.modes(L, M); P = jnp.asarray(Phi)
    data = dict(A=P.T @ G, lam=jnp.asarray(lam), G=G)
    if M == 4 * (K + q):
        ops = rule_ops(nodes, weights, Phi)
        qf = TF.make_query(params, C, K, q, L, dt, trust, 'eq', 'base', ic_budget=400, step_budget=600, gtol=1e-3,
                           ic_gtol=1e-6, linear='lu' if K + q > 64 else 'gj', inner_damping=1e-10, tau_y=.1)
        t = time.perf_counter()
        v = qf(jnp.asarray(e.initial(L, phys[0])), float(phys[0, 4]), dict(data, **ops), cold)
        W = np.asarray(v[7]); print('query s', time.perf_counter() - t, 'iters', np.asarray(v[1]).tolist())
        states = jax.vmap(head)(jnp.asarray(W))
        r = rho(ops, P, states); out[f'M{M}_nnls_m{len(nodes)}'] = [float(r.max()), float(np.median(r)), int(r.argmax()), [float(x) for x in np.sort(r)[-4:]]]
    for s in (32, 64):
        pos, w = lattice(s)
        r = rho(rule_ops(pos, w, Phi), P, states)
        out[f'M{M}_lattice{s}_m{len(pos)}'] = [float(r.max()), float(np.median(r)), int(r.argmax()), [float(x) for x in np.sort(r)[-4:]]]
    print(json.dumps(out), flush=True)
