"""G5 R3: retrain the neural velocity bank on transient POD teachers (DESIGN-G5 §5.4).

Same recipe as the steady run-09 lineage, fixed before any transient data:
  parent  vendor spatial_bank architecture (21 features -> 3x128 tanh -> 2r, envelope),
          seed 3112026; objective sum_i w_i (value + D-scaled gradient relative errors),
          w_i ~ sqrt(sigma_i/sigma_1); Adam 4000 (batch 4096, cosine 1e-3 -> 1e-4, full
          check every 200) then L-BFGS 300 (history 10, 20 line-search steps).
  child   frozen parent + r residual heads 21-64-64-2 (seed 30102026, zero last layer so
          the child starts at the parent); objective base + 1.0*divergence + 1.0*cylinder
          traction; L-BFGS 1000 iterations or 2700 s, whichever first.
Selection: lowest full training objective only. GPU, f64, highest precision.
L-BFGS is scipy's L-BFGS-B (no bounds) on the full-rule objective; the steady lane used its
own deterministic L-BFGS with Armijo search (same method class; implementation differs).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))

BLOCK = 32768


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True); ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--eval-only', help='trained_bank_params.npz: evaluate on this export Q only (quadrature check)')
    ap.add_argument('--out', required=True); ap.add_argument('--smoke', action='store_true',
                                                             help='code smoke: 20 Adam, 5+5 L-BFGS steps')
    a = ap.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True); jax.config.update('jax_default_matmul_precision', 'highest')
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from accuracy_v2.spatial_bank import evaluate_layers, initialize
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    r = a.rank
    steps = {'adam': 20, 'lbfgs': 5, 'child': 5, 'child_seconds': 120} if a.smoke else \
            {'adam': 4000, 'lbfgs': 300, 'child': 1000, 'child_seconds': 2700}
    ex = Path(a.export)
    q = np.load(ex / 'quadrature.npz'); b = np.load(ex / 'basis.npz'); pod = np.load(ex / 'pod.npz')
    names = [str(n) for n in b['names']]
    cols = [names.index(f'pod{i}') for i in range(r)]
    x, w = q['points'], q['weights']
    Vt = b['V'][..., cols]; Gt = 0.1 * b['G'][..., cols]          # D-scaled gradient teacher
    xo, wo, no = q['obstacle_points'], q['obstacle_weights'] / 0.1, q['obstacle_normal_out_of_body']
    Gto = 0.1 * b['obstacle_G'][..., cols]
    sig = pod['singular_values'][:r]
    imp = np.sqrt(sig / sig[0]); imp = imp / imp.sum()
    nv = np.einsum('n,ncr,ncr->r', w, Vt, Vt); ng_ = np.einsum('n,ncdr,ncdr->r', w, Gt, Gt)
    trac = lambda G, n: jnp.einsum('nijr,nj->nir', G + jnp.swapaxes(G, 1, 2), n)
    t_teacher = np.einsum('nijr,nj->nir', Gto + np.swapaxes(Gto, 1, 2), no)
    tr_energy = float(np.sum(sig ** 2 * np.einsum('n,nir,nir->r', wo, t_teacher, t_teacher)))
    record = {'rank': r, 'steps': steps, 'points': int(len(w)), 'obstacle_points': int(len(wo)),
              'singular_values': sig.tolist(), 'mode_weights': imp.tolist(),
              'export_files': json.loads((ex / 'EXPORT.json').read_text())['files'], 'history': []}

    model = initialize(rank=r, width=128, depth=3, seed=3112026)
    shapes = [s.shape for layer in model.layers for s in layer]
    sizes = [int(np.prod(s)) for s in shapes]; offs = np.cumsum([0] + sizes)
    unpack = lambda t, sh=shapes, of=offs: [(t[of[2 * i]:of[2 * i + 1]].reshape(sh[2 * i]),
                                             t[of[2 * i + 1]:of[2 * i + 2]].reshape(sh[2 * i + 1])) for i in range(len(sh) // 2)]
    theta0 = np.concatenate([s.ravel() for layer in model.layers for s in layer])

    def base_terms(val, grd, yv, yg, ww):
        ev = jnp.einsum('n,ncr,ncr->r', ww, val - yv, val - yv) / nv
        eg = jnp.einsum('n,ncdr,ncdr->r', ww, grd - yg, grd - yg) / ng_
        return jnp.sum(imp * (ev + eg)), ev, eg

    def parent_obj(t, xx, yv, yg, ww):
        val, grd = evaluate_layers(unpack(t), xx, jnp)
        return base_terms(val, 0.1 * grd, yv, yg, ww)[0]

    pg = jax.jit(jax.value_and_grad(parent_obj))

    def full(fn, t):
        f, g = 0.0, np.zeros_like(t)
        for i in range(0, len(w), BLOCK):
            e = slice(i, i + BLOCK)
            fi, gi = fn(jnp.asarray(t), x[e], Vt[e], Gt[e], w[e])
            f += float(fi); g += np.asarray(gi)
        if not np.isfinite(f) or not np.isfinite(g).all():
            raise FloatingPointError('nonfinite objective')
        return f, g

    loaded = np.load(a.eval_only) if a.eval_only else None
    if loaded is not None and int(loaded['rank']) != r:
        raise ValueError('eval-only parameters have a different rank')
    seen = {'best': (None, loaded['parent'], 'loaded')} if loaded is not None else None
    if loaded is None:
        # ---- parent: Adam then L-BFGS ---------------------------------------
        rng = np.random.default_rng(3112026)
        theta = theta0.copy(); m = np.zeros_like(theta); v = np.zeros_like(theta)
        best = (full(pg, theta)[0], theta.copy(), 'initial')
        t0 = time.time()
        for k in range(1, steps['adam'] + 1):
            idx = rng.integers(0, len(w), 4096)
            _, g = pg(jnp.asarray(theta), x[idx], Vt[idx], Gt[idx], w[idx] * len(w) / len(idx))
            g = np.asarray(g)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            rate = 1e-3 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * k / steps['adam'])))
            theta = theta - rate * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
            if k % 200 == 0 or k == steps['adam']:
                f = full(pg, theta)[0]
                record['history'].append({'phase': 'adam', 'step': k, 'objective': f, 'seconds': time.time() - t0})
                if f < best[0]:
                    best = (f, theta.copy(), f'adam{k}')
        theta = best[1].copy()
        seen = {'best': best}

        def fun_parent(t):
            f, g = full(pg, t)
            if f < seen['best'][0]:
                seen['best'] = (f, t.copy(), 'lbfgs')
            return f, g
        res = minimize(fun_parent, theta, jac=True, method='L-BFGS-B',
                       options={'maxiter': steps['lbfgs'], 'maxcor': 10, 'maxls': 20, 'gtol': 1e-10, 'ftol': 0.0})
        record['parent'] = {'lbfgs_iterations': int(res.nit), 'lbfgs_message': str(res.message),
                            'selected': seen['best'][2], 'objective': seen['best'][0], 'seconds': time.time() - t0}
    parent = unpack(seen['best'][1])

    # ---- child: frozen parent + residual heads ------------------------------
    hr = np.random.default_rng(30102026)
    hshapes = [(21, 64), (64,), (64, 64), (64,), (64, 2), (2,)]
    heads = []
    for _ in range(r):
        heads += [hr.normal(size=(21, 64)) / np.sqrt(21), np.zeros(64), hr.normal(size=(64, 64)) / 8.0, np.zeros(64),
                  np.zeros((64, 2)), np.zeros(2)]
    phi0 = np.concatenate([h.ravel() for h in heads])
    hsz = [int(np.prod(s)) for s in hshapes] * r; hoff = np.cumsum([0] + hsz)
    def unpack_heads(p):
        parts = [p[hoff[i]:hoff[i + 1]].reshape(hshapes[i % 6]) for i in range(6 * r)]
        return [[(parts[6 * j], parts[6 * j + 1]), (parts[6 * j + 2], parts[6 * j + 3]), (parts[6 * j + 4], parts[6 * j + 5])] for j in range(r)]
    parent_j = [(jnp.asarray(W), jnp.asarray(c)) for W, c in parent]

    def composite(p, xx):
        val, grd = evaluate_layers(parent_j, xx, jnp)
        hv, hg = [], []
        for h in unpack_heads(p):
            a1, a2 = evaluate_layers(h, xx, jnp); hv.append(a1); hg.append(a2)
        return val + jnp.concatenate(hv, -1), grd + jnp.concatenate(hg, -1)

    def child_vol(p, xx, yv, yg, ww):
        val, grd = composite(p, xx); grd = 0.1 * grd
        base, _, _ = base_terms(val, grd, yv, yg, ww)
        tr = grd[:, 0, 0, :] + grd[:, 1, 1, :] - (yg[:, 0, 0, :] + yg[:, 1, 1, :])
        div = jnp.sum(imp * jnp.einsum('n,nr,nr->r', ww, tr, tr) / ng_)
        return base + 1.0 * div

    def child_obstacle(p):
        _, grd = composite(p, jnp.asarray(xo)); grd = 0.1 * grd
        err = trac(grd, jnp.asarray(no)) - t_teacher
        return 1.0 * jnp.sum(sig ** 2 * jnp.einsum('n,nir,nir->r', wo, err, err)) / tr_energy

    cg = jax.jit(jax.value_and_grad(child_vol)); og = jax.jit(jax.value_and_grad(child_obstacle))
    cbest = {'f': None, 'p': loaded['heads'].copy(), 'evals': 0} if loaded is not None else None
    if loaded is None:
        t1 = time.time(); cbest = {'f': np.inf, 'p': phi0.copy(), 'evals': 0}

        class Deadline(Exception):
            pass

        def fun_child(p):
            f, g = full(cg, p)
            fo, go = og(jnp.asarray(p)); f += float(fo); g = g + np.asarray(go)
            cbest['evals'] += 1
            if f < cbest['f']:
                cbest.update(f=f, p=p.copy())
            if time.time() - t1 > steps['child_seconds']:
                raise Deadline()
            return f, g
        try:
            res = minimize(fun_child, phi0, jac=True, method='L-BFGS-B',
                           options={'maxiter': steps['child'], 'maxcor': 10, 'maxls': 20, 'gtol': 1e-10, 'ftol': 0.0})
            stop = {'iterations': int(res.nit), 'message': str(res.message)}
        except Deadline:
            stop = {'iterations': None, 'message': 'deadline stop (2700 s), not convergence'}
        record['child'] = dict(stop, evaluations=cbest['evals'], objective=cbest['f'], seconds=time.time() - t1)
    # ---- evaluate the selected composite on Q and on the boundary rule ---------
    def evaluate(pts):
        vs, gs = [], []
        for i in range(0, len(pts), BLOCK):
            vv, gg = composite(jnp.asarray(cbest['p']), pts[i:i + BLOCK]); vs.append(np.asarray(vv)); gs.append(np.asarray(gg))
        return np.concatenate(vs), np.concatenate(gs)
    V, G = evaluate(x)
    bsave = {}
    for lab in ('inlet', 'outlet', 'obstacle'):
        bv, bg = evaluate(q[f'{lab}_points']); bsave[f'{lab}_V'] = bv; bsave[f'{lab}_G'] = bg
    pdig = hashlib.sha256(np.ascontiguousarray(seen['best'][1]).tobytes() + np.ascontiguousarray(cbest['p']).tobytes()).hexdigest()
    np.savez(out / 'trained_bank_eval.npz', V=V, G=G, params_digest=np.array(pdig), **bsave)
    if loaded is None:
        np.savez(out / 'trained_bank_params.npz', parent=seen['best'][1], heads=cbest['p'], rank=r)
    record['outputs'] = {p.name: sha(p) for p in out.glob('*.npz')}
    record['runtime'] = {'jax': jax.__version__, 'backend': jax.default_backend(), 'devices': [d.device_kind for d in jax.devices()]}
    (out / 'TRAINING.json').write_text(json.dumps(record, indent=2, default=float) + '\n')
    print(json.dumps({k: record.get(k) for k in ('parent', 'child')}, default=float))


if __name__ == '__main__':
    main()
