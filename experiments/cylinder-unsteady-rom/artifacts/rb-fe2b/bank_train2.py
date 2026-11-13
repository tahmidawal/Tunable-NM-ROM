"""Rebuttal push: flag-driven training of the coordinate-network velocity bank (PROTOCOL.md step 2).

One network, same boundary envelope and base features as the vendor bank; options:
  --width/--depth        trunk size                      --fx          extra streamwise Fourier frequencies
  --stream               stream-function (divergence-free) construction, u = curl psi
  --div-weight           divergence penalty              --box-weight  loss weight in x in [0.12, 1.3]
  --ft vp|plain          second L-BFGS stage; vp = variable-projection subspace objective
  --ft-div-weight/--ft-box-weight   penalty/weighting used in the second stage
Teacher: POD modes 0..r-1 on the export rule; objective sum_i w_i (value + D-scaled gradient relative
errors [+ div]); w_i ~ sqrt(sigma_i/sigma_1) (--mode-weight sqrt) or sigma_i/sigma_1 (sigma).
Adam (GPU, minibatch) then scipy L-BFGS-B on the full rule until an iteration cap or a wall deadline.
Output format matches bank_train.py (trained_bank_eval.npz: V, G, boundary traces), so
rom_project.py / rom_r4.py are used unchanged.  The saved functions are in POD order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

CENTER = (0.2, 0.2); RADIUS = 0.05; HEIGHT = 0.41
BASE_F = (1.0, 2.0, 4.0, 8.0); WIDTHS = (0.1, 0.2, 0.4)
BOX = (0.12, 1.3)
BUMP = 0.12   # support of the cylinder stream-function bump (gap to walls/inlet is 0.15)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(jnp, jax, rank, fx, stream):
    """Per-point bank: x (2,) -> velocity (2, r); vmapped values and physical gradients."""
    off = jnp.asarray([1.1, 0.205]); cen = jnp.asarray(CENTER)

    def feats(x):
        z = (x - off) / off
        parts = [z]
        for f in BASE_F:
            t = jnp.pi * f * z
            parts += [jnp.sin(t), jnp.cos(t)]
        for f in fx:
            t = jnp.pi * f * z[0]
            parts += [jnp.stack([jnp.sin(t), jnp.cos(t)])]
        d = x - cen; d2 = jnp.sum(d * d)
        parts += [jnp.stack([jnp.exp(-d2 / (2 * wd ** 2)) for wd in WIDTHS])]
        return jnp.concatenate(parts)

    def env(x):
        d = x - cen; rad = jnp.sqrt(jnp.sum(d * d))
        dist = jnp.stack([x[0], x[1], HEIGHT - x[1], rad - RADIUS])
        return jnp.prod(-jnp.expm1(-dist / 0.1))

    def mlp(layers, h):
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = jnp.tanh(h)
        return h

    if not stream:
        def vel(p, x):
            return env(x) * mlp(p['layers'], feats(x)).reshape(2, rank)
    else:
        def psi(p, x):
            d = x - cen; rad = jnp.sqrt(jnp.sum(d * d))
            s = jnp.clip((rad - RADIUS) / BUMP, 0.0, 1.0)
            return p['c'] * (1 - s * s) ** 4 + env(x) ** 2 * mlp(p['layers'], feats(x))

        def vel(p, x):
            J = jax.jacfwd(psi, argnums=1)(p, x)        # (r, 2)
            return jnp.stack([J[:, 1], -J[:, 0]])

    def both(p, x):
        v = vel(p, x)
        g = jax.jacfwd(vel, argnums=1)(p, x)            # (2, r, 2): component, function, derivative
        return v, jnp.transpose(g, (0, 2, 1))

    nfeat = 2 + 4 * len(BASE_F) + 2 * len(fx) + len(WIDTHS)
    return jax.vmap(both, in_axes=(None, 0)), nfeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True); ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=128); ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--fx', default=''); ap.add_argument('--stream', action='store_true')
    ap.add_argument('--mode-weight', choices=('sqrt', 'sigma'), default='sqrt')
    ap.add_argument('--div-weight', type=float, default=0.0); ap.add_argument('--box-weight', type=float, default=1.0)
    ap.add_argument('--adam-steps', type=int, default=20000); ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lbfgs-iters', type=int, default=100000); ap.add_argument('--lbfgs-seconds', type=float, default=1800)
    ap.add_argument('--ft', choices=('none', 'vp', 'plain'), default='none')
    ap.add_argument('--ft-seconds', type=float, default=900); ap.add_argument('--ft-iters', type=int, default=100000)
    ap.add_argument('--ft-div-weight', type=float); ap.add_argument('--ft-box-weight', type=float)
    ap.add_argument('--chunk', type=int, default=32768); ap.add_argument('--seed', type=int, default=3112026)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True); jax.config.update('jax_default_matmul_precision', 'highest')
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required')
    print(f'jax_backend={jax.default_backend()}', flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    r = a.rank; fx = tuple(float(s) for s in a.fx.split(',') if s)
    if a.smoke:
        a.adam_steps, a.lbfgs_iters, a.ft_iters = 30, 5, 5
    ex = Path(a.export)
    q = np.load(ex / 'quadrature.npz'); b = np.load(ex / 'basis.npz'); pod = np.load(ex / 'pod.npz')
    names = [str(n) for n in b['names']]
    cols = [names.index(f'pod{i}') for i in range(r)]
    x, w = q['points'], q['weights']
    N = len(w); CH = min(a.chunk, N); pad = (-N) % CH
    Vt = b['V'][..., cols]; Gt = 0.1 * b['G'][..., cols]
    sig = pod['singular_values'][:r]
    imp = np.sqrt(sig / sig[0]) if a.mode_weight == 'sqrt' else sig / sig[0]
    imp = imp / imp.sum()
    inbox = ((x[:, 0] >= BOX[0]) & (x[:, 0] <= BOX[1])).astype(float)
    xj = jnp.asarray(np.concatenate([x, np.repeat(x[:1], pad, 0)]))
    Vj = jnp.asarray(np.concatenate([Vt, np.zeros((pad,) + Vt.shape[1:])]))
    Gj = jnp.asarray(np.concatenate([Gt, np.zeros((pad,) + Gt.shape[1:])]))
    del Vt, Gt
    impj = jnp.asarray(imp)

    def weights(box):
        ww = w * (1.0 + (box - 1.0) * inbox)
        return jnp.asarray(np.concatenate([ww, np.zeros(pad)]))

    def norms(wj):
        nv = sum(jnp.einsum('n,ncr,ncr->r', wj[i:i + CH], Vj[i:i + CH], Vj[i:i + CH]) for i in range(0, N + pad, CH))
        ng_ = sum(jnp.einsum('n,ncdr,ncdr->r', wj[i:i + CH], Gj[i:i + CH], Gj[i:i + CH]) for i in range(0, N + pad, CH))
        return nv, ng_

    bank, nfeat = build(jnp, jax, r, fx, a.stream)
    rng = np.random.default_rng(a.seed)
    nout = r if a.stream else 2 * r
    sizes = [nfeat] + [a.width] * a.depth + [nout]
    params = {'layers': [(jnp.asarray(rng.normal(size=(m, n)) / np.sqrt(m)), jnp.zeros(n)) for m, n in zip(sizes[:-1], sizes[1:])]}
    if a.stream:
        params['c'] = jnp.zeros(r)
    theta0, unravel = ravel_pytree(params)
    record = {'args': vars(a), 'rank': r, 'points': int(N), 'features': nfeat, 'parameters': int(theta0.size),
              'singular_values': sig.tolist(), 'mode_weights': imp.tolist(), 'history': [],
              'export_files': json.loads((ex / 'EXPORT.json').read_text())['files']}

    def terms(val, grd, yv, yg, ww):
        """Per-mode weighted squared errors: value, D-scaled gradient, divergence."""
        ev = jnp.einsum('n,ncr,ncr->r', ww, val - yv, val - yv)
        eg = jnp.einsum('n,ncdr,ncdr->r', ww, grd - yg, grd - yg)
        tr = grd[:, 0, 0, :] + grd[:, 1, 1, :] - (yg[:, 0, 0, :] + yg[:, 1, 1, :])
        return ev, eg, jnp.einsum('n,nr,nr->r', ww, tr, tr)

    def make(wj, lam, vp):
        nv, ng_ = norms(wj)

        def loss(t, xx, yv, yg, ww, A):
            val, grd = bank(unravel(t), xx); grd = 0.1 * grd
            if vp:
                val = val @ A; grd = grd @ A
            ev, eg, ed = terms(val, grd, yv, yg, ww)
            return jnp.sum(impj * (ev / nv + eg / ng_ + lam * ed / ng_)), (ev / nv, eg / ng_, ed / ng_)
        vg = jax.jit(jax.value_and_grad(loss, has_aux=True))

        @jax.jit
        def gram(t, xx, yv, yg, ww):
            val, grd = bank(unravel(t), xx); grd = 0.1 * grd
            dv = grd[:, 0, 0, :] + grd[:, 1, 1, :]; ty = yg[:, 0, 0, :] + yg[:, 1, 1, :]
            n = val.shape[0]
            V2 = val.reshape(n * 2, r); G2 = grd.reshape(n * 4, r)
            return (V2.T @ (jnp.repeat(ww, 2)[:, None] * V2), V2.T @ (jnp.repeat(ww, 2)[:, None] * yv.reshape(n * 2, r)),
                    G2.T @ (jnp.repeat(ww, 4)[:, None] * G2), G2.T @ (jnp.repeat(ww, 4)[:, None] * yg.reshape(n * 4, r)),
                    dv.T @ (ww[:, None] * dv), dv.T @ (ww[:, None] * ty))

        def solveA(t):
            acc = None
            for i in range(0, N + pad, CH):
                e = slice(i, i + CH)
                g = gram(t, xj[e], Vj[e], Gj[e], wj[e])
                acc = g if acc is None else tuple(p + q_ for p, q_ in zip(acc, g))
            Mv, Rv, Mg, Rg, Md, Rd = (np.asarray(z) for z in acc)
            nvn, ngn = np.asarray(nv), np.asarray(ng_)
            A = np.zeros((r, r))
            for i in range(r):
                lhs = Mv / nvn[i] + (Mg + lam * Md) / ngn[i]
                rhs = Rv[:, i] / nvn[i] + (Rg[:, i] + lam * Rd[:, i]) / ngn[i]
                A[:, i] = np.linalg.solve(lhs + 1e-12 * np.trace(lhs) / r * np.eye(r), rhs)
            return A

        def full(t):
            t = jnp.asarray(t)
            A = jnp.asarray(solveA(t)) if vp else jnp.eye(r)
            f = 0.0; g = jnp.zeros_like(t); aux = None
            for i in range(0, N + pad, CH):
                e = slice(i, i + CH)
                (fi, ai), gi = vg(t, xj[e], Vj[e], Gj[e], wj[e], A)
                f = f + fi; g = g + gi
                aux = ai if aux is None else tuple(p + q_ for p, q_ in zip(aux, ai))
            f = float(f); g = np.asarray(g)
            if not np.isfinite(f) or not np.isfinite(g).all():
                raise FloatingPointError('nonfinite objective')
            return f, g, [np.asarray(z) for z in aux], np.asarray(A)
        return vg, full

    class Deadline(Exception):
        pass

    def lbfgs(full, theta, seconds, iters, label):
        t0 = time.time(); best = {'f': np.inf, 't': np.asarray(theta).copy(), 'n': 0}

        def fun(t):
            f, g, _, _ = full(t)
            best['n'] += 1
            if f < best['f']:
                best.update(f=f, t=t.copy())
            if best['n'] % 100 == 0:
                record['history'].append({'phase': label, 'evaluations': best['n'], 'objective': best['f'], 'seconds': time.time() - t0})
                print(label, best['n'], best['f'], round(time.time() - t0), flush=True)
            if time.time() - t0 > seconds:
                raise Deadline()
            return f, g
        try:
            res = minimize(fun, np.asarray(theta, dtype=np.float64), jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'maxfun': 10 ** 7, 'maxcor': 30, 'maxls': 30, 'gtol': 1e-12, 'ftol': 0.0})
            stop = {'iterations': int(res.nit), 'message': str(res.message), 'converged_or_cap': True}
        except Deadline:
            stop = {'iterations': None, 'message': f'deadline stop ({seconds:.0f} s), not convergence', 'converged_or_cap': False}
        f, _, aux, A = full(best['t'])
        record[label] = dict(stop, evaluations=best['n'], objective=f, seconds=time.time() - t0,
                             value_rel2=aux[0].tolist(), gradient_rel2=aux[1].tolist(), divergence_rel2=aux[2].tolist())
        print(label, json.dumps({k: record[label][k] for k in ('message', 'evaluations', 'objective', 'seconds')}), flush=True)
        return best['t'], A

    # ---- stage 1: Adam then L-BFGS ---------------------------------------
    w1 = weights(a.box_weight)
    vg1, full1 = make(w1, a.div_weight, False)
    eyeA = jnp.eye(r)

    @jax.jit
    def adam_step(t, m, v, k, key, rate):
        idx = jax.random.randint(key, (a.batch,), 0, N)
        (f, _), g = vg1(t, xj[idx], Vj[idx], Gj[idx], w1[idx] * (N / a.batch), eyeA)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        t = t - rate * (m / (1 - 0.9 ** k)) / (jnp.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
        return t, m, v, f
    theta = theta0; m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    key = jax.random.PRNGKey(a.seed); t0 = time.time()
    for k in range(1, a.adam_steps + 1):
        key, sub = jax.random.split(key)
        rate = a.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * k / a.adam_steps)))
        theta, m, v, f = adam_step(theta, m, v, float(k), sub, rate)
        if k % 2000 == 0 or k == a.adam_steps:
            ff = full1(theta)[0]
            record['history'].append({'phase': 'adam', 'step': k, 'objective': ff, 'seconds': time.time() - t0})
            print('adam', k, ff, round(time.time() - t0), flush=True)
    theta, A = lbfgs(full1, theta, a.lbfgs_seconds, a.lbfgs_iters, 'stage1')
    np.savez(out / 'stage1_params.npz', theta=theta)

    # ---- stage 2: optional fine-tune --------------------------------------
    if a.ft != 'none':
        box2 = a.box_weight if a.ft_box_weight is None else a.ft_box_weight
        lam2 = a.div_weight if a.ft_div_weight is None else a.ft_div_weight
        _, full2 = make(weights(box2), lam2, a.ft == 'vp')
        theta, A = lbfgs(full2, theta, a.ft_seconds, a.ft_iters, 'stage2')
    # unweighted, penalty-free regression report of the final functions (after the PCA-ordering map A)
    _, fullr = make(weights(1.0), 0.0, False)
    Aj = jnp.asarray(A)
    pfin = unravel(jnp.asarray(theta))
    ev = jax.jit(lambda xx: tuple(z @ Aj for z in bank(pfin, xx)))

    def evaluate(pts):
        vs, gs = [], []
        for i in range(0, len(pts), CH):
            blk = pts[i:i + CH]; n = len(blk)
            if n < CH:
                blk = np.concatenate([blk, np.repeat(pts[:1], CH - n, 0)])
            vv, gg = ev(jnp.asarray(blk)); vs.append(np.asarray(vv)[:n]); gs.append(np.asarray(gg)[:n])
        return np.concatenate(vs), np.concatenate(gs)
    V, G = evaluate(x)
    Vt = b['V'][..., cols]; Gt = b['G'][..., cols]
    dv = G[:, 0, 0, :] + G[:, 1, 1, :]
    record['final'] = {
        'value_rel': np.sqrt(np.einsum('n,ncr->r', w, (V - Vt) ** 2) / np.einsum('n,ncr->r', w, Vt ** 2)).tolist(),
        'gradient_rel': np.sqrt(np.einsum('n,ncdr->r', w, (G - Gt) ** 2) / np.einsum('n,ncdr->r', w, Gt ** 2)).tolist(),
        'divergence_over_gradient': np.sqrt(np.einsum('n,nr->r', w, dv ** 2) / np.einsum('n,ncdr->r', w, G ** 2)).tolist(),
        'ordering_map_condition': float(np.linalg.cond(A))}
    bsave = {}
    for lab in ('inlet', 'outlet', 'obstacle'):
        bv, bg = evaluate(q[f'{lab}_points']); bsave[f'{lab}_V'] = bv; bsave[f'{lab}_G'] = bg
    record['final']['max_abs_on_inlet'] = float(np.abs(bsave['inlet_V']).max())
    record['final']['max_abs_on_obstacle'] = float(np.abs(bsave['obstacle_V']).max())
    pdig = hashlib.sha256(np.ascontiguousarray(np.asarray(theta)).tobytes() + np.ascontiguousarray(A).tobytes()).hexdigest()
    np.savez(out / 'trained_bank_eval.npz', V=V, G=G, params_digest=np.array(pdig), **bsave)
    np.savez(out / 'trained_bank_params.npz', theta=np.asarray(theta), A=A, rank=r)
    record['outputs'] = {p.name: sha(p) for p in out.glob('*.npz')}
    record['runtime'] = {'jax': jax.__version__, 'backend': jax.default_backend(), 'devices': [d.device_kind for d in jax.devices()]}
    (out / 'TRAINING.json').write_text(json.dumps(record, indent=2, default=float) + '\n')
    print(json.dumps({k: record[k] for k in ('stage1', 'stage2', 'final') if k in record}, default=float)[:3000])


if __name__ == '__main__':
    main()
