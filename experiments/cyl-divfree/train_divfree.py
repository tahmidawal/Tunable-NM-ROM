"""Train the stream-function bank on the run-09 objective (POD value+gradient, trace, traction).

Objective (identical definition and frozen normalisations to the run-09 lineage):
  J = sum_r m_r (e^v_r/nv_r + e^g_r/ng_r + e^tr_r/ng_r) + 1.0 * traction
with m_r = sqrt(s_r/s_1)/sum, nv/ng the TRAIN teacher value/gradient norms, e^tr the
supervised divergence-trace error (a constant here: the bank is exactly solenoidal), and
traction the sigma^2-weighted cylinder traction error over the global TRAIN traction energy.
Stage 1: Adam on minibatches (cosine schedule as accuracy_v2/train.py). Stage 2:
full-batch L-BFGS (history 10, Armijo backtracking 20 halvings) from the best full-measured
Adam iterate, until gradient inf-norm <= 1e-10, line-search failure, or the deadline.
Selection: lowest full TRAIN objective. No development or validation data is read.

Usage (cluster): train_divfree.py --prepared DIR --release FILE --run09-bank FILE --out DIR
                 --deadline-seconds S [--resume THETA.npz] [--adam-steps N]
Synthetic local smoke: train_divfree.py --synthetic N --out DIR --deadline-seconds S
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent/'ns-obstacle-rom')]
import numpy as np
import divfree_bank as B
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')

BLOCK = 16384


def digest(a):
    a = np.ascontiguousarray(a); h = hashlib.sha256(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes()); return h.hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_actual(prepared, release, out):
    from accuracy_v2.teacher import load
    from accuracy_v2.boundary_teacher import load as load_boundary
    x, w, v, g, metadata, identity = load(prepared, release)
    boundary, record = load_boundary(prepared, metadata, identity, Path(out)/'boundary-teacher')
    return x, w, v, g, identity, boundary, float(record['normalization']['global_training_traction_energy'])


def synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.c_[rng.uniform(0, 2.2, 3*n), rng.uniform(0, .41, 3*n)]
    x = x[np.linalg.norm(x-np.array(B.CHANNEL.center), axis=1) > .05][:n]
    p = B.initialize(seed=7); p['layers'][-1] = (rng.normal(size=p['layers'][-1][0].shape)*.05, np.zeros(4)); p['c'] = rng.normal(size=4)*.1
    v, g = B._fields_jit(jax.tree_util.tree_map(jnp.asarray, p), jnp.asarray(x/B.D))
    th = np.linspace(0, 2*np.pi, 805, endpoint=False); bx = np.c_[.2+.05*np.cos(th), .2+.05*np.sin(th)]
    _, bg = B._fields_jit(jax.tree_util.tree_map(jnp.asarray, p), jnp.asarray(bx/B.D))
    boundary = {'points_physical': bx, 'weights_dimensionless': np.full(805, np.pi/805), 'normals_body': np.c_[np.cos(th), np.sin(th)],
                'teacher_gradient_dimensionless': np.asarray(bg), 'singular_values': np.array([4., 2., 1., .5])}
    identity = {'singular_values': [4., 2., 1., .5], 'synthetic': True}
    return x, np.full(len(x), 89.4/len(x)), np.asarray(v), np.asarray(g), identity, boundary, 1.0


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--prepared', type=Path); a.add_argument('--release', type=Path); a.add_argument('--run09-bank', type=Path)
    a.add_argument('--synthetic', type=int, default=0); a.add_argument('--out', type=Path, required=True)
    a.add_argument('--deadline-seconds', type=float, required=True); a.add_argument('--resume', type=Path)
    a.add_argument('--adam-steps', type=int, default=6000); a.add_argument('--adam-batch', type=int, default=8192)
    a.add_argument('--adam-lr', type=float, default=1e-3); a.add_argument('--full-check-every', type=int, default=500)
    a.add_argument('--lbfgs-max', type=int, default=100000); a.add_argument('--seed', type=int, default=3112026)
    a.add_argument('--width', type=int, default=160); a.add_argument('--depth', type=int, default=3)
    args = a.parse_args()
    t_start = float(os.environ.get('ROM_ALLOCATION_START_UNIX_SECONDS', time.time()))
    deadline = t_start + args.deadline_seconds
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    runtime = {'jax': jax.__version__, 'backend': jax.default_backend(), 'devices': [d.device_kind for d in jax.devices()],
               'x64': bool(jax.config.x64_enabled), 'matmul_precision': str(jax.config.jax_default_matmul_precision),
               'job_id': os.environ.get('SLURM_JOB_ID'), 'argv': sys.argv}
    (out/'RUNTIME.json').write_text(json.dumps(runtime, indent=1)+'\n')
    if not args.synthetic and runtime['backend'] != 'gpu':
        raise RuntimeError('GPU required')

    if args.synthetic:
        x, w, tv, tg, identity, bd, tden = synthetic(args.synthetic)
    else:
        x, w, tv, tg, identity, bd, tden = load_actual(args.prepared, args.release, out)
    n = len(x)
    sigma = np.asarray(identity['singular_values'], np.float64)[:4]
    nv = np.einsum('n,ncr,ncr->r', w, tv, tv); ng = np.einsum('n,ncdr,ncdr->r', w, tg, tg)
    m = np.sqrt(sigma/sigma[0]); m /= m.sum()
    check = {}
    if args.run09_bank:
        from accuracy_v2.residual_head_bank import ResidualHeadBank
        pm = ResidualHeadBank.load(args.run09_bank).parent.metadata
        for key, val in (('value_norms_squared', nv), ('gradient_norms_squared', ng), ('mode_weights', m)):
            ref = np.asarray(pm[key]); check[key] = float(np.max(np.abs(val-ref)/np.abs(ref)))
            if check[key] > 1e-12: raise ValueError(f'normalisation differs from run 09: {key}')
        nv, ng, m = (np.asarray(pm[k], np.float64) for k in ('value_norms_squared', 'gradient_norms_squared', 'mode_weights'))
        check['run09_bank_sha256'] = sha(args.run09_bank)
    teacher_trace = tg[:, 0, 0, :] + tg[:, 1, 1, :]
    trace_const = float(np.sum(m*np.einsum('n,nr,nr->r', w, teacher_trace, teacher_trace)/ng))
    config = {'schema': 'cyl-divfree-training-v1', 'n_points': n, 'teacher_identity': identity,
              'arrays': {'points': digest(x), 'weights': digest(w), 'values': digest(tv), 'gradients': digest(tg)},
              'value_norms_squared': nv.tolist(), 'gradient_norms_squared': ng.tolist(), 'mode_weights': m.tolist(),
              'normalisation_check_vs_run09': check, 'traction_denominator': tden, 'traction_weight': 1.0,
              'supervised_divergence_weight': 1.0, 'divergence_trace_term_constant': trace_const,
              'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              'source_sha256': {f: sha(here/f) for f in ('divfree_bank.py', 'train_divfree.py')},
              'selection': 'minimum full TRAIN objective; no development/validation data read'}
    (out/'CONFIG.json').write_text(json.dumps(config, indent=1)+'\n')

    # device arrays, padded with zero-weight points (safe interior point)
    nb = -(-n//BLOCK); pad = nb*BLOCK - n
    filler = np.array([1.5, .3])
    def padded(arr, fill):
        return np.concatenate([arr, np.broadcast_to(fill, (pad,)+arr.shape[1:])]).reshape((nb, BLOCK)+arr.shape[1:])
    Xd = jnp.asarray(padded(x/B.D, filler)); Wd = jnp.asarray(padded(w, 0.)); Vd = jnp.asarray(padded(tv, 0.)); Gd = jnp.asarray(padded(tg, 0.))
    bX = jnp.asarray(bd['points_physical']/B.D); bW = jnp.asarray(bd['weights_dimensionless']); bN = jnp.asarray(bd['normals_body'])
    bG = jnp.asarray(bd['teacher_gradient_dimensionless']); bS = jnp.asarray(bd['singular_values'])
    nvj, ngj, mj = jnp.asarray(nv), jnp.asarray(ng), jnp.asarray(m)
    p0 = B.initialize(args.width, args.depth, args.seed); theta, shapes = B.flatten(p0)
    if args.resume:
        with np.load(args.resume) as r:
            theta = r['theta'].copy()
        config['resume_theta_sha256'] = digest(theta); (out/'CONFIG.json').write_text(json.dumps(config, indent=1)+'\n')

    data = (Xd, Wd, Vd, Gd, bX, bW, bN, bG, bS, nvj, ngj, mj)  # explicit jit arguments (no captured constants)

    def vol_terms(t, X, W, V, G, nvj, ngj, mj):
        vel, grad = B.fields(B.unflatten(t, shapes), X)
        ev = jnp.einsum('n,ncr,ncr->r', W, vel-V, vel-V)
        eg = jnp.einsum('n,ncdr,ncdr->r', W, grad-G, grad-G)
        tr = (grad[:, 0, 0, :]+grad[:, 1, 1, :]) - (G[:, 0, 0, :]+G[:, 1, 1, :])
        et = jnp.einsum('n,nr,nr->r', W, tr, tr)
        return jnp.sum(mj*(ev/nvj + eg/ngj + et/ngj)), (ev/nvj, eg/ngj, et/ngj)

    def traction(t, bX, bW, bN, bG, bS):
        _, grad = B.fields(B.unflatten(t, shapes), bX)
        def tr(g): return jnp.einsum('nijr,nj->nir', g+jnp.swapaxes(g, 1, 2), bN)
        err = tr(grad) - tr(bG); mode = jnp.einsum('n,nir,nir->r', bW, err, err)
        return jnp.sum(bS**2*mode)/tden, mode

    vol_vg = jax.value_and_grad(vol_terms, has_aux=True)
    tra_vg = jax.value_and_grad(traction, has_aux=True)

    @jax.jit
    def full(t, data):
        Xd, Wd, Vd, Gd, bX, bW, bN, bG, bS, nvj, ngj, mj = data
        def body(carry, blk):
            (l, parts), g = vol_vg(t, *blk, nvj, ngj, mj)
            return (carry[0]+l, carry[1]+g, tuple(c+p for c, p in zip(carry[2], parts))), None
        z4 = jnp.zeros(4)
        (l, g, parts), _ = jax.lax.scan(body, (0., jnp.zeros_like(t), (z4, z4, z4)), (Xd, Wd, Vd, Gd))
        (lt, mode), gt = tra_vg(t, bX, bW, bN, bG, bS)
        return l+lt, g+gt, {'value': parts[0], 'gradient': parts[1], 'trace': parts[2], 'traction': lt}

    @jax.jit
    def mini(t, idx, data):
        Xd, Wd, Vd, Gd, bX, bW, bN, bG, bS, nvj, ngj, mj = data
        X = Xd.reshape(-1, 2)[idx]; W = Wd.reshape(-1)[idx]*n/idx.shape[0]
        V = Vd.reshape(-1, 2, 4)[idx]; G = Gd.reshape(-1, 2, 2, 4)[idx]
        (l, _), g = vol_vg(t, X, W, V, G, nvj, ngj, mj); (lt, _), gt = tra_vg(t, bX, bW, bN, bG, bS)
        return l+lt, g+gt

    evals = 0; best = {'objective': np.inf}; log = (out/'LOG.jsonl').open('a'); started = time.monotonic()

    def measure(t):
        nonlocal evals
        f, g, parts = full(jnp.asarray(t), data); evals += 1
        f = float(f); g = np.asarray(g)
        if not np.isfinite(f) or not np.isfinite(g).all(): raise FloatingPointError('nonfinite objective')
        return f, g, {k: np.asarray(v).tolist() for k, v in parts.items()}

    def record(phase, step, t, f, g, parts=None, extra=None):
        nonlocal best
        row = {'phase': phase, 'step': step, 'objective': f, 'gradient_inf': float(np.max(np.abs(g))),
               'elapsed': time.monotonic()-started, 'full_evaluations': evals}
        if parts is not None: row['parts'] = parts
        if extra: row.update(extra)
        log.write(json.dumps(row)+'\n'); log.flush()
        if f < best['objective']:
            best = {'objective': f, 'phase': phase, 'step': step, 'theta': t.copy(), 'gradient_inf': row['gradient_inf']}

    def save_ckpt(tag, t):
        np.savez_compressed(out/f'checkpoint-{tag}.npz', theta=t, shapes=np.array(json.dumps([list(s) for s in shapes])))

    def time_left():
        return deadline - time.time()

    status = 'started'
    tic = time.monotonic(); f, g, parts = measure(theta); t_full = time.monotonic()-tic
    tic = time.monotonic(); measure(theta); t_full2 = time.monotonic()-tic
    record('initial', 0, theta, f, g, parts, {'first_eval_seconds_with_compile': t_full, 'eval_seconds': t_full2})
    print(f'initial objective {f:.6e}; full eval {t_full2:.3f}s (first {t_full:.1f}s)', flush=True)
    try:
        # ---- Adam ----
        mom = np.zeros_like(theta); vel = np.zeros_like(theta); rng = np.random.default_rng(args.seed)
        steps = 0 if args.resume else args.adam_steps
        for k in range(1, steps+1):
            if time_left() < 900: status = 'deadline_in_adam'; break
            idx = jnp.asarray(rng.integers(0, n, size=args.adam_batch))
            _, gr = mini(jnp.asarray(theta), idx, data); gr = np.asarray(gr)
            if not np.isfinite(gr).all(): raise FloatingPointError('nonfinite Adam gradient')
            mom = .9*mom+.1*gr; vel = .999*vel+.001*gr*gr
            rate = args.adam_lr*(.1+.9*.5*(1+np.cos(np.pi*k/steps)))
            theta = theta - rate*(mom/(1-.9**k))/(np.sqrt(vel/(1-.999**k))+1e-8)
            if k % args.full_check_every == 0 or k == steps:
                f, g, parts = measure(theta); record('adam', k, theta, f, g, parts)
                print(f'adam {k} objective {f:.6e}', flush=True)
        save_ckpt('adam-best', best['theta'])
        # ---- L-BFGS (same two-loop/Armijo as accuracy_v2/train.py) ----
        theta = best['theta'].copy(); f, g, parts = measure(theta); record('lbfgs', 0, theta, f, g, parts)
        pairs = []; status = 'iteration_limit'
        for it in range(1, args.lbfgs_max+1):
            if time_left() < 600: status = 'deadline'; break
            if np.max(np.abs(g)) <= 1e-10: status = 'gradient_stationary'; break
            q = g.copy(); alphas = []
            for s, y, rho in reversed(pairs):
                al = rho*np.dot(s, q); alphas.append(al); q -= al*y
            if pairs: q *= np.dot(pairs[-1][0], pairs[-1][1])/np.dot(pairs[-1][1], pairs[-1][1])
            for (s, y, rho), al in zip(pairs, reversed(alphas)): q += s*(al-rho*np.dot(y, q))
            d = -q; slope = np.dot(g, d)
            if not np.isfinite(slope) or slope >= 0: d = -g; slope = -np.dot(g, g); pairs = []
            ok = False
            for att in range(20):
                step = 2.**(-att); trial = theta+step*d; nf, ngr, nparts = measure(trial)
                if nf <= f+1e-4*step*slope: ok = True; break
            if not ok: status = 'line_search_stopped'; break
            s = trial-theta; y = ngr-g; sy = float(np.dot(s, y))
            theta, f, g = trial, nf, ngr
            if sy > 1e-12*np.linalg.norm(s)*np.linalg.norm(y): pairs = (pairs+[(s, y, 1/sy)])[-10:]
            record('lbfgs', it, theta, f, g, nparts if it % 100 == 0 else None, {'step': step, 'attempts': att+1})
            if it % 100 == 0: print(f'lbfgs {it} objective {f:.6e} ginf {np.max(np.abs(g)):.3e}', flush=True)
            if it % 1000 == 0: save_ckpt('last', theta)
    except BaseException as exc:
        status = 'failed: '+repr(exc)
        (out/'FAILURE.json').write_text(json.dumps({'exception': repr(exc)})+'\n')
    finally:
        save_ckpt('last', theta)
        if np.isfinite(best['objective']):
            save_ckpt('best', best['theta'])
            bank = B.DivFreeBank(B.unflatten(best['theta'], shapes), {'trained': True, 'selected': {k: v for k, v in best.items() if k != 'theta'},
                                 'optimizer_status': status, 'config_sha256': sha(out/'CONFIG.json'), 'theta_sha256': digest(best['theta'])})
            bsha = bank.save(out/'divfree-bank.npz')
            # serialized replay: reload and re-measure the objective
            rb = B.DivFreeBank.load(out/'divfree-bank.npz'); rt, _ = B.flatten(rb.params); rf, rg, rparts = measure(rt)
        else:
            bsha = None; rf = None; rparts = None
        final = {'status': status, 'best': {k: v for k, v in best.items() if k != 'theta'}, 'bank_sha256': bsha,
                 'replay_objective': rf, 'replay_parts': rparts, 'full_evaluations': evals,
                 'elapsed_seconds': time.monotonic()-started, 'validation_seen': False, 'development_seen': False}
        (out/'FINAL.json').write_text(json.dumps(final, indent=1, default=float)+'\n')
        print(json.dumps(final, default=float), flush=True)


if __name__ == '__main__':
    main()
