"""jcp-heads matched head training (DESIGN 2). The incumbent recipe is `sep_hfit_run.py` arm `mid` (copied unchanged as
`base/sep_hfit_run.py`; the fit routine `sep_hfit.py` is copied unchanged). This driver is that file's `main` with three
changes and nothing else:

  * the arm list is a JSON plan of (name, k, seed, steps) instead of the ARM_SPECS table; every arm uses the `mid` spec
    (hidden 512, layers 2, every other `hf.fit` argument at its default) -- only K, SEED0 and (for the under-trained
    control) the step count vary;
  * every arm is emitted as its own checkpoint (bank parameters of the incumbent pkl, new head, codes, cfg with the
    seed, K, pick and sha256 of the data file), not only one EMIT arm;
  * an `identity` arm evaluates the incumbent itself (its own head and its own codes Z_tr, aligned with the data file's
    pick -- asserted) through the same `evaluate`, for the driver-identity gate (DESIGN 2.3).

The objective, optimiser, schedule, batch, normalisation, initialisation, RNG keys (PRNGKey(SEED0 + 11) for the fit,
PRNGKey(SEED0 + 7) for the oracle encoder) and the diagnostics are the copied text.

    python htrain.py --config configs/<attempt>.json --out <dir>     (plan: {"npz", "ckpt", "arms": [{name, k, seed, steps?, identity?}]})
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import sep_hfit as hf  # noqa: E402

F64 = jnp.float64
log = hf.log
MID = dict(hidden=512, layers=2)                         # sep_hfit_run.ARM_SPECS['mid']
NPZ_SHA256 = '4a7715659fe24b640c68e3c5f8b44e804383d4705aa62efa7dd6a3aaef73ebbb'
CKPT_SHA256 = '18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'
BATCH, LR, ORACLE_ITERS, ENC_STEPS = 4096, 1e-3, 150, 12000   # dn256b stage C (run_dn256b.sbatch:58-62)
CODEDIAG_N, CODEDIAG_ITERS = 2048, 60


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '--plan', dest='plan', required=True)
    ap.add_argument('--npz')
    ap.add_argument('--ckpt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    a.npz = a.npz or plan['npz']                 # absolute cluster path of the frozen training file (sha256-checked)
    a.ckpt = a.ckpt or plan['ckpt']              # the incumbent checkpoint, staged with the job
    a.allow_cpu = a.allow_cpu or bool(plan.get('allow_cpu_smoke'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]
    if not a.allow_cpu:
        assert jax.default_backend() == 'gpu', jax.default_backend()
        assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} plan={[x['name'] for x in plan['arms']]}")
    sha_npz, sha_ck = sha_file(a.npz), sha_file(a.ckpt)
    assert sha_npz == NPZ_SHA256, ('coefficient file differs from the incumbent training file', sha_npz)
    assert sha_ck == CKPT_SHA256, ('checkpoint differs from the incumbent', sha_ck)
    t_all = time.time()
    d = np.load(a.npz)
    with open(a.ckpt, 'rb') as f:
        ck = pickle.load(f)
    # DESIGN 2.2: every dev6 / val32 descriptor is absent from the training descriptors of all 4,608 trajectories
    import p1core
    Qc = p1core.load_qcore()
    mu = np.unique(np.asarray(d['mu_tr']), axis=0)
    disjoint = {}
    for coh in ('dev6', 'val32'):
        ph = np.asarray(Qc.cohort(coh))
        dmin = float(min(np.min(np.max(np.abs(mu - x[None]), axis=1)) for x in ph))
        assert dmin > 1e-9, f'{coh} descriptor present in the training data'
        disjoint[coh] = dict(min_maxabs_distance=dmin, n=len(ph))
    disjoint['n_training_descriptors'] = int(len(mu))
    assert len(mu) == 4608 and len(np.unique(np.asarray(d['traj_tr']))) == 4608, 'expected 4,608 training trajectories'
    pick_all = np.asarray(d['pick'])
    assert np.array_equal(np.asarray(ck['cfg']['hfit_pick']), pick_all), 'incumbent codes not aligned with the data pick'
    p_ck = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Z_inc = jnp.asarray(ck['Z_tr'], dtype=F64)
    Gram = jnp.asarray(d['Gram'], dtype=F64)
    R = Gram.shape[0]
    eps = 1e-12 * jnp.trace(Gram) / R
    L = jnp.linalg.cholesky(Gram + eps * jnp.eye(R, dtype=F64))
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(R, dtype=F64), lower=True)
    A_tr = jnp.asarray(d['C_tr'], dtype=F64) @ L
    A_te = jnp.asarray(d['C_te'], dtype=F64) @ L
    un2_tr = jnp.asarray(d['un2_tr']); fl2_tr = jnp.asarray(d['fl2_tr'])
    un2_te = jnp.asarray(d['un2_te']); fl2_te = jnp.asarray(d['fl2_te'])
    t_te = np.asarray(d['t_te'])
    fl_tr = np.asarray(jnp.sqrt(fl2_tr / un2_tr))
    fl_te = np.asarray(jnp.sqrt(fl2_te / un2_te))
    S = A_tr.shape[0]

    # whitening round-trip gate (copied text): to_h(to_q(h)) must return h to ~1e-12
    qp_ck = hf.to_q(dict(h=p_ck['h'], h_lin=p_ck['h_lin']), L)
    hp_rt = hf.to_h(qp_ck, Linv)
    rt = float(max(jnp.max(jnp.abs(hp_rt['h'][-1][0] - p_ck['h'][-1][0])) / jnp.max(jnp.abs(p_ck['h'][-1][0])),
                   jnp.max(jnp.abs(hp_rt['h_lin'] - p_ck['h_lin'])) / jnp.max(jnp.abs(p_ck['h_lin']))))
    zt = jnp.asarray(Z_inc[:8])
    hv = hf.head_apply(dict(h=p_ck['h'], h_lin=p_ck['h_lin']), zt)
    qv = hf.head_apply(qp_ck, zt)
    wt = float(jnp.max(jnp.abs(qv - hv @ L)) / jnp.max(jnp.abs(qv)))
    log(f"  whitening round-trip {rt:.3e}   q == L^T h deviation {wt:.3e}")
    assert rt < 1e-10 and wt < 1e-10, 'whitening reparameterisation is not exact'

    report = dict(config=dict(npz=os.path.basename(a.npz), npz_sha256=sha_npz, ckpt=os.path.basename(a.ckpt),
                              ckpt_sha256=sha_ck, plan=plan, batch=BATCH, lr=LR, oracle_iters=ORACLE_ITERS,
                              enc_steps=ENC_STEPS, R=int(R), S=int(S), n_test_states=int(A_te.shape[0]), x64=True,
                              matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), backend=dev.platform,
                              gpu=getattr(dev, 'device_kind', str(dev)), slurm_job=os.environ.get('SLURM_JOB_ID'),
                              node=os.environ.get('SLURMD_NODENAME', 'local'),
                              source_commit=os.environ.get('SOURCE_COMMIT')),
                  gates=dict(whitening_round_trip=rt, q_equals_LT_h=wt, disjointness=disjoint),
                  floors=dict(train_span_floor_mean=float(fl_tr.mean()), train_span_floor_max=float(fl_tr.max()),
                              test_span_floor_mean=float(fl_te.mean()), test_span_floor_max=float(fl_te.max())),
                  arms={}, complete=False)

    def save():
        json.dump(report, open(out / 'result.json', 'w'), indent=1, default=float)

    T = int(t_te.max()) + 1

    def train_encoder(key, A, Z, steps):          # copied text (sep_hfit_run.main.train_encoder)
        k_ = Z.shape[1]
        ep = hf.init_head(key, A.shape[1], k_, hidden=256, layers=2, lin_scale=0.0)
        sch = optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(500, steps // 10 + 1), steps, 1e-5)
        o = optax.adam(sch)
        st = o.init(ep)
        zs = float(jnp.std(Z)) + 1e-30

        def ls(p, Ab, Zb):
            return jnp.mean((hf.head_apply(p, Ab) - Zb) ** 2) / zs ** 2

        @jax.jit
        def stp(p, st, kk, A_, Z_):
            nb = min(4096, A_.shape[0])
            i = jax.random.choice(kk, A_.shape[0], shape=(nb,), replace=False)
            v, g = jax.value_and_grad(ls)(p, A_[i], Z_[i])
            u, st = o.update(g, st)
            return optax.apply_updates(p, u), st, v
        v = jnp.inf
        for i in range(steps):
            key, kk = jax.random.split(key)
            ep, st, v = stp(ep, st, kk, A, Z)
        return ep, float(v)

    @jax.jit
    def nn_idx(Ate, Atr):
        d2 = (jnp.sum(Ate * Ate, 1)[:, None] - 2.0 * (Ate @ Atr.T) + jnp.sum(Atr * Atr, 1)[None, :])
        return jnp.argmin(d2, axis=1)
    nnj = np.asarray(nn_idx(A_te, A_tr))

    def evaluate(spec, qp, Z, tinfo, seed0):       # copied text (sep_hfit_run.main.evaluate), SEED0 passed in
        ent = dict(spec=spec, train=tinfo, k=int(Z.shape[1]))
        rec = np.asarray(hf.batched_rel(qp, Z, A_tr, fl2_tr, un2_tr))
        ent['recon_train'] = dict(mean=float(rec.mean()), max=float(rec.max()))
        key = jax.random.PRNGKey(seed0 + 7)
        ep, ev = train_encoder(key, A_tr, Z, ENC_STEPS)
        ent['encoder_final_loss'] = ev
        zbar = jnp.mean(Z, axis=0)
        z_enc = hf.head_apply(ep, A_te)
        inits = jnp.stack([jnp.tile(zbar[None], (A_te.shape[0], 1)), z_enc], axis=1)
        o_rel, _ = hf.oracle(qp, A_te, fl2_te, un2_te, inits, ORACLE_ITERS)
        ent['oracle_test'] = dict(mean=float(o_rel.mean()), max=float(o_rel.max()), t0=float(o_rel[t_te == 0].mean()),
                                  early=float(o_rel[t_te <= 5].mean()), late=float(o_rel[t_te > 5].mean()),
                                  per_time=[float(o_rel[t_te == t].mean()) for t in range(T)],
                                  note='representation oracle, DIAGNOSTIC (uses pipeline-test truth); inits = mean '
                                       'training code + training-only encoder')
        ent['oracle_over_span_floor'] = float(o_rel.mean() / fl_te.mean())
        inits_nn = jnp.concatenate([inits, Z[jnp.asarray(nnj)][:, None]], axis=1)
        onn, _ = hf.oracle(qp, A_te, fl2_te, un2_te, inits_nn, ORACLE_ITERS)
        ent['oracle_test_nn'] = dict(mean=float(onn.mean()), max=float(onn.max()),
                                     note='UPPER BOUND: adds a nearest-training-state init selected USING the test target')
        sub = np.linspace(0, A_tr.shape[0] - 1, min(CODEDIAG_N, A_tr.shape[0])).astype(int)
        sj = jnp.asarray(sub)
        o_tr, _ = hf.oracle(qp, A_tr[sj], fl2_tr[sj], un2_tr[sj], Z[sj][:, None, :], CODEDIAG_ITERS, chunk=1024)
        rec_sub = float(rec[sub].mean())
        ent['oracle_train_from_codes'] = dict(mean=float(o_tr.mean()), max=float(o_tr.max()), n=int(len(sub)),
                                              recon_on_same_subset=rec_sub,
                                              gain_vs_recon=float(rec_sub / max(o_tr.mean(), 1e-300)))
        log(f"  ARM k={Z.shape[1]}: recon {rec.mean():.4e}  oracle {o_rel.mean():.4e}  "
            f"oracle/floor {ent['oracle_over_span_floor']:.1f}x  oracle_nn {onn.mean():.4e}")
        return ent

    for arm in plan['arms']:
        name, k, seed0 = arm['name'], int(arm['k']), int(arm['seed'])
        steps = int(arm.get('steps', 200000))
        t0 = time.time()
        if arm.get('identity'):
            assert k == Z_inc.shape[1]
            qp, Z, tinfo = qp_ck, Z_inc, dict(steps=0, seconds=0.0, note='incumbent as-is (own head, own codes)')
        else:
            qp, Z, tinfo = hf.fit(jax.random.PRNGKey(seed0 + 11), A_tr, un2_tr, fl2_tr, k, R, steps=steps, lr=LR,
                                  batch=BATCH, tag=name, **MID)
        ent = evaluate(dict(MID, k=k, steps=steps, seed0=seed0), qp, Z, tinfo, seed0)
        ent['seconds'] = time.time() - t0
        if not arm.get('identity'):
            hp = hf.to_h(qp, Linv)
            newp = {kk: vv for kk, vv in ck['params'].items()}
            newp['h'] = [(np.asarray(w), np.asarray(b)) for w, b in hp['h']]
            newp['h_lin'] = np.asarray(hp['h_lin'])
            newp.pop('hB', None)
            cfg = dict(ck.get('cfg', {}))
            cfg.update(k=int(k), hfit_arm='mid', hfit_spec=dict(MID), hfit_source_ckpt=os.path.basename(a.ckpt),
                       hfit_source_ckpt_sha256=sha_ck, hfit_npz_sha256=sha_npz, jcp_heads_name=name,
                       jcp_heads_seed0=seed0, jcp_heads_steps=steps, hfit_pick=pick_all.tolist())
            path = out / f'head_{name}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(dict(params=newp, Z_tr=np.asarray(Z), cfg=cfg), f)
            ent['emitted'] = path.name
            ent['emitted_sha256'] = sha_file(path)
            # compact archive (DESIGN A5): head weights + candidate codes + trust and code statistics from ALL codes
            Zn = np.asarray(Z)
            zdev = np.linalg.norm(Zn - Zn.mean(0), axis=1)
            cpath = out / f'head_{name}.npz'
            hw = {f'h{i}_w': np.asarray(w) for i, (w, b) in enumerate(newp['h'])}
            hw.update({f'h{i}_b': np.asarray(b) for i, (w, b) in enumerate(newp['h'])})
            np.savez(cpath, h_lin=np.asarray(newp['h_lin']), n_layers=len(newp['h']), Zsub=Zn[::max(1, len(Zn) // 8192)],
                     trust=.01 * float(zdev.max()), code_std=float(np.std(Zn)), code_max_dev=float(zdev.max()),
                     n_codes=len(Zn), k=int(k), seed0=seed0, steps=steps, name=name, source_ckpt_sha256=sha_ck,
                     npz_sha256=sha_npz, pkl_sha256=ent['emitted_sha256'], **hw)
            ent['compact'] = cpath.name
            ent['compact_sha256'] = sha_file(cpath)
            # the compact archive defines the same decoder head as the emitted checkpoint (reloaded, 8 random codes)
            zc = np.load(cpath)
            hc = dict(h=[(jnp.asarray(zc[f'h{i}_w']), jnp.asarray(zc[f'h{i}_b'])) for i in range(int(zc['n_layers']))],
                      h_lin=jnp.asarray(zc['h_lin']))
            zt_ = jax.random.normal(jax.random.PRNGKey(5), (8, int(k))) * .1
            a_, b_ = hf.head_apply(hc, zt_), hf.head_apply(dict(h=hp['h'], h_lin=hp['h_lin']), zt_)
            ent['compact_vs_emitted'] = float(jnp.max(jnp.abs(a_ - b_)) / jnp.max(jnp.abs(b_)))
            assert ent['compact_vs_emitted'] == 0., ent['compact_vs_emitted']
            # bank parameters untouched (identity of every non-head leaf)
            for kk in ck['params']:
                if kk not in ('h', 'h_lin', 'hB'):
                    la, lb = jax.tree_util.tree_leaves(ck['params'][kk]), jax.tree_util.tree_leaves(newp[kk])
                    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)), kk
        report['arms'][name] = ent
        save()
    report['total_seconds'] = time.time() - t_all
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    log(f"HTRAIN done in {report['total_seconds']:.0f}s")


if __name__ == '__main__':
    main()
