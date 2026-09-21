"""Phase 1 of the bank-floor lane: representation floors of POD and learned banks.

One job per PDE. Arms run in order and `result.json` is rewritten after each, so
an interrupted job loses only the arm in flight. Protocol: DESIGN.md.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import sep_common as sc
import bf_core as K

F64 = jnp.float64


def host(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def assert_disjoint(a, b, label):
    a = np.asarray(a)
    b = np.asarray(b)
    d = np.abs(a[:, None, :] - b[None, :, :]).max(-1)
    assert d.min() > 1e-9, f'{label}: parameter overlap'


# ---------------------------------------------------------------- Burgers ----

def burgers_setup(cfg):
    import engines as e
    import common as C
    L, dt = cfg['intervals'], cfg['dt']
    q, _ = e.make_fom(L, dt, None, .25, dt)
    nstate = int(round(.25 / dt)) + 1

    def roll(physical, keep, tag):
        out = np.empty((len(physical) * len(keep), (L - 1) ** 2))
        worst, t0 = 0., time.perf_counter()
        kj = jnp.asarray(keep)
        for i, phys in enumerate(physical):
            f, it, rn = q(jnp.asarray(e.initial(L, phys)), float(phys[4]),
                          cfg['snapshot_ntol'], cfg['snapshot_ltol'])
            worst = max(worst, float(jnp.max(rn)))
            out[i * len(keep):(i + 1) * len(keep)] = np.asarray(
                f[kj][:, 1:-1, 1:-1].reshape(len(keep), -1))
            if (i + 1) % 128 == 0:
                print(f'  GEN[{tag}] {i + 1}/{len(physical)} worst_res {worst:.2e} '
                      f'[{time.perf_counter() - t0:.0f}s]', flush=True)
        assert np.isfinite(worst) and worst <= 1e-8, f'FOM residual {worst:.2e}'
        return out, dict(trajectories=len(physical), states=len(keep), worst_fom_residual=worst,
                         seconds=time.perf_counter() - t0)

    train_p = C.incumbent_draw()[:cfg['train_trajectories']]
    dev_p = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                            e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    hold_p = e.params_draw(cfg['holdout_seed'], cfg['holdout_trajectories'])
    assert_disjoint(train_p, dev_p, 'train/dev6')
    assert_disjoint(train_p, hold_p, 'train/hold64')
    assert_disjoint(dev_p, hold_p, 'dev6/hold64')
    keep_tr = np.arange(0, nstate, cfg['state_stride'])
    Utr, itr = roll(train_p, keep_tr, 'train')
    Udev, idev = roll(dev_p, np.arange(0, nstate, 10), 'dev6')
    Uho, iho = roll(hold_p, np.arange(nstate), 'hold64')
    sub = np.arange(cfg['incumbent_bank_trajectories'] * len(keep_tr))
    info = dict(train=dict(itr, parameters_sha256=K.sha_array(train_p)),
                dev6=dict(idev, parameters_sha256=K.sha_array(dev_p)),
                hold64=dict(iho, parameters_sha256=K.sha_array(hold_p)))
    del q
    jax.clear_caches()
    return Utr, sub, dict(dev6=Udev, hold64=Uho), info, 'dev6', {}


# ---------------------------------------------------------------- Poisson ----

def poisson_setup(cfg):
    import core as C
    import pbh_core as P
    n = cfg['intervals']
    base = C.source_params(cfg['incumbent_seed'], cfg['incumbent_sources'])
    fit, _ = P.fit_validation_split(cfg['incumbent_sources'], cfg['split_seed'],
                                    cfg['validation_fraction'])
    extra = C.source_params(cfg['extra_seed'], cfg['train_sources'] - len(fit))
    train_p = np.concatenate((base[fit], extra))
    coh = dict(dev12=np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                                     C.source_params(cfg['fresh_seed'], cfg['fresh_count']))),
               common256=C.source_params(cfg['common_seed'], 256),
               fresh256=C.source_params(cfg['holdout_seed'], 256))
    # the incumbent's validation sources are not training data here either
    for k, v in coh.items():
        assert_disjoint(train_p, v, f'train/{k}')
    assert_disjoint(coh['dev12'], coh['fresh256'], 'dev12/fresh256')
    t0 = time.perf_counter()
    Utr = np.asarray(P.fields(train_p, n))
    held = {k: np.asarray(P.fields(v, n)) for k, v in coh.items()}
    info = dict(train=dict(sources=len(train_p), parameters_sha256=K.sha_array(train_p),
                           seconds=time.perf_counter() - t0),
                **{k: dict(sources=len(v), parameters_sha256=K.sha_array(v))
                   for k, v in coh.items()})
    fine = cfg['fine_intervals']
    extra_mesh = dict(intervals=fine, U=np.asarray(P.fields(coh['dev12'], fine)))
    return Utr, np.arange(len(fit)), held, info, 'dev12', extra_mesh


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--incumbent', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import json
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    smoke = bool(cfg.get('smoke', False))

    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    R_ = dict(lane='bank-floor', phase='representation', pde=cfg['pde'], config=cfg,
              source_commit=os.environ.get('SOURCE_COMMIT', 'unknown'),
              slurm_job=os.environ.get('SLURM_JOB_ID', 'local'), gpu=gpu,
              jax=jax.__version__, backend=jax.default_backend(), x64=True,
              matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'unset'),
              incumbent_sha256=K.sha_file(a.incumbent), arms={}, complete=False)
    assert smoke or R_['backend'] == 'gpu'
    save = lambda: K.dump(out / 'result.json', R_)

    Utr_h, sub, held_h, dinfo, primary, fine = (burgers_setup if cfg['pde'] == 'burgers2d'
                                                else poisson_setup)(cfg)
    R_['data'] = dinfo
    n = Utr_h.shape[1]
    xy = K.grid(cfg['intervals'])
    assert len(xy) == n
    U = jnp.asarray(Utr_h)
    held = {k: jnp.asarray(v) for k, v in held_h.items()}
    np.savez(out / 'ckpt' / f'{cfg["pde"]}_{primary}_fields.npz', U=held_h[primary])
    S = U.shape[0]
    un = np.asarray(jnp.sqrt(jnp.sum(U * U, axis=1)))
    probe_idx = np.sort(np.random.default_rng(cfg['probe_seed']).choice(
        S, size=min(cfg['probe_snapshots'], S), replace=False))
    Uprobe = U[jnp.asarray(probe_idx)]
    R_['data']['train_snapshots'] = int(S)
    R_['data']['n'] = int(n)
    save()

    inc = pickle.load(open(a.incumbent, 'rb'))
    inc_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, F64), inc['params'])
    inc_block = K.bank_block(inc['params'])
    inc_sha = K.weights_sha(inc_block)
    Kdim = int(inc['params']['h_lin'].shape[0])
    G_inc = K.bank_of([inc_block], xy)
    # head-transplant fields: the incumbent head's own decoded manifold samples
    Z = np.asarray(inc['Z_tr'])
    zi = np.sort(np.random.default_rng(cfg['probe_seed']).choice(
        len(Z), size=min(cfg['transplant_codes'], len(Z)), replace=False))
    Fman = jax.jit(lambda p, z, G: sc.head(p, z) @ G.T)(inc_params, jnp.asarray(Z[zi]), G_inc)

    def score(Q):
        o = dict(train_probe=K.summarise(K.floors(Q, Uprobe)),
                 train_full=K.summarise(K.floors(Q, U, chunk=2048)),
                 train_sub=K.summarise(K.floors(Q, U[:len(sub)], chunk=2048)))
        for k, v in held.items():
            e = K.floors(Q, v)
            o[k] = dict(K.summarise(e), per_snapshot=e.tolist() if e.size <= 512 else None)
        o['head_transplant_defect'] = K.summarise(K.floors(Q, Fman))
        return o

    def score_bank(blocks, tag):
        G = K.bank_of(blocks, xy)
        Q, info = K.orth_basis(G)
        o = dict(basis=info, floors=score(Q), cost=K.cost_model(n, G.shape[1], Kdim))
        del G, Q
        if fine and tag:
            Gf = K.bank_of(blocks, K.grid(fine['intervals']))
            Qf, inff = K.orth_basis(Gf)
            o['fine_mesh'] = dict(intervals=fine['intervals'], basis=inff,
                                  dev=K.summarise(K.floors(Qf, fine['U'], chunk=4)))
            del Gf, Qf
        return o

    def keep(blocks, tag):
        p = out / 'ckpt' / f'{cfg["pde"]}_{tag}.pkl'
        with open(p, 'wb') as f:
            pickle.dump(dict(blocks=host(blocks), pde=cfg['pde'], tag=tag,
                             intervals=cfg['intervals'], source_commit=R_['source_commit'],
                             slurm_job=R_['slurm_job']), f)
        return dict(file=p.name, sha256=K.sha_file(p), bytes=p.stat().st_size,
                    weights_sha256=K.weights_sha(blocks))

    arms = cfg['arms']

    # ---- incumbent control ------------------------------------------------------
    r = score_bank([inc_block], 'inc512')
    r['weights_sha256'] = inc_sha
    got = r['floors'][primary]['worst']
    r['reproduces_logged_floor'] = dict(logged=cfg['logged_incumbent_floor'], measured=got,
                                        relative_difference=abs(got / cfg['logged_incumbent_floor'] - 1))
    R_['arms']['inc512'] = r
    save()
    print(f'ARM inc512 {primary} worst {got:.6e} (logged {cfg["logged_incumbent_floor"]:.6e})', flush=True)
    assert smoke or r['reproduces_logged_floor']['relative_difference'] < cfg['floor_gate'], \
        'FIDELITY GATE: incumbent floor not reproduced'
    inc_probe = K.floors(K.orth_basis(G_inc)[0], Uprobe)

    # ---- control that must fail: an untrained random block ------------------------
    rb = K.fresh_block(jax.random.PRNGKey(cfg['seed'] + 99), 512, **cfg['fresh_arch'],
                       g_hidden=1024, out_scale=float(inc_block['out_scale']))
    r = score_bank([rb], '')
    r['must_exceed'] = 0.05
    r['control_failed_as_required'] = bool(r['floors'][primary]['worst'] > 0.05)
    R_['arms']['random512'] = r
    save()
    print(f'ARM random512 {primary} worst {r["floors"][primary]["worst"]:.4e}', flush=True)

    # ---- POD controls -------------------------------------------------------------
    if 'pod' in arms:
        rmax = max(cfg['pod_ranks'])
        variants = (('pod', 1. / un, None), ('podraw', np.ones(S), None),
                    ('pod_sub', 1. / un, sub), ('podraw_sub', np.ones(S), sub))
        # The deflated POD holds a scaled working copy of its snapshots next to a 26k eigh
        # workspace; with U resident too that exhausted an 80 GB A100 (job 4053195). So U leaves
        # the device while the bases are computed (kept on the host) and returns for scoring.
        del U
        bases = {}
        for name, scale, idx in variants:
            rows = np.arange(S) if idx is None else np.asarray(idx)
            Qall, info = K.pod_deflated(Utr_h, scale, rmax, idx=idx, ranks=cfg['pod_ranks'])
            assert info['rank'] == min(rmax, len(rows)), info
            bases[name] = (np.asarray(Qall), info)
            del Qall
        U = jnp.asarray(Utr_h)
        for name, scale, idx in variants:
            Qh, info = bases.pop(name)
            Qall = jnp.asarray(Qh)
            for rk in cfg['pod_ranks']:
                if rk > Qall.shape[1]:
                    continue
                Q = Qall[:, :rk]
                R_['arms'][f'{name}{rk}'] = dict(basis=dict(info, columns=rk, rank=rk),
                                                 floors=score(Q), cost=K.cost_model(n, rk, Kdim),
                                                 grid_bound=True)
                f = R_['arms'][f'{name}{rk}']['floors']
                if name in ('pod', 'pod_sub') and str(rk) in info['tail_mean_sq']:
                    # known answer: the mean squared relative floor on the POD's own snapshots
                    # equals the energy left after rk modes / snapshot count
                    own = f['train_full' if name == 'pod' else 'train_sub']['rms'] ** 2
                    tail = info['tail_mean_sq'][str(rk)]
                    chk = abs(own / max(tail['mean_sq'], 1e-300) - 1)
                    R_['arms'][f'{name}{rk}']['eigen_tail_check'] = dict(
                        measured=own, eigen=tail['mean_sq'], relative_difference=chk,
                        resolvable=tail['resolvable'])
                    assert (not tail['resolvable']) or chk < 1e-2, \
                        f'{name}{rk}: deflated POD fails its energy identity ({chk:.2e})'
                print(f'ARM {name}{rk} probe rms {f["train_probe"]["rms"]:.4e} '
                      + ' '.join(f'{k} worst {f[k]["worst"]:.4e}' for k in held), flush=True)
            if name in ('pod', 'podraw'):
                p = out / 'ckpt' / f'{cfg["pde"]}_{name}{rmax}_modes.npy'
                np.save(p, Qh)
                R_['arms'][f'{name}{rmax}']['checkpoint'] = dict(
                    file=p.name, sha256=K.sha_file(p), bytes=p.stat().st_size,
                    note='columns are nested: the first r columns are the rank-r basis')
            del Qall, Qh
            save()
    del Utr_h

    # ---- learned arms ---------------------------------------------------------------
    def probe_fn(blocks):
        Q, info = K.orth_basis(K.bank_of(blocks, xy))
        return dict(rank=info['rank'], cond=info['condition_number'],
                    train_probe_rms=K.summarise(K.floors(Q, Uprobe))['rms'],
                    **{k: K.summarise(K.floors(Q, held[k]))['worst'] for k in held})

    for spec in cfg['learned']:
        tag = spec['tag']
        if tag not in arms:
            continue
        t0 = time.perf_counter()
        blocks = [jax.tree_util.tree_map(lambda x: x, inc_block)]
        lrs = [spec['lr_warm']]
        if spec['R'] > 512:
            d = spec['R'] - 512
            blocks.append(K.fresh_block(jax.random.PRNGKey(cfg['seed'] + spec['R']), d,
                                        **cfg['fresh_arch'], g_hidden=max(1024, d),
                                        out_scale=float(inc_block['out_scale'])))
            lrs.append(spec['lr_fresh'])
        # ---- warm-start assertions (DESIGN): hard, before step 1
        assert K.weights_sha(blocks[0]) == inc_sha, 'warm start: block 0 is not the incumbent'
        G0 = K.bank_of(blocks, xy)
        dmax = float(jnp.max(jnp.abs(G0[:, :512] - G_inc)))
        assert dmax == 0.0, f'warm start: block-0 features differ from the incumbent bank ({dmax:.2e})'
        # the incumbent head, zero-padded, must decode to the identical field on the new bank
        hz = sc.head(inc_params, jnp.asarray(Z[zi[:64]]))
        hz_pad = jnp.concatenate([hz, jnp.zeros((hz.shape[0], G0.shape[1] - 512), F64)], axis=1)
        ddec = float(jnp.max(jnp.abs(hz_pad @ G0.T - hz @ G_inc.T)))
        assert ddec == 0.0, f'warm start: zero-padded head decodes differently ({ddec:.2e})'
        Q0, info0 = K.orth_basis(G0)
        init_probe = K.floors(Q0, Uprobe)
        excess = float(np.max(init_probe - inc_probe))
        assert excess <= 1e-9, f'warm start: initial floor exceeds the incumbent by {excess:.2e}'
        init_scores = score(Q0)
        del G0, Q0
        warm = dict(block0_weights_match=True, zero_padded_head_decode_max_abs_difference=ddec, block0_feature_max_abs_difference=dmax,
                    max_probe_floor_excess_over_incumbent=excess, verified=True,
                    initial_basis=info0, initial_floors=init_scores)
        print(f'ARM {tag} warm start verified: init probe rms {init_scores["train_probe"]["rms"]:.4e} '
              f'(incumbent {float(np.sqrt(np.mean(inc_probe ** 2))):.4e})', flush=True)
        blocks, tinfo = K.train_varpro(blocks, lrs, xy, U, spec['steps'], spec['batch'],
                                       cfg['seed'] + 7 * spec['R'], ridge=cfg['ridge'],
                                       log_every=cfg['log_every'], tag=tag, probe=probe_fn)
        r = score_bank(blocks, tag)
        r['warm_start'] = warm
        r['training'] = tinfo
        took = r['floors']['train_probe']['rms'] < init_scores['train_probe']['rms']
        r['training_took'] = bool(took)
        r['label'] = 'ok' if took else 'training_did_not_take'
        r['checkpoint'] = keep(blocks, tag)
        r['seconds'] = time.perf_counter() - t0
        R_['arms'][tag] = r
        save()
        f = r['floors']
        print(f'ARM {tag} [{r["label"]}] rank {r["basis"]["rank"]} cond {r["basis"]["condition_number"]:.2e} '
              f'probe rms {f["train_probe"]["rms"]:.4e} '
              + ' '.join(f'{k} worst {f[k]["worst"]:.4e}' for k in held), flush=True)

    R_['complete'] = True
    save()
    print('REP-DONE', flush=True)


if __name__ == '__main__':
    main()
