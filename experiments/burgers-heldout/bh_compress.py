"""burgers-heldout bh1, steps 1 and 3: seed checkpoints, and the rank-R' bank inside cat1024's span.

    bh_compress.py seed1024 --blocks cat1024.pkl --incumbent inc.pkl --out in/cat1024_seed.pkl
        one separable parameter set equal to cat1024 (V = identity, R = 1024), with the incumbent's
        head zero-padded to R = 1024 and its codes, ONLY so `sep_coeff_extract.py` can run on it
        unchanged (its Gram-identity gate holds for any head and any codes).
    bh_compress.py compress --npz sep_coeff_N256_K16_R1024.npz --blocks ... --incumbent ... --out DIR
        field-metric POD variants of the 131072 extracted training coefficients, floors on dev6 and
        sel32 at 256^2 (hold64 is never touched), pre-registered selection, and the R = 512 seed
        checkpoint `cpod512_seed.pkl` (merged bank + incumbent head + incumbent codes; the head and
        codes are placeholders that `sep_hfit_run.py` replaces from scratch).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import sep_common as sc
import bh_bank as BB

F64 = jnp.float64
OUT_STEPS = np.arange(0, 51, 10)          # the six output times of the error metric (dt = 0.005)
R_LADDER = [256, 384, 512, 640, 768, 1024]


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def fit_xy(n_nodes):
    """Interior nodes of the extraction grid (burgers2d_film / blat_common: n NODES, linspace(0, 1, n)),
    in `blat_common.interior_indices` order. The head and bank were fitted on this grid."""
    x = np.linspace(0., 1., n_nodes)[1:-1]
    return np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)


def seed1024(a):
    cat = load(a.blocks)
    inc = load(a.incumbent)
    merged = BB.merge_blocks(cat['blocks'])
    par = BB.merge_parity(cat['blocks'], merged)
    print('PARITY seed1024', json.dumps(par), flush=True)
    assert par['passed'], par
    R = merged['g'][-1][0].shape[1]
    h = [(np.asarray(w), np.asarray(b)) for w, b in inc['params']['h']]
    w_, b_ = h[-1]
    h[-1] = (np.concatenate([w_, np.zeros((w_.shape[0], R - w_.shape[1]))], 1),
             np.concatenate([b_, np.zeros(R - b_.shape[0])]))
    hl = np.asarray(inc['params']['h_lin'])
    hl = np.concatenate([hl, np.zeros((hl.shape[0], R - hl.shape[1]))], 1)
    params = dict(merged, h=h, h_lin=hl)
    cfg = dict(inc['cfg'], r=int(R), bh_role='seed for sep_coeff_extract only: bank = cat1024 exactly; head/codes = '
               'incumbent zero-padded placeholders', bh_blocks_sha256=BB.sha_array(np.concatenate(
                   [np.asarray(b['B']).ravel() for b in cat['blocks']])), bh_parity=par)
    with open(a.out, 'wb') as f:
        pickle.dump(dict(params=params, Z_tr=np.asarray(inc['Z_tr']), cfg=cfg), f)
    print('WROTE', a.out, flush=True)


def fom_states(N, physical, ntol=1e-6, ltol=1e-8):
    """Every-step same-grid states (51 per case) at mesh N, interior, plus ||u0||."""
    q, _ = e.make_fom(N, .005, None, .25, .005)
    out, worst = [], 0.
    for ph in physical:
        f, it, rn = jax.device_get(q(jnp.asarray(e.initial(N, ph)), float(ph[4]), ntol, ltol))
        assert np.isfinite(f).all()
        worst = max(worst, float(np.max(rn)))
        out.append(np.asarray(f)[:, 1:-1, 1:-1].reshape(f.shape[0], -1))
    return np.stack(out), worst                 # (cases, 51, n)


def floors_for(G, U):
    """per case, per step: ||u - P u|| / ||u0|| (the metric) and / ||u|| (per state)."""
    C, T, n = U.shape
    fl = BB.span_floor(G, U.reshape(C * T, n)).reshape(C, T)
    n0 = np.linalg.norm(U[:, 0], axis=1)
    nu = np.linalg.norm(U, axis=2)
    return fl / n0[:, None], fl / np.maximum(nu, 1e-300)


def summarise(m, s):
    ev = m[:, OUT_STEPS[1:]]
    return dict(worst_evolved_metric=float(ev.max()), worst_all_metric=float(m[:, OUT_STEPS].max()),
                median_evolved_metric=float(np.median(ev)),
                worst_evolved_every_step_metric=float(m[:, 1:].max()),
                worst_evolved_state_relative=float(s[:, OUT_STEPS[1:]].max()),
                worst_case_evolved=int(ev.max(1).argmax()),
                per_case_worst_evolved_metric=ev.max(1).tolist())


def compress(a):
    t_all = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(job=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               x64=True, matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), inputs={}, gates={})
    save = lambda: (out / 'compress.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    for k in ('npz', 'blocks', 'incumbent'):
        rep['inputs'][k] = dict(path=os.path.basename(getattr(a, k)), sha256=BB.sha_array(np.frombuffer(
            Path(getattr(a, k)).read_bytes(), np.uint8)))
    cat = load(a.blocks)
    inc = load(a.incumbent)
    d = np.load(a.npz)
    Gram = jnp.asarray(d['Gram'], F64)
    Rc = Gram.shape[0]
    Lc = BB.chol_whitener(Gram)
    C = np.asarray(d['C_tr'])
    un2, fl2 = np.asarray(d['un2_tr']), np.asarray(d['fl2_tr'])
    traj, tt = np.asarray(d['traj_tr']), np.asarray(d['t_tr'])
    S = C.shape[0]
    rep['training'] = dict(states=int(S), trajectories=int(len(np.unique(traj))), R_source=int(Rc),
                           cat_floor_rms=float(np.sqrt(fl2.sum() / un2.sum())),
                           cat_floor_state_max=float(np.sqrt(np.max(fl2 / un2))))
    # ||u0|| per trajectory: every trajectory's t = 0 state is in the pick (T_EARLY), asserted
    u0n = {}
    for i in np.nonzero(tt == 0)[0]:
        u0n[int(traj[i])] = float(np.sqrt(un2[i]))
    assert len(u0n) == len(np.unique(traj)), 'a trajectory has no t=0 state in the pick'
    variants = dict(raw=np.ones(S), state=1. / np.sqrt(un2), u0=np.array([1. / u0n[int(j)] for j in traj]))
    # scale s: the incumbent bank's Gram trace at the extraction mesh, so coefficient scales carry over
    n_nodes = int(round(np.sqrt(d['U_id'].shape[1]))) + 2
    xy = fit_xy(n_nodes)
    assert len(xy) == d['U_id'].shape[1]
    inc_bank = dict(B=inc['params']['B'], g=inc['params']['g'], out_scale=inc['params']['out_scale'])
    G_inc_fit = BB.block_features([BB.to_device(inc_bank)], xy)
    G_cat_fit = BB.block_features([BB.to_device(b) for b in cat['blocks']], xy)
    gram_dev = float(jnp.linalg.norm(G_cat_fit.T @ G_cat_fit - Gram) / jnp.linalg.norm(Gram))
    rep['gates']['cat_gram_matches_extraction'] = dict(relative=gram_dev, passed=bool(gram_dev <= 1e-10))
    assert gram_dev <= 1e-10, gram_dev
    s_target = float(jnp.sqrt(jnp.trace(G_inc_fit.T @ G_inc_fit)))
    del G_inc_fit, G_cat_fit
    # floors are measured on the EVALUATION grid convention (engines: N intervals, (N-1)^2 interior)
    N = a.floor_mesh
    xe = e.coords(N)
    G_inc = BB.block_features([BB.to_device(inc_bank)], xe)
    G_cat = BB.block_features([BB.to_device(b) for b in cat['blocks']], xe)
    rep['fit_grid'] = dict(nodes=n_nodes, interior=int(len(xy)), convention='burgers2d_film linspace(0,1,nodes)')
    rep['floor_mesh'] = dict(intervals=N, convention='engines: arange(1,N)/N interior, the hires-burgers grid')
    save()

    # selection cohorts at the extraction mesh (hold64 is NOT opened here)
    dev6 = np.concatenate([e.params_draw(7090702, 4), e.params_draw(911702, 2)])
    sel32 = e.params_draw(20260927, 32)
    t0 = time.perf_counter()
    Udev, wdev = fom_states(N, dev6)
    Usel, wsel = fom_states(N, sel32)
    rep['cohorts'] = dict(dev6=dict(cases=6, sha256=BB.sha_array(dev6), fom_max_rel_residual=wdev),
                          sel32=dict(cases=32, sha256=BB.sha_array(sel32), fom_max_rel_residual=wsel,
                                     definition='params_draw(20260927, 32)'),
                          seconds=time.perf_counter() - t0, fom='engines.make_fom ntol 1e-6 ltol 1e-8, every step')
    print('COHORTS', round(time.perf_counter() - t0, 1), flush=True)

    rep['floors'] = {}

    def score(tag, G):
        r = {}
        for cn, U in (('dev6', Udev), ('sel32', Usel)):
            m, s = floors_for(G, U)
            r[cn] = summarise(m, s)
        rep['floors'][tag] = r
        print('FLOOR', tag, 'sel32', f"{100 * r['sel32']['worst_evolved_metric']:.4f}%",
              'dev6', f"{100 * r['dev6']['worst_evolved_metric']:.4f}%", flush=True)
        save()
        return r

    score('incumbent_inc512', G_inc)
    score('cat1024', G_cat)
    Ws = {}
    for vn, w in variants.items():
        W, sv = BB.pod_directions(C, Lc, w, max(R_LADDER))
        Ws[vn] = W
        A = np.asarray(jnp.asarray(C) @ Lc)
        for r in R_LADDER:
            if r > Rc:
                continue
            Wr = W[:, :r]
            res = A - (A @ Wr) @ Wr.T                    # in-span residual of each training state
            tr = (fl2 + np.sum(res * res, 1)) / un2
            G_r = G_cat @ jnp.asarray(BB.compressed_V(Lc, Wr, 1.0))
            fr = score(f'pod_{vn}_R{r}', G_r)
            fr['train'] = dict(state_relative_rms=float(np.sqrt(np.mean(tr))), state_relative_max=float(np.sqrt(tr.max())))
        rep.setdefault('pod_singular_values', {})[vn] = sv[:1024].tolist()
        save()

    # pre-registered selection: smallest worst evolved sel32 floor at R = 512 (metric: /||u0||)
    cand = {vn: rep['floors'][f'pod_{vn}_R512']['sel32']['worst_evolved_metric'] for vn in variants}
    choice = min(cand, key=cand.get)
    inc_sel = rep['floors']['incumbent_inc512']['sel32']['worst_evolved_metric']
    rep['selection'] = dict(rule='smallest worst evolved sel32 floor at R=512 in the error metric', candidates=cand,
                            chosen=choice, chosen_floor=cand[choice], incumbent_floor=inc_sel,
                            below_incumbent=bool(cand[choice] < inc_sel),
                            stop_rule_0p6_percent=bool(cand[choice] < 6e-3))
    print('SELECTED', choice, cand, 'incumbent', inc_sel, flush=True)
    W = Ws[choice][:, :512]
    s = s_target / np.sqrt(512.)
    V = BB.compressed_V(Lc, W, s)
    merged = BB.merge_blocks(cat['blocks'], V)
    par = BB.merge_parity(cat['blocks'], merged, V)
    rep['gates']['merged_bank_parity'] = par
    print('PARITY cpod512', json.dumps(par), flush=True)
    assert par['passed'], par
    Gmf = sc.features(BB.to_device(merged), jnp.asarray(xy, F64))
    gm = Gmf.T @ Gmf
    del Gmf
    Gm = sc.features(BB.to_device(merged), jnp.asarray(xe, F64))
    orth = float(jnp.max(jnp.abs(gm / s ** 2 - jnp.eye(512))))
    rep['gates']['compressed_bank_orthogonal_at_fit_mesh'] = dict(max_abs_dev=orth, passed=bool(orth <= 1e-8),
                                                                  scale=s, trace_incumbent=s_target ** 2,
                                                                  trace_new=float(jnp.trace(gm)))
    assert orth <= 1e-8, orth
    fm = score('cpod512_final_merged', Gm)
    rep['gates']['final_floor_matches_selection'] = dict(
        passed=bool(abs(fm['sel32']['worst_evolved_metric'] - cand[choice]) <= 1e-9 * max(1., cand[choice]) + 1e-12))
    np.savez_compressed(out / 'cpod512_V.npz', V=V, W=W, scale=s, variant=choice)
    params = dict(merged, h=[(np.asarray(w), np.asarray(b)) for w, b in inc['params']['h']],
                  h_lin=np.asarray(inc['params']['h_lin']))
    cfg = dict(inc['cfg'], r=512, bh_role=('seed for sep_coeff_extract/sep_hfit_run: bank = cpod512 (cat1024 span, '
                                          f'POD variant {choice}); head and codes are incumbent PLACEHOLDERS'),
               bh_variant=choice, bh_scale=s, bh_V_sha256=BB.sha_array(V), bh_parity=par)
    with open(out / 'cpod512_seed.pkl', 'wb') as f:
        pickle.dump(dict(params=params, Z_tr=np.asarray(inc['Z_tr']), cfg=cfg), f)
    rep['seconds'] = time.perf_counter() - t_all
    rep['complete'] = True
    save()
    print('COMPRESS COMPLETE', round(rep['seconds'], 1), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['seed1024', 'compress'])
    p.add_argument('--blocks', required=True)
    p.add_argument('--incumbent', required=True)
    p.add_argument('--npz')
    p.add_argument('--out', required=True)
    p.add_argument('--floor-mesh', type=int, default=256)
    a = p.parse_args()
    assert jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()}', flush=True)
    (seed1024 if a.mode == 'seed1024' else compress)(a)


if __name__ == '__main__':
    main()
