"""A1 smearing test with a sine control (DESIGN §2 A1). CPU.

  gen      (imports JAX; run with JAX_PLATFORMS=cpu under the slot guard): selects T6 and X, runs the incumbent upwind
           generator (byte-identical deps/burgers2d_film.py) at n = 257 on X and at n = 256 on T6, saves the six output
           fields per case to runs/a1/upw_<cohort>_<case>_n<n>.npz and writes runs/a1/MANIFEST.sha256.
  analyze  (NumPy only): floors, ratios, smearing excess D, LOO POD controls, gates; writes checks/a1_smear.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import fcommon as F

OUT = F.HERE / 'runs/a1'
STEPS = (0, 10, 20, 30, 40, 50)
CLIP = 1e-14                     # relative floor clip (Amendment 1)


def floors384_inrange():
    Q = F.orth(F.G(384))
    rows = []
    for coh, c in F.all_cases(F.IN_RANGE):
        u, m = F.load_ref(coh, c)
        if m['eligible']:
            rows.append((f'{coh}|{c}', F.floor_Q(u, Q, m['n0'])))
    return rows


def select_sets():
    rows = floors384_inrange()
    others = sorted([r for r in rows if r[0] != 'val32|19'], key=lambda r: -r[1])[:5]
    T6 = ['val32|19'] + [r[0] for r in others]
    X = [r[0] for r in rows] + [f'N0s|{c}' for c in F.regime_cases('N0s') if F.load_ref('N0s', c)[1]['eligible']]
    return T6, X, dict(rows)


def gen():
    sys.path.insert(0, str(F.HERE / 'deps'))
    import jax
    import jax.numpy as jnp
    import burgers2d_film as BF
    assert jax.config.jax_enable_x64 and jax.default_backend() == 'cpu'
    T6, X, fl = select_sets()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sets.json').write_text(json.dumps(dict(T6=T6, X=X, floor384_E0=fl), indent=1))
    jobs = [(cid, 257) for cid in X] + [(cid, 256) for cid in T6]
    ro = {n: BF.make_rollout(n)[0] for n in (256, 257)}
    for cid, n in jobs:
        coh, c = cid.split('|')
        p = OUT / f'upw_{coh}_{int(c):03d}_n{n}.npz'
        if p.exists():
            continue
        d = F.load_ref(coh, int(c))[1]['desc']
        u0 = BF.blob_ic(n, d['cx'], d['cy'], d['w'], d['a'])
        t = time.time()
        snaps, res = ro[n](jnp.asarray(u0[None]), jnp.asarray([d['nu']]))
        snaps, res = np.asarray(snaps)[:, 0], np.asarray(res)[:, 0]
        np.savez(p, u=snaps[list(STEPS)].reshape(6, n, n), res=res, nu=d['nu'], a=d['a'], n=n,
                 gen_sha256=F.sha(F.HERE / 'deps/burgers2d_film.py'))
        print(cid, n, f'{time.time() - t:.1f}s', 'max_res', float(res.max()), flush=True)
    lines = [f'{F.sha(p)}  {p.name}' for p in sorted(OUT.glob('upw_*.npz'))]
    (OUT / 'MANIFEST.sha256').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------- analysis
def grid_bases_256(r_list, K_list):
    """Bank and sine columns at the n = 256 training grid (x = i/255), zero walls."""
    x = np.arange(256) / 255.
    Xp = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    P, T = F.NC.bank()
    V = np.concatenate([F.NC.NB.features_grad(P, Xp[s:s + 8192])[0] for s in range(0, len(Xp), 8192)])
    wall = (Xp[:, 0] == 0) | (Xp[:, 0] == 1) | (Xp[:, 1] == 0) | (Xp[:, 1] == 1)
    V[wall] = 0.
    k = np.arange(1, 255)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    order = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
    out = {}
    for r in r_list:
        out[f'bank{r}'] = F.orth(V @ T[:, :r])
    for K in K_list:
        kx, ky = KX.ravel()[order[:K]], KY.ravel()[order[:K]]
        c = np.sqrt(2. / 255.)
        S = (c * np.sin(np.pi * x[:, None] * kx[None]))[:, None, :] * (c * np.sin(np.pi * x[:, None] * ky[None]))[None]
        out[f'sine{K}'] = S.reshape(256 * 256, K)
    return out


def exact_d_fixture():
    """Amendment 2: synthetic fields with analytically prescribed D (= 10 and 0.1) at r = 128, 384; returns max rel err."""
    rng = np.random.default_rng(424242)
    errs = {}
    for r in (128, 384):
        B, S = F.orth(F.G(r)), F.sine_cols(np.arange(r))
        y = F.sine_cols([5])[:, 0]
        z = rng.standard_normal(B.shape[0])
        QU = F.orth(np.concatenate([B, S], 1))
        for _ in range(2):
            z -= QU @ (QU.T @ z)
        z /= np.linalg.norm(z)
        c = float(np.linalg.norm(y - B @ (B.T @ y)))
        alpha = 1.0
        for target in (10., 0.1):
            gamma = alpha * np.sqrt(target ** 2 - 1) / c if target > 1 else alpha * np.sqrt(1 / target ** 2 - 1) / c
            u0 = np.ones(B.shape[0])
            a_f = np.stack([u0] + [alpha * z] * 5)
            b_f = np.stack([u0] + [alpha * z + gamma * y] * 5)
            upw, ref = (a_f, b_f) if target > 1 else (b_f, a_f)
            n0 = float(np.linalg.norm(u0))
            fl = lambda u, Q: max(F.floor_Q(u, Q, n0), CLIP)
            D = (fl(ref, B) / fl(upw, B)) / (fl(ref, S) / fl(upw, S))
            errs[f'r{r}_D{target:g}'] = abs(D - target) / target
    return dict(rel_err=errs, ok=all(e <= 1e-10 for e in errs.values()))


def analyze():
    sets = json.loads((OUT / 'sets.json').read_text())
    T6, X = sets['T6'], sets['X']
    ref, upw, meta = {}, {}, {}
    gates = dict(ic_rel_max=0., newton_res_max=0., all_finite=True)
    for cid in X:
        coh, c = cid.split('|')
        u, m = F.load_ref(coh, int(c))
        z = np.load(OUT / f'upw_{coh}_{int(c):03d}_n257.npz')
        ref[cid], upw[cid], meta[cid] = u, z['u'], m
        gates['ic_rel_max'] = max(gates['ic_rel_max'], float(np.linalg.norm(z['u'][0] - u[0]) / np.linalg.norm(u[0])))
        gates['newton_res_max'] = max(gates['newton_res_max'], float(z['res'].max()))
        gates['all_finite'] &= bool(np.isfinite(z['u']).all())
    gates['G1_ic_pass'] = gates['ic_rel_max'] <= 1e-12
    gates['G2_newton_pass'] = gates['newton_res_max'] <= 1e-8 and gates['all_finite']
    bases = {f'bank{r}': F.orth(F.G(r)) for r in (128, 384)}
    bases.update({f'sine{K}': F.sine_cols(np.arange(K)) for K in (128, 384)})
    rows = {}
    h = 1 / 255
    for cid in X:
        n0 = meta[cid]['n0']
        d = meta[cid]['desc']
        row = dict(case=cid, a=d['a'], nu=d['nu'], s=d['a'] / d['nu'], q=d['a'] * h / 2 / d['nu'],
                   e_upw_vs_ref=F.err(upw[cid], ref[cid], n0))
        for b, Q in bases.items():
            row[f'f_ref_{b}'] = F.floor_Q(ref[cid], Q, n0)
            row[f'f_upw_{b}'] = F.floor_Q(upw[cid], Q, np.linalg.norm(upw[cid][0]))
        # LOO POD controls (size 128) from the six output times of the other cases
        oth = [o for o in X if o != cid]
        for src, store in (('upw', upw), ('ref', ref)):
            M = np.concatenate([store[o].reshape(6, -1) for o in oth]).T
            Q = np.linalg.svd(M, full_matrices=False)[0][:, :128]
            row[f'f_ref_pod_{src}'] = F.floor_Q(ref[cid], Q, n0)
            row[f'f_upw_pod_{src}'] = F.floor_Q(upw[cid], Q, np.linalg.norm(upw[cid][0]))
        # constructed detector checks (Amendment 1): own upwind / own E0 fields + low sines, sizes 128 and 384
        if cid in T6:
            for r in (128, 384):
                S = F.sine_cols(np.arange(r - 5))
                for src, store in (('upw', upw), ('ref', ref)):
                    Q = F.orth(np.concatenate([S, store[cid].reshape(6, -1)[1:].T], 1))
                    row[f'f_ref_con{src}{r}'] = F.floor_Q(ref[cid], Q, n0)
                    row[f'f_upw_con{src}{r}'] = F.floor_Q(upw[cid], Q, np.linalg.norm(upw[cid][0]))
        for k in [k for k in row if k.startswith('f_')]:
            if row[k] < CLIP:
                row.setdefault('clipped', []).append(k)
                row[k] = CLIP
        rho = lambda b: row[f'f_ref_{b}'] / row[f'f_upw_{b}']
        row['D_bank384'] = rho('bank384') / rho('sine384')
        row['D_bank128'] = rho('bank128') / rho('sine128')
        row['D_pod_upw'] = rho('pod_upw') / rho('sine128')
        row['D_pod_ref'] = rho('pod_ref') / rho('sine128')
        if cid in T6:
            for r in (128, 384):
                row[f'D_conupw{r}'] = rho(f'conupw{r}') / rho(f'sine{r}')
                row[f'D_conref{r}'] = rho(f'conref{r}') / rho(f'sine{r}')
        rows[cid] = row
        print(cid, {k: round(v, 3) for k, v in row.items() if k.startswith('D_')}, flush=True)
    # n = 256 sensitivity on T6 (native training grid)
    b256 = grid_bases_256((128, 384), (128, 384))
    sens = {}
    for cid in T6:
        coh, c = cid.split('|')
        z = np.load(OUT / f'upw_{coh}_{int(c):03d}_n256.npz')['u']
        n0u = np.linalg.norm(z[0])
        fu = {b: F.floor_Q(z, Q, n0u) for b, Q in b256.items()}
        r = rows[cid]
        sens[cid] = dict(D_bank384=(r['f_ref_bank384'] / fu['bank384']) / (r['f_ref_sine384'] / fu['sine384']),
                         D_bank128=(r['f_ref_bank128'] / fu['bank128']) / (r['f_ref_sine128'] / fu['sine128']),
                         **{f'f_upw256_{b}': v for b, v in fu.items()})
    med = lambda key, S: float(np.median([rows[c][key] for c in S]))
    ctrl = dict(Dpos_med_T6=med('D_pod_upw', T6), Dneg_med_T6=med('D_pod_ref', T6),
                resolution_med_T6=float(np.median([rows[c]['D_pod_upw'] / rows[c]['D_pod_ref'] for c in T6])))
    ctrl['C_A1plus_fires'] = ctrl['Dpos_med_T6'] >= 2
    ctrl['C_A1minus_quiet'] = ctrl['Dneg_med_T6'] <= 1.25
    ctrl['resolution_ok'] = ctrl['resolution_med_T6'] >= 2
    ctrl['constructed_upw_D_ge10_all_T6'] = all(rows[c][f'D_conupw{r}'] >= 10 for c in T6 for r in (128, 384))
    ctrl['constructed_ref_D_le0.1_all_T6'] = all(rows[c][f'D_conref{r}'] <= 0.1 for c in T6 for r in (128, 384))
    ctrl['exact_D_fixture'] = exact_d_fixture()
    # Amendment 2: own-field constructions are empirical self-fit checks (reported, not gating)
    ctrl['controls_pass'] = bool(ctrl['C_A1plus_fires'] and ctrl['C_A1minus_quiet'] and ctrl['resolution_ok']
                                 and ctrl['exact_D_fixture']['ok'])
    Db = [rows[c]['D_bank384'] for c in T6]
    mD = float(np.median(Db))
    if not (gates['G1_ic_pass'] and gates['G2_newton_pass']):
        verdict = 'STOPPED (gate failed)'
    elif not ctrl['controls_pass']:
        verdict = 'INCONCLUSIVE (control failed)'
    elif mD >= 2 and sum(d > 1 for d in Db) >= 5:
        verdict = 'smearing-consistent'
    elif mD <= 1.25:
        verdict = 'not smearing-consistent'
    else:
        verdict = 'ambiguous'
    from scipy.stats import spearmanr
    out = dict(design='DESIGN.md §2 A1', provenance=F.provenance(), manifest=(OUT / 'MANIFEST.sha256').read_text(),
               T6=T6, X=X, gates=gates, controls=ctrl,
               summary=dict(D_bank384_med_T6=mD, D_bank384_gt1_T6=int(sum(d > 1 for d in Db)),
                            D_bank128_med_T6=med('D_bank128', T6),
                            D_bank384_med_X=med('D_bank384', X), D_bank128_med_X=med('D_bank128', X),
                            spearman_D384_q_X=float(spearmanr([rows[c]['D_bank384'] for c in X], [rows[c]['q'] for c in X])[0]),
                            e_upw_vs_ref_med_X=med('e_upw_vs_ref', X), e_upw_vs_ref_max_X=float(max(rows[c]['e_upw_vs_ref'] for c in X)),
                            D_bank384_med_T6_n256=float(np.median([sens[c]['D_bank384'] for c in T6]))),
               verdict=verdict, rows=rows, sensitivity_n256=sens)
    F.write_json('a1_smear', out)
    print(json.dumps(dict(verdict=verdict, gates=gates, controls=ctrl, summary=out['summary']), indent=1))


if __name__ == '__main__':
    {'gen': gen, 'analyze': analyze}[sys.argv[1]]()
