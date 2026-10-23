"""imp-box cell driver (DESIGN §1-§6 + A1-A4). One cluster job = one (W, level) cell; stages run in order, each
checkpointed in the job directory (a re-run skips finished stages):

  ref     dev/val references at N_ref (+ 3N/4, N/2, 2n, padding 2N), gates, retries, eps, allowance U, G7 / MF-IC,
          MF-REF-real, MF-PAD-chk, cohort hashes (the test split is NOT generated here)
  data    96 training trajectories (21 snapshots, sine coefficients at N_ref) [+ train-c for W40], training gates,
          bank samples (lattice 1/256; A4: 1/512)
  pod     POD (PODG / ECSW) and centred POD (TRANS); k_q per representation (A2.3); M, K checks; G-OPS
  bank    (bank cells only) Stage-A bank on the box envelope, ordering on the reference rule, k_q^bank
  ladder  ladders.py: every arm's ladder on val32
  time    timing.py: finalists, Q1/Q2, paired timing, components, r_x, diagnostics, freeze
  test    teststage.py (separate, final job): test references and gates, frozen finalists, test Q1/Q2, re-timing

  python cell.py --W W40 --lev A2 --out <jobdir> --stages ref,data,pod,bank,ladder,time [--smoke] [--bank]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'vendor'))
import geom as G        # noqa: E402
import spec as SP       # noqa: E402
import refgate as RG    # noqa: E402
import podsine as PS    # noqa: E402

BANK_CELLS_W = ('W40', 'W08')
TRANS_W = ('W40',)
SMOKE = dict(train=12, dev=2, val=4, test=2)
KQ_LADDER = (4, 8, 16, 32, 64)          # memory cap (A5): k_q = 64, q = 16 is ~1M points


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def env_record():
    return dict(jax=jax.__version__, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), x64=bool(jax.config.read('jax_enable_x64')),
                host=os.uname().nodename, commit=os.environ.get('LANE_COMMIT'),
                IGEO_NCAP=os.environ.get('IGEO_NCAP'), IMPBOX_TINY=os.environ.get('IMPBOX_TINY'),
                IMPBOX_BANK_STEPS=os.environ.get('IMPBOX_BANK_STEPS'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:                        # streamed (code audit 2)
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()


def eligible(D, z):
    """Selection population: the eligible rows of a split (descriptors and reference fields)."""
    m = np.asarray(z['eligible'], bool)
    return D[m], z['U'][m]


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def cohorts(a):
    D = G.cohorts(a.W, a.lev, n_train=a.n_train)
    if a.smoke:
        D = {k: (v[:SMOKE[k]] if k in SMOKE else v[:SMOKE['train']]) for k, v in D.items()}
    return D


def nref(a):
    N = G.LEVELS[a.lev][1]
    return a.nref_override or N


# ----------------------------------------------------------------------------------------- runners (batched) ---

def run_cohort(q, D, chunk=8):
    import hoist as HZ                                    # A16: hoisted vmapped query (code audit 12)
    vq = HZ.hoisted(jax.vmap(q), example=jax.ShapeDtypeStruct((chunk, 5), jnp.float64))
    outs, sts = [], []
    for s in range(0, len(D), chunk):
        part = D[s:s + chunk]
        pad = chunk - len(part)
        P_ = np.concatenate([part, np.repeat(part[-1:], pad, 0)]) if pad else part
        o, st = vq(jnp.asarray(P_))
        outs.append(np.asarray(o)[:len(part)])
        sts.append(np.asarray(st)[:len(part)])
    return np.concatenate(outs), np.concatenate(sts)


def synth_cohort(B, N, lat=G.LAT):
    S = P_lattice_S(N, lat)
    f = jax.jit(jax.vmap(jax.vmap(lambda b, S_: jnp.pad(S_ @ b @ S_.T, 1).reshape(-1), in_axes=(0, None)),
                         in_axes=(0, None)))                # synthesis matrix passed as an argument (A16)
    return np.concatenate([np.asarray(f(jnp.asarray(B[s:s + 4]), S)) for s in range(0, len(B), 4)])


def P_lattice_S(N, lat):
    import pscore as P_
    return P_.lattice_S(N, lat)


# ------------------------------------------------------------------------------------------------ stage ref ----

def ref_levels(N):
    return N, 3 * N // 4, N // 2


def ref_split(D, N, pad_idx, log_tag=''):
    """Gates on one split (A1.2/A2.2) at reference size N. Returns (lattice U at N (n,6,P), rec dict)."""
    n = SP.ref_cfl_steps(N)
    Nf, N34, N12 = ref_levels(N)
    t0 = time.perf_counter()
    B, st = run_cohort(SP.make_fom(N, n, out='coef'), D, chunk=4)
    B34, _ = run_cohort(SP.make_fom(N34, n, out='coef'), D, chunk=4)
    B12, _ = run_cohort(SP.make_fom(N12, n, out='coef'), D, chunk=4)
    Bt, _ = run_cohort(SP.make_fom(N, 2 * n, out='coef'), D, chunk=4)
    U = synth_cohort(B, N)
    lat = lambda X, M_: synth_cohort(X, M_)
    d34 = np.maximum(RG.coef_diff(B34, B), RG.lat_diff(lat(B34, N34), U))
    d12 = np.maximum(RG.coef_diff(B12, B), RG.lat_diff(lat(B12, N12), U))
    dt = np.maximum(RG.coef_diff(Bt, B), RG.lat_diff(lat(Bt, N), U))
    dp = np.full(len(D), np.nan)
    if len(pad_idx):
        Bp, _ = run_cohort(SP.make_fom(N, n, M=2 * N, out='coef'), D[pad_idx], chunk=4)
        dp[pad_idx] = np.maximum(RG.coef_diff(Bp, B[pad_idx]), RG.lat_diff(lat(Bp, N), U[pad_idx]))
    measured = np.isfinite(dp)
    dp_fill = np.where(measured, dp, np.nanmax(dp))
    lab = RG.case_labels(d34, d12, dt, dp_fill)
    rec = dict(N=N, n=n, d34=d34.tolist(), d12=d12.tolist(), dt=dt.tolist(), dp=dp_fill.tolist(),
               dp_measured=measured.tolist(), nonfinite=int(np.sum(st[:, 0] > 0)), seconds=time.perf_counter() - t0,
               **lab)
    log(f'ref{log_tag} N{N} n{n}: d34 max {d34.max():.2e} d12 max {d12.max():.2e} dt max {dt.max():.2e} '
        f'dp max {np.nanmax(dp):.2e} eligible {sum(lab["eligible"])}/{len(D)} [{rec["seconds"]:.0f}s]')
    raw = dict(B=B, B34=B34, N34=N34, d12=d12, dt=dt, dp=dp_fill, d34=d34, pad_idx=np.asarray(pad_idx),
               Bp=(Bp if len(pad_idx) else None))
    return U, raw, rec


def pad_population(D, lev):
    """A2.2: every case at A1-A3; at A4 the 8 steepest (by s = a / nu)."""
    if lev != 'A4' or len(D) <= 8:
        return np.arange(len(D))
    return np.sort(np.argsort(-G.steep(D))[:8])


def gate_split(a, D, split):
    """References + gates + one retry at ceil(4N/3) (multiple of 32) for failing cases (A2.2)."""
    N = nref(a)
    pidx = np.arange(len(D)) if split == 'dev' else pad_population(D, a.lev)
    U, raw, rec = ref_split(D, N, pidx, f' {split}')
    elig = np.array(rec['eligible'])
    rec['N_case'] = [N] * len(D)
    rec['retry'] = None
    if not elig.all():
        N2 = int(math.ceil(4 * N / 3 / 32) * 32)
        bad = np.where(~elig)[0]
        U2, _, r2 = ref_split(D[bad], N2, np.arange(len(bad)), f' {split} retry')
        for j, i in enumerate(bad):
            for k in ('d34', 'd12', 'dt', 'dp', 'eligible', 'eps', 'roundoff', 'reason'):
                rec[k][i] = r2[k][j]
            rec['N_case'][i] = N2
            U[i] = U2[j]
        rec['retry'] = dict(N=N2, cases=bad.tolist(), eligible=r2['eligible'])
    n_bad = int(len(D) - sum(rec['eligible']))
    rec['n_ineligible'] = n_bad
    rec['scorable'] = bool(n_bad <= 2 and sum(rec['eligible']) > 0)
    return U, raw, rec                       # raw: first-pass arrays at N (MF checker controls only)


def g7_all(a, D):
    """G7 for every distinct IC table the cell uses (A1.2; code audit 1): FOM ladder sizes, extensions, reference levels and
    retry levels, on val[0] and the narrowest-bump val case, against a 512 x 32 table."""
    import ladders as LS
    N = nref(a)
    N2 = int(math.ceil(4 * N / 3 / 32) * 32)
    sizes = sorted(set(LS.SPEC_N + LS.SPEC_EXT + ref_levels(N) + ref_levels(N2)))
    cases = [D['val'][0], D['val'][int(np.argmin(D['val'][:, 2]))]]
    worst = {}
    for n_ in sizes:
        e = 0.0
        for d in cases:
            fine = np.asarray(SP.ICTable(n_, 512, 32).coefs(jnp.asarray(d)))
            prod = np.asarray(SP.ICTable(n_).coefs(jnp.asarray(d)))
            e = max(e, float(np.linalg.norm(prod - fine) / np.linalg.norm(fine)))
        worst[str(n_)] = e
    return dict(per_N=worst, max=max(worst.values()), passed=bool(max(worst.values()) <= 1e-12))


def stage_ref(a, out, D):
    hashes = G.check_disjoint(a.n_train)
    cell_h = {s: sha_arr(v) for s, v in D.items()}           # every split's descriptors (test: descriptors only)
    (out / 'cohorts.json').write_text(json.dumps(dict(base_hashes=hashes, cell_hashes=cell_h, W=a.W, lev=a.lev,
                                                      n={k: len(v) for k, v in D.items()},
                                                      note='test descriptors are hashed; no test solution exists '
                                                           'before the test stage'), indent=1))
    res, raws = {}, {}
    for split in ('dev', 'val'):
        U, raw, rec = gate_split(a, D[split], split)
        np.savez(out / f'refs_{split}.npz', U=U, eps=np.array(rec['eps']), eligible=np.array(rec['eligible']))
        res[split], raws[split] = rec, raw
    eps = res['dev']['eps'] + res['val']['eps']
    el = res['dev']['eligible'] + res['val']['eligible']
    U_allow = RG.allowance(eps, el)
    scorable = bool(res['dev']['scorable'] and res['val']['scorable'])
    N = nref(a)
    d0 = jnp.asarray(D['val'][0])
    fine = np.asarray(SP.ICTable(N, 512, 32).coefs(d0))
    bad = np.asarray(SP.ICTable(N, 2, 8).coefs(d0))
    mfic = float(np.linalg.norm(bad - fine) / np.linalg.norm(fine))
    # MF-REF-real / MF-PAD-chk through the production gate (case_labels) on a real reference (A3.5; code audit 1)
    rv = raws['val']
    rng = np.random.default_rng(7)

    def perturb(Bc, size):
        Bq = Bc.copy()
        for k in range(1, Bq.shape[1]):
            dlt = rng.standard_normal(Bq.shape[2:])
            Bq[0, k] += size * np.linalg.norm(Bc[0, 0]) * dlt / np.linalg.norm(dlt)
        return Bq
    # designated cases: the first val case accepted at the first-pass resolution (its accepted reference), and the first
    # such case in the padding population (code audit 2)
    fp = np.array(RG.case_labels(rv['d34'], rv['d12'], rv['dt'], rv['dp'])['eligible'])
    i0 = int(np.argmax(fp)) if fp.any() else None
    if i0 is None:
        mf_ref = dict(fires=False, reason='no val case accepted at the first-pass resolution')
    else:
        B0 = rv['B'][i0:i0 + 1]
        Bp0 = perturb(B0, 5 * RG.BAR_SPACE)
        d34p = np.maximum(RG.coef_diff(rv['B34'][i0:i0 + 1], Bp0),
                          RG.lat_diff(synth_cohort(rv['B34'][i0:i0 + 1], rv['N34']), synth_cohort(Bp0, N)))
        sl = slice(i0, i0 + 1)
        clean_ok = RG.case_labels(rv['d34'][sl], rv['d12'][sl], rv['dt'][sl], rv['dp'][sl])['eligible'][0]
        fault_ok = RG.case_labels(d34p, rv['d12'][sl], rv['dt'][sl], rv['dp'][sl])['eligible'][0]
        mf_ref = dict(case=i0, d34_clean=float(rv['d34'][i0]), d34_fault=float(d34p[0]), clean_eligible=bool(clean_ok),
                      fault_eligible=bool(fault_ok), margin=float(d34p[0] / RG.BAR_SPACE),
                      fires=bool(clean_ok and not fault_ok and d34p[0] >= 2 * RG.BAR_SPACE))
    pj = [k for k, i in enumerate(rv['pad_idx']) if fp[int(i)]]
    if not pj:
        mf_pad = dict(fires=False, reason='no padded case accepted at the first-pass resolution')
    j = int(rv['pad_idx'][pj[0]]) if pj else int(rv['pad_idx'][0])
    jp = pj[0] if pj else 0
    Bj = rv['B'][j:j + 1]
    Bpj = perturb(Bj, 3 * RG.BAR_PAD)
    dpp = np.maximum(RG.coef_diff(rv['Bp'][jp:jp + 1], Bpj),
                     RG.lat_diff(synth_cohort(rv['Bp'][jp:jp + 1], N), synth_cohort(Bpj, N)))
    c_ok = RG.case_labels(rv['d34'][j:j + 1], rv['d12'][j:j + 1], rv['dt'][j:j + 1], rv['dp'][j:j + 1])['eligible'][0]
    f_ok = RG.case_labels(rv['d34'][j:j + 1], rv['d12'][j:j + 1], rv['dt'][j:j + 1], dpp)['eligible'][0]
    if pj:
        mf_pad = dict(case=j, dp_clean=float(rv['dp'][j]), dp_fault=float(dpp[0]), clean_eligible=bool(c_ok),
                      fault_eligible=bool(f_ok), fires=bool(c_ok and not f_ok and dpp[0] >= 2 * RG.BAR_PAD))
    Uv = np.load(out / 'refs_val.npz')['U']
    zero = np.max(np.linalg.norm(Uv[:, 1:], axis=-1), axis=-1) / np.linalg.norm(Uv[:, 0], axis=-1)
    rec = dict(N_ref=N, splits=res, scorable=scorable,
               allowance=dict(U=U_allow, eps_max=float(np.max(np.array(eps)[np.array(el)])) if any(el) else None),
               G7=g7_all(a, D), MF_IC=dict(rel=mfic, fires=bool(mfic > 2e-8)),
               MF_REF_real=mf_ref, MF_PAD_chk=mf_pad,
               refgate_selftest=RG.selftest(), zero_baseline=dict(worst=float(zero.max()), median=float(np.median(zero))),
               steep_val=G.steep(D['val']).tolist(), env=env_record())
    (out / 'ref.json').write_text(json.dumps(rec, indent=1))
    log('ref U', U_allow, 'scorable', scorable, 'G7', rec['G7']['max'], 'MF-IC', mfic, 'MF-REF', mf_ref['fires'],
        'MF-PAD', mf_pad['fires'])


# ----------------------------------------------------------------------------------------------- stage data ---

def train_traj(N, D):
    n = SP.ref_cfl_steps(N)
    B, st = run_cohort(SP.make_fom(N, n, out='coef', n_out=20), D, chunk=4)
    assert int(np.sum(st[:, 0])) == 0, 'non-finite training trajectory'
    return B, n


def train_gate(N, D4, B4):
    n = SP.ref_cfl_steps(N)
    B34, _ = run_cohort(SP.make_fom(3 * N // 4, n, out='coef', n_out=20), D4, chunk=4)
    Bt, _ = run_cohort(SP.make_fom(N, 2 * n, out='coef', n_out=20), D4, chunk=4)
    U = synth_cohort(B4, N)
    ds = np.maximum(RG.coef_diff(B34, B4), RG.lat_diff(synth_cohort(B34, 3 * N // 4), U))
    dt = np.maximum(RG.coef_diff(Bt, B4), RG.lat_diff(synth_cohort(Bt, N), U))
    return dict(space=ds.tolist(), time=dt.tolist(), passed=bool(max(ds.max(), dt.max()) <= RG.TRAIN_BAR))


def bank_lattice(lev):
    return 512 if lev == 'A4' else 256


def stage_data(a, out, D):
    N = nref(a)
    rec = dict(N=N)
    sets = [('train', 'train_coef.npy')] + ([('train_c', 'trainc_coef.npy')] if a.trans else [])
    for key, fn in sets:
        t0 = time.perf_counter()
        B, n = train_traj(N, D[key])
        np.save(out / fn, B)
        g = train_gate(N, D[key][:4], B[:4])
        rec[key] = dict(n=n, shape=list(B.shape), seconds=time.perf_counter() - t0, gate=g, sha=sha(out / fn))
        log(f'data {key}: {B.shape} {rec[key]["seconds"]:.0f}s gate {g["passed"]} '
            f'(space {max(g["space"]):.1e}, time {max(g["time"]):.1e})')
    if a.bank:
        B = np.load(out / 'train_coef.npy', mmap_mode='r')
        lat = bank_lattice(a.lev)
        S = B.reshape(-1, N - 1, N - 1)
        f = jax.jit(jax.vmap(SP.synth_fn(N, lat)))
        Ub = np.concatenate([np.asarray(f(jnp.asarray(S[s:s + 16]))) for s in range(0, len(S), 16)])
        X = G.lattice('B', lat)[0]
        inner = np.all((X > 0) & (X < 1), axis=1)            # interior sample points only (A1.3)
        Ub, X = Ub[:, inner], X[inner]
        np.save(out / 'bank_U.npy', Ub)
        np.save(out / 'bank_X.npy', X)
        rec['bank_samples'] = dict(lattice=lat, states=int(Ub.shape[0]), points=int(Ub.shape[1]),
                                   sha_U=sha(out / 'bank_U.npy'))
    rec['env'] = env_record()
    (out / 'data.json').write_text(json.dumps(rec, indent=1))


# ------------------------------------------------------------------------------------------------ stage pod ----

def snap64(B):
    S = B.reshape(-1, B.shape[-2], B.shape[-1])
    return np.asarray(S[np.linspace(0, len(S) - 1, 64).astype(int)])


def stage_pod(a, out, D):
    rec = {}
    sets = [('pod', 'train_coef.npy')] + ([('podc', 'trainc_coef.npy')] if a.trans else [])
    for key, fn in sets:
        t0 = time.perf_counter()
        B = np.load(out / fn)
        S = B.reshape(-1, B.shape[-2], B.shape[-1])
        Phi, lam = PS.pod(S)
        np.save(out / f'{key}_Phi.npy', Phi)
        ks = (2, 4) if a.smoke else KQ_LADDER
        kq, krec = PS.choose_kq(snap64(B), Phi, ks=ks)            # all retained modes (code audit 1)
        capped = kq is None
        if kq is None:
            kq = ks[-1]
        Kex = PS.exact_K(Phi)
        sp16 = PS.pod_space(Phi, kq, 16, name=key)
        dK = float(np.linalg.norm(sp16.K - Kex) / np.linalg.norm(Kex))
        Z = PS.project(S, Phi)
        rec[key] = dict(R=int(len(Phi)), lam=lam[:600].tolist(), k_q=kq, k_q_search=krec, k_q_capped=capped, M_dev=sp16.meta['M_dev'],
                        K_rel=dK, M_rel=sp16.meta['M_dev'] / np.sqrt(len(Phi)),
                        M_pass=bool(sp16.meta['M_dev'] / np.sqrt(len(Phi)) <= 1e-8), K_pass=bool(dK <= 1e-8),
                        operator_ok=bool(sp16.meta['M_dev'] / np.sqrt(len(Phi)) <= 1e-8 and dK <= 1e-8),   # A8.1
                        seconds=time.perf_counter() - t0, sha=sha(out / f'{key}_Phi.npy'))
        np.save(out / f'{key}_Z.npy', Z)
        del sp16
        log(f'pod {key}: R {len(Phi)} k_q {kq} M_dev {rec[key]["M_dev"]:.1e} K_rel {dK:.1e} '
            f'[{rec[key]["seconds"]:.0f}s]')
    import gops
    rec['G_OPS'] = gops.run()
    rec['env'] = env_record()
    (out / 'pod.json').write_text(json.dumps(rec, indent=1))
    if not rec['G_OPS']['passed']:
        raise SystemExit('G-OPS failed: engine operators disagree with the exact integrals (release blocked, A3.2)')


# ----------------------------------------------------------------------------------------------- stage bank ---

def stage_bank(a, out, D):
    import bank as BK
    U = np.load(out / 'bank_U.npy')
    X = np.load(out / 'bank_X.npy')
    steps = int(os.environ.get('IMPBOX_BANK_STEPS', 2000)) if a.smoke else None
    seed = 7_400_000 + 10 * G.W_ID[a.W] + G.LEV_ID[a.lev]
    params, Z, info = BK.train('B', X, U, seed, steps=steps, log=log)
    B = np.load(out / 'train_coef.npy', mmap_mode='r')
    snaps = snap64(B)
    ks = (2, 4) if a.smoke else KQ_LADDER
    krec, kq = [], None
    for k in ks:                                       # A2.3: Parseval on q=12 and the bank Gram criterion
        p = PS.parseval_check(snaps, k, 12)
        g = None
        if p <= 1e-8 or k == ks[-1]:
            Ms = []
            for qq in (12, 16):
                Xq, wq = G.composite_rule('B', k, qq)
                pj = jax.tree_util.tree_map(jnp.asarray, params)
                F = np.concatenate([np.asarray(BK.features('B', pj, jnp.asarray(Xq[s:s + 16384])))
                                    for s in range(0, len(Xq), 16384)])
                Fj = jnp.asarray(F)
                Ms.append(np.asarray(jnp.matmul((Fj * jnp.asarray(wq)[:, None]).T, Fj, precision='highest')))
            g = float(np.linalg.norm(Ms[0] - Ms[1]) / np.linalg.norm(Ms[1]))
        krec.append(dict(k_q=k, parseval=p, gram=g))
        if p <= 1e-8 and g is not None and g <= 1e-8:
            kq = k
            break
    capped = kq is None
    if kq is None:
        kq = ks[-1]
    Xq, wq = G.composite_rule('B', kq, 16)
    T, sv, condR, dropped = BK.rotation('B', params, Z, Xq, wq)
    h1 = BK.save(out / 'bank.pkl', dict(params=params, Z=Z, T=T, k_q=kq, q_ref=16, dom='B', info=info))
    rec = dict(info=info, sv=sv[:600].tolist(), condR=condR, dropped_codes=dropped, R_eff=int(T.shape[1]), sha=h1,
               seed=seed, k_q=kq, k_q_search=krec, k_q_capped=capped, env=env_record())
    (out / 'bank.json').write_text(json.dumps(rec, indent=1))
    log('bank', info['seconds'], 'recon', info['recon_mean'], info['recon_max'], 'R_eff', T.shape[1], 'k_q', kq)


# ------------------------------------------------------------------------------------------------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True, choices=list(G.WIDTHS))
    ap.add_argument('--lev', required=True, choices=list(G.LEVELS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--stages', default='ref,data,pod,bank,ladder,time')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n_train', type=int, default=None)
    ap.add_argument('--nref_override', type=int, default=None)
    ap.add_argument('--frozen', default=None)
    ap.add_argument('--no_bank', action='store_true', help='debug only: skip the bank arm')
    a = ap.parse_args()
    a.bank = a.W in BANK_CELLS_W and not a.no_bank
    dbg = [k for k in ('IGEO_NCAP', 'IMPBOX_TINY', 'IMPBOX_BANK_STEPS') if os.environ.get(k)]
    if dbg and not a.smoke:
        raise SystemExit(f'debug overrides {dbg} are not allowed in a full-size run')
    a.trans = a.W in TRANS_W
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # imp-box A9: persistent compilation cache inside the staged code dir (excluded from collection and archives); rebuilt
    # queries of an identical configuration load their executable instead of recompiling
    cache = Path(os.environ.get('IMPBOX_JAX_CACHE', HERE / '.jaxcache')).resolve()
    if cache != (HERE / '.jaxcache').resolve() and not os.environ.get('IMPBOX_LOCAL_DEBUG'):
        raise SystemExit('the compilation cache must be the job-local code/.jaxcache (excluded from collection)')
    cache.mkdir(parents=True, exist_ok=True)
    import shutil
    free = shutil.disk_usage(cache).free
    if free < 200 * 2 ** 30 and not a.smoke:
        raise SystemExit(f'less than 200 GiB free at the cache location ({free / 2 ** 30:.0f} GiB)')
    jax.config.update('jax_compilation_cache_dir', str(cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_compilation_cache_max_size', 30 * 2 ** 30)          # bytes; LRU eviction beyond
    log('env', env_record())
    D = cohorts(a)
    for st in a.stages.split(','):
        mk = out / f'.done_{st}'
        if mk.exists():
            log(f'stage {st}: done ({mk.name})')
            continue
        if st == 'bank' and not a.bank:
            continue
        if st in ('ladder', 'time', 'controls') and (out / 'ref.json').exists() and \
                not json.loads((out / 'ref.json').read_text()).get('scorable', False):
            (out / 'final.json').write_text(json.dumps(dict(scorable=False, selection={}, S={})))
            log(f'stage {st}: cell UNSCORABLE (selection references failed the A2.2 policy); no selection')
            raise SystemExit(3)                        # explicit incomplete status (job audit 1)
        log(f'stage {st}: start')
        t0 = time.perf_counter()
        try:
            run_stage(a, out, D, st)
        finally:                                       # peaks are recorded on failure too (code audit 4)
            record_peak(out, st, t0)
        mk.write_text('ok')
        log(f'stage {st}: {time.perf_counter() - t0:.0f}s')


def run_stage(a, out, D, st):
    if True:
        if st in ('ladder',):
            import ladders
            ladders.stage_ladder(a, out, D)
        elif st == 'time':
            import timing
            timing.stage_time(a, out, D)
        elif st == 'controls':
            import controls
            controls.stage_controls(a, out, D)
        elif st == 'test':
            import teststage
            teststage.stage_test(a, out, D)
        else:
            dict(ref=stage_ref, data=stage_data, pod=stage_pod, bank=stage_bank)[st](a, out, D)


def record_peak(out, st, t0):
    dev_peak = None                            # None = unavailable (distinct from a measured zero)
    try:
        dev_peak = int(jax.devices()[0].memory_stats().get('peak_bytes_in_use'))
    except Exception:                          # noqa: BLE001
        pass
    import resource
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024          # bytes (Linux: KiB)
    with (out / 'stage_seconds.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage=st, seconds=time.perf_counter() - t0, device_peak_bytes=dev_peak,
                                host_peak_bytes=int(host_peak))) + '\n')


if __name__ == '__main__':
    main()
