"""Offline acceptance of the phase-I smoke (DESIGN.md 10; A1.9, A2.9-A2.11, A3.2-A3.3, A3.9, A4.1-A4.4, A4.9, A5.1, A5.4,
A6-A11). Every predicate is recomputed from raw evidence: per-step certification lists (gamma, exit reason, LM damping)
checked against the registered step counts, saved coefficient trajectories (finiteness), saved lattice fields
(distances, E0 scores against freshly loaded pinned references, swap controls), raw timing attempts (repeat/retry
inventory, drift, every per-output discrepancy), projection errors, and the CPU gate file bound to the exact source
inventory. The frozen A5.4 inventory is validated, not read from the output. A required item that is missing, skipped,
errored, non-finite or unavailable is NOT accepted; unavailable fixtures are 'unfulfilled'. Writes ACCEPTANCE.json.

Usage: JAX_PLATFORMS=cpu python smoke_accept.py <out_dir> <job log> <staged lane dir or this lane dir>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REQUIRED_PHASES = ('refs', 'G-fom', 'basis', 'adequacy', 'generator_probe_cost', 'adv_pod', 'qdeim_max_shape',
                   'ecsw_states', 'ecsw_smoke_rank', 'ecsw_max_shape', 'none_max_shape', 'grid', 'G-fd', 'G-all',
                   'N-under', 'N-grad', 'N-swap', 'N-time+ours')
KINDS_WANT = {'3|S|BE|0.005': 'raised', '3|S|BE|0.01': 'raised', '2|S|CNR|0.005': 'raised', '2|S|BE|0.01': 'raised',
              '2|S|BE|0.005': 'allowed', '2|ST|CNR|0.005': 'allowed'}
LEVEL_GTOL = {'prod': 1e-3, 'tight': 1e-7, 'tighter': 1e-8}
KNOWN_REASONS = {0, 1, 2, 3, 4, 5}
CPU_GATES = ('provenance-train-draw-2d', 'disjoint-train-cohorts', 'G-fd-a operators vs refcore',
             'G-fd-b online Jacobians (FD + jacfwd), every arm', 'N-fd mutant Jacobian fails the FD check',
             'G-fd-c ECSW rows and Q-DEIM objective vs independent assembly', 'FOM (hrcore) vs refcore.make_solver, BE and CN',
             'G-lmm step equations (refcore operators) and temporal order, every scheme',
             'G-full complete basis reproduces the FOM (every arm, option, solver)',
             'G-nnls greedy NNLS vs scipy (objective, KKT, tau stop)', 'G-qdeim pivots vs scipy, exact interpolation on span(U)',
             'G-pod orthonormality, snapshot reproduction and POD optimality',
             'N-dom decision table on synthetic data (A2.11, A3.3, A4.2; hand-derived labels)')
CPU_SOURCES = ('hrcore.py', 'rom.py', 'build.py', 'nnls.py', 'stats.py', 'pipeline.py', 'tests/test_core.py')
# A5.4 frozen smoke inventory (+ pilot_support, A11)
FROZEN = {2: dict(L=1024, n_traj=192, rmax=512, pmax=2048, rs=128, ps=256, kappas=[2, 4], kappa_run=4, cases=[0, 1],
                  under_fixtures=[2, 3], n_adequacy=8, n_probe=4, options=[['BE', 0.005], ['CNR', 0.005]], pilot_support=1024),
          3: dict(L=64, n_traj=342, rmax=1024, pmax=4096, rs=256, ps=512, kappas=[2, 4], kappa_run=4, cases=[0, 1],
                  under_fixtures=[5, 0], n_adequacy=8, n_probe=4, options=[['BE', 0.01], ['CN', 0.01], ['BDF2', 0.025]],
                  pilot_support=1024)}
DT0 = {2: 0.005, 3: 0.01}


def fin(*xs):
    try:
        return all(x is not None and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def cert_ok(v):
    """A6.1 certificate recomputed from (frob_rel, pod_tail_sq)."""
    t2, e = v.get('pod_tail_sq'), v.get('frob_rel')
    if not fin(t2, e) or t2 < -1e-12:
        return False
    tau = math.sqrt(max(t2, 0.))
    return e <= 1e-6 if tau < 1e-6 else abs(e - tau) <= 0.01 * tau + 1e-6


def nsteps(dt):
    return int(round(0.25 / dt))


def run_evidence(cert, level, steps_want, C, r_want):
    """(valid, nonstat, steps, anchor_ok) from the per-step lists and the saved trajectory (A2.7, A6.5, A9.1, A11.1).
    Shapes are validated: 1-D lists of the registered step count, integer reasons, C of shape (steps + 1, r)."""
    g = np.asarray(cert['gamma_list'], float)
    rraw = np.asarray(cert['reason_list'])
    lam = np.asarray(cert['lam_list'], float)
    C = np.asarray(C, float)
    ints = rraw.ndim == 1 and np.all(np.equal(np.mod(rraw.astype(float), 1), 0))
    rsn = rraw.astype(int) if ints else np.full(rraw.shape, -1)
    ok_len = (g.ndim == 1 and lam.ndim == 1 and ints and len(g) == len(rsn) == len(lam) == steps_want and steps_want > 0
              and C.ndim == 2 and C.shape == (steps_want + 1, r_want))
    known = bool(set(rsn.tolist()) <= KNOWN_REASONS)
    cfin = bool(np.all(np.isfinite(C)))
    damp = bool(np.all(np.isfinite(lam)) and np.all(lam < 1e14))
    valid = bool(ok_len and known and cfin and damp and not np.any(rsn == 3) and not np.any(rsn == 5))
    stationary = np.isfinite(g) & (g <= 1e-3)
    gl = LEVEL_GTOL[level]
    stall = (rsn == 3) & np.isfinite(g) & (g <= 100. * gl)
    anchor_ok = bool(ok_len and known and cfin and damp and not np.any(rsn == 5)
                     and np.all((np.isfinite(g) & (g <= gl)) | stall) and np.all((rsn != 3) | stall))
    return valid, int(np.sum(~stationary)), len(g), anchor_ok


def q1_from_evidence(prod, tight, tighter, Cs, dist, adist, cases, subset, steps_want, r_want):
    """prod/tight/tighter: case -> cert; Cs: dict(prod=..., tight=..., tighter=...) case -> trajectory."""
    if sorted(prod) != sorted(cases) or sorted(tight) != sorted(cases) or sorted(tighter) != sorted(subset):
        return False, 'coverage'
    if sorted(dist) != sorted(cases) or sorted(adist) != sorted(subset) or not all(fin(v) for v in list(dist.values()) + list(adist.values())):
        return False, 'distance coverage or non-finite distance'
    ev = [run_evidence(prod[c], 'prod', steps_want, Cs['prod'][c], r_want) for c in cases]
    valid = all(e[0] for e in ev)
    ns, st = sum(e[1] for e in ev), sum(e[2] for e in ev)
    a_ok = all(run_evidence(tight[c], 'tight', steps_want, Cs['tight'][c], r_want)[3] for c in cases) and \
        all(run_evidence(tighter[c], 'tighter', steps_want, Cs['tighter'][c], r_want)[3] for c in subset)
    ok = bool(valid and st > 0 and ns <= 0.01 * st and a_ok and all(fin(v) and v <= 2.5e-4 for v in dist.values())
              and all(fin(v) and v <= 2.5e-5 for v in adist.values()))
    return ok, dict(valid=valid, nonstat=ns, steps=st, anchors=a_ok, max_dist=max(dist.values()))


def _good(t, keys=('tA1', 'tB', 'tA2')):
    return all(fin(t[k]) and t[k] > 0 for k in keys) and abs(t['tA1'] / t['tA2'] - 1.) <= 0.25


def timing_ok(cases_rec, card=(1, 1, 1), reps=3, max_retry=3, tol=1e-12):
    """Repeat/retry inventory, drift recomputed from raw times, every per-output discrepancy finite and <= tol.
    Returns (ok, rho: case -> valid ratios, attempts)."""
    rho, ok, attempts = {}, True, 0
    for j, v in cases_rec.items():
        tr = v['triples']
        attempts += len(tr)
        valid_ratios = []
        if sorted({t['rep'] for t in tr}) != list(range(reps)):
            ok = False
        for rep in range(reps):
            at = [t for t in tr if t['rep'] == rep]
            if not at or [t['attempt'] for t in at] != list(range(len(at))) or len(at) > max_retry + 1:
                ok = False
                continue
            for t in at:
                if _good(t) != t['valid']:
                    ok = False
                for kk, nc in zip(('dA1', 'dB', 'dA2'), card):
                    ds = t[kk] if isinstance(t[kk], list) else [t[kk]]
                    if len(ds) != nc or not all(fin(x) and x <= tol for x in ds):
                        ok = False
            if any(_good(t) for t in at[:-1]):          # a valid attempt must end its repeat
                ok = False
            if not _good(at[-1]) and len(at) != max_retry + 1:   # an unsuccessful repeat must exhaust its retries
                ok = False
            if _good(at[-1]):
                valid_ratios.append(at[-1]['tB'] / (0.5 * (at[-1]['tA1'] + at[-1]['tA2'])))
        rho[int(j)] = valid_ratios
    return ok, rho, attempts


def load_chunks(d):
    out = {}
    if d.is_dir():
        for f in sorted(d.glob('*.npz')):
            out[f.stem] = np.load(f)
    return out


def main():
    out = Path(sys.argv[1])
    logp = Path(sys.argv[2])
    lane = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE
    S = json.loads((out / 'smoke.json').read_text())
    acc = {}

    def put(k, ok, **ev):
        acc[k] = dict(ok=bool(ok), **ev)

    txt = logp.read_text() if logp.exists() else ''
    put('backend=gpu', 'jax_backend=gpu' in txt)
    put('disjoint', S.get('disjoint') == [True, True])
    try:
        cg = json.loads((lane / 'checks' / 'test_core.json').read_text())
        names_ok = sorted(cg['gates']) == sorted(CPU_GATES) and all(cg['gates'][n][0] == 'PASS' for n in CPU_GATES)
        src_ok = sorted(cg['source_sha256']) == sorted(CPU_SOURCES) and all(
            hashlib.sha256((lane / f).read_bytes()).hexdigest() == cg['source_sha256'][f] for f in CPU_SOURCES)
        put('cpu-gates', cg.get('all_pass') is True and names_ok and src_ok, names_ok=names_ok, sources_match=src_ok)
    except Exception as e:  # noqa: BLE001
        put('cpu-gates', False, error=str(e))
    cfg_cells = {c['d']: c for c in S['config']['cells']}
    if os.environ.get('HRB_ACCEPT_DRY') == '1':      # local CPU dry run only: never a binding acceptance
        for d_ in (2, 3):
            if d_ in cfg_cells:
                FROZEN[d_] = {k: cfg_cells[d_][k] for k in FROZEN[d_]}
        acc['DRY-RUN (not binding)'] = dict(ok=False)
    put('frozen-inventory', sorted(cfg_cells) == [2, 3] and all(
        all(cfg_cells[d].get(k) == v for k, v in FROZEN[d].items()) for d in (2, 3)),
        diffs={d: {k: (cfg_cells.get(d, {}).get(k), v) for k, v in FROZEN[d].items() if cfg_cells.get(d, {}).get(k) != v}
               for d in (2, 3)})
    import hrcore as H
    import score as SC
    import stats as ST
    for d in (2, 3):
        c = dict(FROZEN[d], d=d)
        L = c['L']
        key = f'{d}d_L{L}'
        R = S.get(key)
        if R is None:
            put(f'{key}|present', False)
            continue
        n = (L - 1) ** d
        Z = load_chunks(out / f'arrays_{key}')
        coh = 'dev6' if d == 2 else 'val64'
        cases, subset, fxs = c['cases'], c['cases'][:2], sorted(c['under_fixtures'])
        try:
            famc = H.Family(d, coh, count=None if d == 2 else max(cases + fxs) + 1)
            refs = SC.Refs(d, coh, famc, sorted(set(cases + fxs + [0])))
        except Exception as ex:  # noqa: BLE001
            put(f'{key}|refs', False, error=str(ex))
            continue
        n0 = {j: refs.n0(j) for j in refs.f}
        saved_n0 = {int(k): v for k, v in (R.get('n0') or {}).items()}
        put(f'{key}|n0', all(j in saved_n0 and abs(saved_n0[j] - n0[j]) <= 1e-12 * n0[j] for j in n0))
        ph = R.get('phases', {})
        miss = [p for p in REQUIRED_PHASES if ph.get(p, {}).get('status') != 'ok']
        put(f'{key}|phases', not miss, missing_or_failed={p: ph.get(p) for p in miss},
            peaks={k: (v.get('device_peak'), v.get('host_maxrss'), v.get('seconds')) for k, v in ph.items()})
        g = R.get('G-fom') or {}
        put(f'{key}|G-fom', set(g) == {'BE', 'CN'} and all(fin(v['max_diff'], v['norm0']) and v['norm0'] > 0 and
                                                          v['max_diff'] <= 1e-10 * v['norm0'] for v in g.values()), ev=g)
        b = R.get('basis') or {}
        proj = b.get('proj') or {}
        dev = (b.get('gate') or {}).get('orthonormality')
        ranks = [str(r) for r in (c['rs'], c['rmax'])]
        put(f'{key}|G-pod', all(r in proj and cert_ok(proj[r]) for r in ranks) and fin(dev) and dev <= 1e-10,
            orthonormality=dev, ranks={r: proj.get(r) for r in ranks})
        a = R.get('adv') or {}
        ap = a.get('proj') or {}
        ladder, p = [], c['rs'] // 8
        while p >= 1:
            ladder.append(p)
            p //= 2
        plist = [str(p) for p in sorted(set([c['ps']] + ladder))]       # A14.1: p_max is a shape pilot, reported only
        put(f'{key}|G-pod(adv)', all(p in ap and cert_ok(ap[p]) for p in plist) and fin(a.get('orthonormality'))
            and a['orthonormality'] <= 1e-10, p=plist)
        q = R.get('qdeim_pilot') or {}
        put(f'{key}|qdeim-max-shape', fin(q.get('solve_backward_error'), q.get('device_peak')) and
            q['solve_backward_error'] <= 1e-12 and q.get('B_shape') == [n, 2 * c['rmax'] + c['pmax']] and q['device_peak'] > 0,
            ev={k: q.get(k) for k in ('B_shape', 'interp_const', 'solve_backward_error', 'seconds', 'device_peak')})
        e = R.get('ecsw_rs') or {}
        # A16: retry trigger and path selection recomputed; supports counted from the saved weights
        try:
            ps_ = R['ecsw_path_summary']
            zp = Z['ECSWpath']
            atts = ps_['attempts']
            ncand = ps_['n_cand']
            need_retry = atts['initial']['banned'] > ps_['ban_retry_frac'] * ncand
            ok_e = ('retry' in atts) == need_retry and atts['initial']['max_drops'] == 3 and \
                (('retry' not in atts) or atts['retry']['max_drops'] >= 10 ** 9)
            for a, sm in atts.items():
                for kk, cp in sm['checkpoints'].items():
                    if cp is None:
                        ok_e &= f'{a}~{kk}~weights' not in zp
                        continue
                    w = zp[f'{a}~{kk}~weights']
                    ok_e &= int(np.sum(w > 0)) == cp['size'] >= int(kk[1:]) * c['rs'] and np.all(np.isfinite(w)) \
                        and len(set(zp[f'{a}~{kk}~nodes'].tolist())) == len(w)
                ok_e &= sm['reached'] == sum(v is not None for v in sm['checkpoints'].values())
            sel = 'initial'
            if 'retry' in atts:
                s0, s1 = atts['initial'], atts['retry']
                if s1['reached'] > s0['reached']:
                    sel = 'retry'
                elif s1['reached'] == s0['reached']:
                    com = [k for k in s0['checkpoints'] if s0['checkpoints'][k] and s1['checkpoints'][k]]
                    l0 = sum(math.log10(max(s0['checkpoints'][k]['relres'], 1e-300)) for k in com)
                    l1 = sum(math.log10(max(s1['checkpoints'][k]['relres'], 1e-300)) for k in com)
                    sel = 'retry' if l1 < l0 else 'initial'
            ok_e &= sel == ps_['selected']
            ok_e &= all(e.get(f'k{k}') is not None and e[f'k{k}']['size'] == atts[sel]['checkpoints'][f'k{k}']['size']
                        for k in (2, 4, 8, 16))
            put(f'{key}|ecsw-smoke-rank', ok_e, selected=ps_['selected'], retry=need_retry,
                reached={a: sm['reached'] for a, sm in atts.items()}, supports=e)
        except Exception as ex:  # noqa: BLE001
            put(f'{key}|ecsw-smoke-rank', False, error=f'{type(ex).__name__}: {ex}')
        ep = R.get('ecsw_pilot') or {}
        nst = (R.get('ecsw_states') or {}).get('n_states')
        shp = list(ep.get('shape', [0, 0]))
        kk_ = ep.get('kkt') or {}
        sup_ok = (ep.get('support') == c['pilot_support'] and ep.get('reason') == 'cap') or \
            (ep.get('reason') == 'kkt' and isinstance(ep.get('support'), int) and ep['support'] <= c['pilot_support']
             and kk_.get('ok') is True and fin(kk_.get('support_max_abs_gradient'), kk_.get('tol'))
             and kk_['support_max_abs_gradient'] <= kk_['tol'])
        put(f'{key}|ecsw-max-shape', fin(ep.get('relres'), ep.get('device_peak')) and ep['device_peak'] > 0 and
            nst is not None and shp[0] > 0 and shp[0] % c['rmax'] == 0 and shp[0] <= nst * c['rmax'] and
            shp[1] == min(32768, n) and sup_ok, ev=ep, n_states=nst)
        npil = R.get('none_pilot') or {}
        put(f'{key}|none-max-shape', fin(npil.get('seconds'), npil.get('device_peak')) and npil['device_peak'] > 0 and
            npil.get('J_shape') == [n, c['rmax']] and npil.get('W_finite') is True and npil.get('C_finite') is True
            and npil.get('fields_finite') is True,
            ev={k: npil.get(k) for k in ('seconds', 'device_peak', 'J_shape', 'W_finite', 'C_finite', 'fields_finite')})
        # ---- grid: Q1 recomputed from per-step lists + saved trajectories + recomputed distances; E0 rescored
        grid = R.get('grid') or {}
        expect = sorted(f'{k}|{c["rs"]}|{h}|{o[0]}|{o[1]:.8g}|{s}|central' for k, h in (('none', '-'), ('ecsw', f"k{c['kappa_run']}"),
                        ('qdeim', c['ps'])) for o in c['options'] for s in ('native', 'common'))
        q1s, bad, worst_dec, worst_sc, n_dec, n_invalid = {}, [], 0., 0., 0, 0
        Pl = Z['basis']['Phi_lat'] if 'basis' in Z else None
        for cid in expect:
            try:
                zz = Z['grid~' + cid.replace('|', '~')]
                gg = grid[cid]
                sw = nsteps(float(cid.split('|')[4]))
                F, Ft, Ftt = zz['F'], zz['Ftight'], zz['Ftighter']
                dist = {j: SC.lattice_distance(F[i], Ft[i], n0[j]) for i, j in enumerate(cases)}
                adist = {j: SC.lattice_distance(Ft[i], Ftt[i], n0[j]) for i, j in enumerate(subset)}
                prod = {int(k): v['cert'] for k, v in gg['runs'].items()}
                anc = {kk: {int(j): vv for j, vv in v.items()} for kk, v in gg['anchors'].items()}
                Cs = dict(prod={j: zz['C'][i] for i, j in enumerate(cases)}, tight={j: zz['Ctight'][i] for i, j in enumerate(cases)},
                          tighter={j: zz['Ctighter'][i] for i, j in enumerate(subset)})
                ok, ev = q1_from_evidence(prod, anc['tight'], anc['tighter'], Cs, dist, adist, cases, subset, sw, c['rs'])
                keep = int(round(0.05 / float(cid.split('|')[4])))
                for i, j in enumerate(cases):              # saved outputs W are the scheduled rows of the trajectory C
                    vj = run_evidence(prod[j], 'prod', sw, Cs['prod'][j], c['rs'])[0]
                    if zz['W'][i].shape != (6, c['rs']) or not np.array_equal(zz['W'][i], zz['C'][i][::keep][:6],
                                                                              equal_nan=not vj):
                        bad.append(f'{cid}: W inconsistent with C rows (case {j})')
                q1s[cid] = ok
                if ok != gg['q1']['q1']:
                    bad.append(f'{cid}: in-job q1 {gg["q1"]["q1"]} != recomputed {ok}')
                for i, j in enumerate(cases):
                    tv = (gg['runs'].get(str(j)) or gg['runs'].get(j))['timed_vs_scored']
                    if not (fin(tv) and tv <= 1e-12):
                        bad.append(f'{cid}: timed != scored ({tv})')
                    W = zz['W'][i]
                    if Pl is None:
                        bad.append(f'{cid}: lattice basis missing')
                        continue
                    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(F[i]))):
                        vj = run_evidence(prod[j], 'prod', sw, Cs['prod'][j], c['rs'])[0]
                        e_rec = (gg['e0'].get(str(j)) or gg['e0'].get(j))['e']
                        if vj or fin(e_rec):          # a non-finite run must be invalid and carry no usable score
                            bad.append(f'{cid}: non-finite run classified valid or scored (case {j})')
                        else:
                            n_invalid += 1
                        continue
                    n_dec += 1
                    dec = H.to_lattice(W @ Pl.T, d)
                    worst_dec = max(worst_dec, float(np.max(np.abs(dec - F[i]))) / max(float(np.max(np.abs(F[i]))), 1e-300))
                    ej = refs.score(dec, j)['e']
                    rec = (gg['e0'].get(str(j)) or gg['e0'].get(j))['e']
                    if not fin(ej, rec):
                        bad.append(f'{cid}: non-finite score case {j}')
                        continue
                    worst_sc = max(worst_sc, abs(ej - rec) / max(rec, 1e-300))
            except Exception as ex:  # noqa: BLE001
                bad.append(f'{cid}: {type(ex).__name__}: {ex}')
        put(f'{key}|grid-Q1-machinery', not bad and len(q1s) == len(expect), expected=len(expect), q1=q1s, problems=bad)
        put(f'{key}|G-score', not bad and len(q1s) == len(expect) and n_dec + n_invalid == len(expect) * len(cases)
            and n_dec > 0 and worst_dec <= 1e-12 and worst_sc <= 1e-12, decode=worst_dec, score=worst_sc, n_decoded=n_dec,
            n_invalid=n_invalid)
        sw_main_ = nsteps(DT0[d])
        # ---- G-fd
        fd = R.get('G-fd') or {}
        ga = fd.get('G-fd-a') or {}
        okfd = fin(ga.get('rows'), ga.get('jac')) and ga['rows'] <= 1e-12 and ga['jac'] <= 1e-11
        want_arms = {f'{k}|{m}' for k in ('none', 'ecsw', 'qdeim') for m in ('None', 'drop_adv_jac')}
        okfd &= {k for k in fd if k != 'G-fd-a'} == want_arms
        for k in want_arms & set(fd):
            v = fd[k]
            if k.endswith('|None'):
                okfd &= fin(v.get('fd'), v.get('jacfwd'), v.get('res_vs_rJ')) and v['fd'] <= 1e-6 and v['jacfwd'] <= 1e-11 \
                    and v['res_vs_rJ'] <= 1e-14
                if k.startswith('ecsw'):
                    okfd &= fin(v.get('rows')) and v['rows'] <= 1e-10
                if k.startswith('qdeim'):
                    okfd &= all(fin(v.get(f'{m}_{t}')) and v[f'{m}_{t}'] <= 1e-10 for m in ('norm', 'grad')
                                for t in ('generic', 'cancel'))
            else:
                okfd &= fin(v.get('fd')) and v['fd'] > 1e-3
        put(f'{key}|G-fd+N-fd', okfd, ev=fd)
        gal = R.get('G-all') or {}
        try:
            za = Z['G-all']
            dga = SC.lattice_distance(za['Fa'], za['Fb'], n0[cases[0]])
            va = run_evidence(gal['cert_a'], 'tight', sw_main_, za['Ca'], c['rs'])[0]
            vb = run_evidence(gal['cert_b'], 'tight', sw_main_, za['Cb'], c['rs'])[0]
            put(f'{key}|G-all', fin(dga) and dga <= 1e-10 and va and vb, dist=dga, valid=[va, vb])
        except Exception as ex:  # noqa: BLE001
            put(f'{key}|G-all', False, error=f'{type(ex).__name__}: {ex}')
        # ---- N-under: exact ladder inventory; Q1 and E0 recomputed from saved fields
        nu_ = R.get('N-under') or {}
        sw_main = nsteps(DT0[d])
        want_tags = ['none'] + [f'ecsw_cap{int(math.ceil(m * c["rs"]))}' for m in (3., 2.5, 2., 1.5, 1.25)] + \
            [f'qdeim_p{p}' for p in ladder]
        levels, base, inv_ok = {}, None, sorted(nu_) == sorted(want_tags)
        for tag in want_tags:
            v = nu_.get(tag)
            if v is None:
                continue
            if 'construction_failure' in v:
                levels[tag] = dict(construction_failure=v['construction_failure'])
                continue
            try:
                zz = Z['NU~' + tag]
                dist = {j: SC.lattice_distance(zz[f'{j}~F'], zz[f'{j}~Ftight'], n0[j]) for j in fxs}
                adist = {j: SC.lattice_distance(zz[f'{j}~Ftight'], zz[f'{j}~Ftighter'], n0[j]) for j in fxs}
                Cs = dict(prod={j: zz[f'{j}~C'] for j in fxs}, tight={j: zz[f'{j}~Ctight'] for j in fxs},
                          tighter={j: zz[f'{j}~Ctighter'] for j in fxs})
                ok, ev = q1_from_evidence({int(j): x for j, x in v['cert'].items()}, {int(j): x for j, x in v['tight'].items()},
                                          {int(j): x for j, x in v['tighter'].items()}, Cs, dist, adist, fxs, fxs, sw_main, c['rs'])
                if tag.startswith('ecsw') and not (isinstance(v.get('support'), int) and v['support'] > c['rs']):
                    ok = False
                es = [refs.score_safe(zz[f'{j}~F'], j)['e'] for j in fxs]
                med = float(np.median(es)) if all(fin(x) for x in es) else float('nan')
                levels[tag] = dict(q1=ok, median=med, support=v.get('support'))
                if tag == 'none':
                    base = med if ok else None
            except Exception as ex:  # noqa: BLE001
                levels[tag] = dict(error=f'{type(ex).__name__}: {ex}')
        for f in ('ecsw', 'qdeim'):
            lv = {k: x for k, x in levels.items() if k.startswith(f)}
            broken = [k for k, x in levels.items() if 'error' in x]
            fired = inv_ok and not broken and base is not None and any(x.get('q1') and fin(x.get('median')) and x['median'] >= 2 * base
                                                        for x in lv.values())
            put(f'{key}|N-under[{f}]', fired, status='fired' if fired else ('evidence broken' if broken else 'unfulfilled'),
                inventory_ok=inv_ok, broken=broken, none_median=base, levels=lv)
        # ---- N-grad: inventory, oracle agreement, shared certifier, production classification reconstructed
        ng = R.get('N-grad') or {}
        if ng.get('status') == 'run' and 'NG' in Z:
            tested = list(range(0, sw_main, 5))
            bg = np.asarray(Z['NG']['base_gamma'], float)
            tp = [t for t in ng['tuples'] if t.get('oracle') is not None]
            tu = [t for t in ng['tuples'] if t.get('oracle') is None]
            skipped = sorted(x['step'] for x in ng.get('skipped_steps', []))
            prod_set = {(k, sg) for k in range(7) for sg in (1., -1.)}
            inv = ng.get('tested_steps') == tested and sorted(t['step'] for t in tu) == tested and len(bg) == sw_main and \
                all(x.get('sigma1') == 0. for x in ng.get('skipped_steps', [])) and set(skipped) <= set(tested) and \
                all({(t['k'], float(t['sign'])) for t in tp if t['step'] == st} == (set() if st in skipped else prod_set)
                    for st in tested) and len(tp) == 14 * (len(tested) - len(skipped)) and \
                {(x['k'], float(x['sign'])) for x in (ng.get('classify') or [])} == prod_set
            base_ok = run_evidence(ng['base_cert'], 'prod', sw_main, Z['NG']['base_C'], c['rs'])[0]
            agree = all(fin(t['cert'], t['oracle']) and abs(t['cert'] - t['oracle']) <= 1e-9 * t['oracle'] + 1e-12 and
                        ((t['cert'] > 1e-3) == (t['oracle'] > 1e-3)) for t in tp)
            same = all(fin(t['cert']) and abs(t['cert'] - bg[t['step']]) <= 1e-9 * abs(bg[t['step']]) + 1e-12 for t in tu)
            cls_ok = len(ng.get('classify') or []) == 14
            for x in ng.get('classify') or []:
                o_var = bg.copy()
                for t in tp:
                    if t['k'] == x['k'] and t['sign'] == x['sign']:
                        o_var[t['step']] = t['oracle']
                want = int(np.sum(~(np.isfinite(o_var) & (o_var <= 1e-3))))
                cls_ok &= x['nonstat_production'] == want == x['nonstat_oracle']
            disc = [t for t in tp if t['oracle'] > 1e-3]
            status = 'pass' if (inv and base_ok and agree and same and cls_ok and disc) else (
                'fixture unavailable' if (inv and base_ok and agree and same and cls_ok) else 'fail')
            put(f'{key}|N-grad(a)', status == 'pass', status=status, inventory=inv, base_valid=base_ok, agree=agree,
                shared_vs_trajectory=same, classification=cls_ok, n_discriminating=len(disc))
            try:                                       # part (b): complete Q1 recomputed (reported; any outcome allowed)
                zb = Z['NGb']
                bc = ng['b_cert']
                distb = {j: SC.lattice_distance(zb[f'{j}~F'], zb[f'{j}~Ftight'], n0[j]) for j in fxs}
                adistb = {j: SC.lattice_distance(zb[f'{j}~Ftight'], zb[f'{j}~Ftighter'], n0[j]) for j in fxs}
                Csb = dict(prod={j: zb[f'{j}~C'] for j in fxs}, tight={j: zb[f'{j}~Ctight'] for j in fxs},
                           tighter={j: zb[f'{j}~Ctighter'] for j in fxs})
                qb, evb = q1_from_evidence({int(j): x for j, x in bc['prod'].items()}, {int(j): x for j, x in bc['tight'].items()},
                                           {int(j): x for j, x in bc['tighter'].items()}, Csb, distb, adistb, fxs, fxs,
                                           sw_main, c['rs'])
                put(f'{key}|N-grad(b) evidence complete', isinstance(evb, dict) and qb == ((ng.get('b') or {}).get('q1') or {}).get('q1'),
                    q1=qb, detail=evb)
            except Exception as ex:  # noqa: BLE001
                put(f'{key}|N-grad(b) evidence complete', False, error=f'{type(ex).__name__}: {ex}')
        else:
            put(f'{key}|N-grad(a)', False, status=ng.get('status', 'missing'), reason=ng.get('reason'))
        # ---- N-swap: recomputed from pinned references and saved fields
        sw = R.get('N-swap') or {}
        if sw:
            j0, j1 = cases[0], cases[1]
            eps = max(refs.score(refs.f[j0], j)['eps_ref'] for j in (j0, j1))
            Rf = refs.f[j0]
            rf = dict(true=refs.score(Rf, j0)['e'], case_swap=refs.score(Rf, j0, ref_case=j1)['e'],
                      time_shift=refs.score(Rf, j0, shift=1)['e'])
            ok_ref = abs(rf['case_swap'] - rf['true']) > 10 * eps and abs(rf['time_shift'] - rf['true']) > 10 * eps
            o0 = c['options'][0]
            cand_ids = [f'ecsw|{c["rs"]}|k{c["kappa_run"]}|{o0[0]}|{o0[1]:.8g}|native|central',
                        f'none|{c["rs"]}|-|{o0[0]}|{o0[1]:.8g}|native|central']
            chosen, rs_ = None, 'fixture unavailable'
            for cid in cand_ids:
                gg = grid.get(cid)
                zk = 'grid~' + cid.replace('|', '~')
                if gg is None or zk not in Z:
                    continue
                if q1s.get(cid):                                  # A15.5: recomputed Q1 of the candidate
                    chosen = cid
                    break
            if chosen is not None:
                F = Z['grid~' + chosen.replace('|', '~')]['F'][0]
                rom = dict(true=refs.score(F, j0)['e'], case_swap=refs.score(F, j0, ref_case=j1)['e'],
                           time_shift=refs.score(F, j0, shift=1)['e'], refstore=SC.RS.score(F, d, coh, j0, 'ST')['e'])
                ok_rom = abs(rom['case_swap'] - rom['true']) > 10 * eps and abs(rom['time_shift'] - rom['true']) > 10 * eps \
                    and abs(rom['refstore'] - rom['true']) <= 1e-12 * max(rom['true'], 1e-300)
                rs_ = 'pass' if ok_rom else 'fail'
                same_choice = isinstance(sw.get('rom_fixture'), dict) and sw['rom_fixture'].get('cid') == chosen
            else:
                rom, ok_rom = None, True
                same_choice = sw.get('rom_fixture') == 'fixture unavailable'
            put(f'{key}|N-swap', ok_ref and ok_rom and same_choice and sw.get('kinds') == KINDS_WANT, ref=rf, rom=rom,
                rom_status=rs_, chosen=chosen, kinds=sw.get('kinds'))
        else:
            put(f'{key}|N-swap', False, status='missing')
        # ---- N-time and ours' adapter
        tm = R.get('timing') or {}
        for tagk, label, band in (('N-time', 'N-time', (1.8, 2.2)),
                                  ('ours', 'ours-adapter (phase-I execution check; not G-ours)', None)):
            cs_ = (tm.get(tagk) or {}).get('cases', {})
            tok, rho, attempts = timing_ok(cs_, card=(1, 2, 1) if tagk == 'N-time' else (1, 1, 1))
            full = tok and sorted(rho) == list(range(6)) and all(rho.values())
            if full:
                ts = ST.time_stats(rho)
                ok = band is None or (band[0] <= ts['median'] <= band[1] and ts['ci95'][0] > 1.)
                if tagk == 'ours':
                    ok = ok and 'ours' in Z and fin(refs.score_safe(Z['ours']['F'], 0)['e'])
                put(f'{key}|{label}', ok, T=ts['median'], ci95=ts['ci95'], attempts=attempts)
            else:
                put(f'{key}|{label}', False, reason='timing evidence incomplete or inconsistent', inventory_ok=tok,
                    cases=sorted(rho))
    req = {k: v for k, v in acc.items() if v['ok'] is not None}
    acc['ALL'] = dict(ok=all(v['ok'] for v in req.values()), failed=[k for k, v in req.items() if not v['ok']])
    (out / 'ACCEPTANCE.json').write_text(json.dumps(acc, indent=1, default=str))
    for k, v in acc.items():
        extra = {kk: vv for kk, vv in v.items() if kk in ('status', 'reason', 'problems', 'missing_or_failed', 'error')}
        print(f"{k}: {v['ok']}" + ('' if v['ok'] or k == 'ALL' else f'  {json.dumps(extra, default=str)[:400]}'))
    return 0 if acc['ALL']['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
