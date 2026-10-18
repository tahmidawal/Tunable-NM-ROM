"""Offline 2D B7 (long horizon) and B8 (physical diagnostics, sampling adequacy) analysis (DESIGN 5, 6, A1.4, A1.5, A1.7,
A2.2-A2.4, A3.6; CPU, NumPy only, independent decode npcheck.decode257).

Inputs: a pulled b2run output directory (B7_*.npz, B8_*.npz, result.json); this lane's references (the pulled lattice
files <refs>/2d/<cohort>/T<T>/c###.{npz,json}); the anaref summary (G-R2 per case, cross-check tiers); the E0 store
(W1 primary); the jcp-qualify / P1 archives (G-L1).

    python ana7.py --run runs/<attempt>/archive/output/b2run_<cfg> --refs <dir with 2d/...> --anaref checks/refs-<x>.json
                   [--fine <dir with the pulled dev6 *_fine.npz>]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import npcheck as NC  # noqa: E402
import ana2d as A2  # noqa: E402

RS = A2.RS
BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3
WIN = {'W1': (1, 5), 'W2': (6, 10), 'W3': (11, 20)}
RNG = np.random.default_rng(0)


def boot_median(v, n=10000):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None, None
    idx = RNG.integers(0, len(v), (n, len(v)))
    m = np.median(v[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_newref(refs, cid, T):
    coh, c = cid.split('|')
    stem = Path(refs) / '2d' / coh / f'T{T:g}' / f'c{int(c):03d}'
    if not Path(str(stem) + '.json').exists():
        return None
    meta = json.loads(Path(str(stem) + '.json').read_text())
    z = np.load(str(stem) + '.npz')
    return meta, np.asarray(z[meta['final'][2]])


def dmax(Fa, Fb, n0, k0=1):
    d = [float(np.linalg.norm(Fa[k] - Fb[k])) / n0 for k in range(k0, len(Fa))]
    return float('inf') if not np.isfinite(d).all() else max(d)


def b7_member(run, mk, member, mesh, refs, anaref, T):
    z = np.load(run / f"B7_{mk.replace('|', '_')}.npz")
    r = int(member['Rp'])
    nout = int(z['nout'])
    ck = [str(x) for x in z['case_ids']]
    arch = A2.archive_f257(mesh, member)
    dev, q2 = [str(x) for x in z['dev_case_ids']], [str(x) for x in z['q2_case_ids']]
    rows = []
    gr2 = {f"{c['cohort']}|{c['case']}": c.get('G_R2', {}).get('passed', False) for c in anaref.get('cases', [])
           if c['horizon'] == T}
    for i, cid in enumerate(ck):
        coh, c = cid.split('|')
        R0, m0 = RS.load(2, coh, int(c))
        n0 = float(m0['judge']['n0'])
        F = NC.decode257(z['W'][i], r)
        row = dict(case=cid, valid=bool(z['valid'][i]), gcert=bool(z['gcert'][i]), noncert=int(z['noncert'][i]),
                   steps=int(z['steps']), budget_exits=int(z['budget_exits'][i]))
        row['dist_tight'] = dmax(F, NC.decode257(z['W_tight'][i], r), n0) if z['valid_tight'][i] else float('inf')
        if cid in dev:
            k = dev.index(cid)
            Ft = NC.decode257(z['W_tight'][i], r)
            row['dist_tt'] = dmax(Ft, NC.decode257(z['W_tighter'][k], r), n0) if z['valid_tighter'][k] else float('inf')
            row['dist_qp_qpp'] = None
        if cid in q2:
            k = q2.index(cid)
            row['dist_qp'] = dmax(F, NC.decode257(z['W_qp'][k], r), n0) if z['valid_qp'][k] else float('inf')
            rr = z['rho'][k]
            row['rho_max'] = float(np.nanmax(rr)) if np.isfinite(rr).any() or np.isinf(rr).any() else None
            row['rho_inf'] = int(np.isinf(rr).sum())
            if cid in dev:
                kd = dev.index(cid)
                row['dist_qp_qpp'] = (dmax(NC.decode257(z['W_qp'][k], r), NC.decode257(z['W_qpp'][kd], r), n0)
                                      if (z['valid_qp'][k] and z['valid_qpp'][kd]) else float('inf'))
        # G-L1: first six outputs reproduce the archived T = 0.25 run
        if cid in arch:
            row['G_L1'] = dmax(F[:6], arch[cid], n0)
        # errors: W1 vs E0 (primary); every window vs this lane's reference (same realisation for growth)
        s0 = RS.score(F[:6], 2, coh, int(c))
        row['e_W1_E0'], row['eligible_W1_E0'], row['eps_W1_E0'] = s0['e'], s0['eligible'], s0['eps_ref']
        nr = load_newref(refs, cid, T)
        if nr is not None:
            meta, Rn = nr
            n0n = float(np.linalg.norm(Rn[0]))
            et = [float(np.linalg.norm(F[k] - Rn[k])) / n0n for k in range(nout)]
            row['e_t_new'] = et
            row['e_rel_t'] = [float(np.linalg.norm(F[k] - Rn[k]) / np.linalg.norm(Rn[k])) for k in range(nout)]
            row['E_rom'] = [.5 * float(np.sum(F[k] ** 2)) for k in range(nout)]
            row['E_ref'] = [.5 * float(np.sum(Rn[k] ** 2)) for k in range(nout)]
            jw = meta.get('judge_windows') or {}
            for w, (a, b) in WIN.items():
                if b < nout:
                    row[f'e_{w}'] = max(et[a:b + 1])
                    row[f'eps_{w}'] = (jw.get(w) or {}).get('eps_ref')
                    row[f'acc_{w}'] = (jw.get(w) or {}).get('outcome') == 'accepted'
            row['G_R2'] = gr2.get(cid, False)
            # floor over time (bank extrapolation)
            span = SPANS.setdefault(r, A2.Span(r))
            RI = A2.inner(Rn)
            row['e_P_t'] = (np.linalg.norm(RI - span.proj(Rn), axis=1) / n0n).tolist()
        rows.append(row)
    # Q1 / Q2 at T (A1.7, 5.2)
    q1 = dict(all_valid=all(x['valid'] for x in rows), gcert=all(x['gcert'] for x in rows),
              noncert_fraction=sum(x['noncert'] for x in rows) / max(1, sum(x['steps'] for x in rows)),
              worst_dist_tight=max(x['dist_tight'] for x in rows),
              worst_tt=max([x['dist_tt'] for x in rows if 'dist_tt' in x], default=None),
              budget_exits=sum(x['budget_exits'] for x in rows))
    q1['pass'] = bool(q1['all_valid'] and q1['gcert'] and q1['noncert_fraction'] <= .01 and q1['worst_dist_tight'] <= BAR_Q1
                      and q1['worst_tt'] is not None and q1['worst_tt'] <= BAR_TT)
    qr = [x for x in rows if 'dist_qp' in x]
    q2r = dict(population=len(qr), worst_dist_qp=max([x['dist_qp'] for x in qr], default=None),
               rho_max=max([x['rho_max'] for x in qr if x['rho_max'] is not None], default=None),
               rho_inf=sum(x['rho_inf'] for x in qr),
               worst_qp_qpp=max([x['dist_qp_qpp'] for x in qr if x.get('dist_qp_qpp') is not None], default=None))
    conv = q2r['worst_qp_qpp'] is not None and q2r['worst_qp_qpp'] <= BAR_Q2
    ok = (q2r['worst_dist_qp'] is not None and q2r['worst_dist_qp'] <= BAR_Q2 and q2r['rho_inf'] == 0 and
          q2r['rho_max'] is not None and q2r['rho_max'] <= BAR_RHO)
    q2r['label'] = 'pass' if (ok and conv) else ('unverified (finer rule not converged)' if ok else 'fail')
    gl1 = [x['G_L1'] for x in rows if 'G_L1' in x]
    # stability at the recorded output times (5.3)
    st = [x for x in rows if 'E_rom' in x]
    stab = dict(all_valid=q1['all_valid'],
                growth_cases=sum(any(e > (1 + 1e-3) * x['E_rom'][0] for e in x['E_rom'][1:]) for x in st),
                monotone_violations_rom=sum(sum(x['E_rom'][k] > (1 + 1e-3) * x['E_rom'][k - 1] for k in range(1, len(x['E_rom'])))
                                            for x in st),
                monotone_violations_ref=sum(sum(x['E_ref'][k] > (1 + 1e-3) * x['E_ref'][k - 1] for k in range(1, len(x['E_ref'])))
                                            for x in st))
    stab['pass'] = bool(stab['all_valid'] and stab['growth_cases'] == 0 and len(st) == len(rows))
    # growth (A2.4: both windows eligible and resolved; same reference realisation)
    g = []
    excl = 0
    for x in rows:
        if all(k in x for k in ('e_W1', 'e_W2')) and x['acc_W1'] and x['acc_W2'] and x['G_R2'] and \
                x['eps_W1'] is not None and x['eps_W2'] is not None and x['e_W1'] > 2 * x['eps_W1'] and x['e_W2'] > 2 * x['eps_W2']:
            g.append(x['e_W2'] / x['e_W1'])
        else:
            excl += 1
    lo, hi = boot_median(g)
    if len(g) < 8:
        glab = 'inconclusive (population)'
    elif hi <= 1:
        glab = 'no growth'
    elif lo > 1 and hi <= 2:
        glab = 'bounded growth'
    elif lo > 2:
        glab = 'accumulating'
    else:
        glab = 'between labels'
    return dict(rows=rows, Q1=q1, Q2=q2r, qualified_at_T=bool(q1['pass'] and q2r['label'] == 'pass'),
                G_L1=dict(worst=max(gl1) if gl1 else None, n=len(gl1), passed=bool(gl1 and max(gl1) <= 1e-8)),
                stability=stab, growth=dict(n=len(g), excluded=excl, median=float(np.median(g)) if g else None,
                                            ci=[lo, hi], label=glab, values=g))


SPANS = {}


def _band_frac(L=1024, width=1 / 32):
    x = np.arange(L + 1) / L
    m1 = np.minimum(x, 1 - x) <= width + 1e-15
    inter = np.zeros(L + 1, bool)
    inter[1:-1] = True
    band = (m1[:, None] | m1[None, :]) & inter[:, None] & inter[None, :]
    return float(band.sum() / (inter[:, None] & inter[None, :]).sum())


BAND_FRAC = _band_frac()


def b8_member(run, mk, refs, T, min_pop=8):
    p = run / f"B8_{mk.replace('|', '_')}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    ck = [str(x) for x in z['case_ids']]
    nout = z['e_fine'].shape[1]
    rows, cons = [], []
    ca_raw, ca_cls, ce = [], [], []
    for i, cid in enumerate(ck):
        if not z['has_ref'][i]:
            continue
        nr = load_newref(refs, cid, T)
        if nr is None:
            continue
        meta, _ = nr
        jw, jf = meta.get('judge_windows') or {}, meta.get('judge_fine_windows') or {}
        row = dict(case=cid)
        for w, (a, b) in WIN.items():
            if b >= nout:
                continue
            e2, ef = float(np.max(z['e_257'][i][a:b + 1])), float(np.max(z['e_fine'][i][a:b + 1]))
            el, efp = (jw.get(w) or {}).get('eps_ref'), (jf.get(w) or {}).get('eps_ref')
            acc = (jw.get(w) or {}).get('outcome') == 'accepted' and (jf.get(w) or {}).get('outcome') == 'accepted'
            row[w] = dict(e257=e2, efine=ef, s=ef / e2 if e2 > 0 else None, eps257=el, epsfine=efp, eligible=acc)
            if acc and el is not None and efp is not None:
                if e2 <= 2 * el:
                    row[w]['status'] = 'reference-limited'
                else:
                    row[w]['status'] = 'consistent' if abs(ef - e2) <= .10 * e2 + 2 * (el + efp) else 'inconsistent'
            # C-A on the check cases
            if np.isfinite(z['alias_efine'][i][a:b + 1]).any() and el is not None and efp is not None:
                ka = a + int(np.nanargmax(z['e_257'][i][a:b + 1]))
                ae2 = float(np.max(z['alias_e257'][i][a:b + 1]))
                aef = float(np.max(z['alias_efine'][i][a:b + 1]))
                raw = bool(abs(ae2 - e2) <= 1e-10 * e2 and aef - e2 > .10 * e2 + 2 * (el + efp))
                ca_raw.append(raw)
                if e2 > 2 * el:
                    ca_cls.append(bool(abs(aef - ae2) > .10 * ae2 + 2 * (el + efp)))
        # C-E (raw balance): zero and scaled-reference closed form
        if np.isfinite(z['CE_meas'][i]).any():
            meas, pred, zero = z['CE_meas'][i], z['CE_pred'][i], z['CE_zero'][i]
            E0r = float(z['E_ref'][i][0])
            okz = bool(np.all(np.abs(zero) <= 1e-15 * E0r))
            okp = bool(np.all(np.abs(meas - pred) <= 1e-10 * np.abs(pred) + 1e-15 * E0r) and np.any(np.abs(pred) > 1e-15 * E0r))
            ce.append(okz and okp)
        # diagnostics
        n0f = float(z['n0_fine'][i])
        E_r, E_f = z['E_rom'][i], z['E_ref'][i]
        row['E_ratio_minus1'] = (E_r / E_f - 1).tolist()
        row['D_ratio_minus1'] = (z['D_rom'][i] / z['D_ref'][i] - 1).tolist()
        Br = z['E_rom'][i][1:] - z['E_rom'][i][:-1] + .025 * (z['D_rom'][i][1:] + z['D_rom'][i][:-1])
        Bf = z['E_ref'][i][1:] - z['E_ref'][i][:-1] + .025 * (z['D_ref'][i][1:] + z['D_ref'][i][:-1])
        row['beta'] = ((Br - Bf) / E_f[0]).tolist()
        row['B_ref'] = (Bf / E_f[0]).tolist()
        u0max = float(z['umax_ref'][i][0])
        row['undershoot'] = float(max(0., -np.min(z['umin_rom'][i])) / u0max)
        row['overshoot'] = float(max(0., np.max(z['umax_rom'][i]) - u0max) / u0max)
        row['undershoot_ref'] = float(max(0., -np.min(z['umin_ref'][i])) / u0max)
        row['overshoot_ref'] = float(max(0., np.max(z['umax_ref'][i]) - u0max) / u0max)
        row['wall_max'] = float(z['wall_max_rom'][i])
        tot = z['all_err2'][i]
        frac_band = BAND_FRAC                                           # node fraction of the 1/32 band (budcommon.band_mask)
        row['band_density'] = [(float(z['band_err2'][i][k] / tot[k] / frac_band) if tot[k] > 0 else 'zero error')
                               for k in range(nout)]
        row['e0_ic'] = float(z['e0_ic'][i])
        rows.append(row)
    res = dict(rows=rows)
    for w in ('W1', 'W2'):
        st_ = [r_[w]['status'] for r_ in rows if w in r_ and 'status' in r_[w]]
        used = [s for s in st_ if s != 'reference-limited']
        if len(used) < min_pop:
            lab = 'inconclusive (population)'
        else:
            fc = sum(s == 'consistent' for s in used) / len(used)
            lab = 'adequate' if fc >= .9 else ('inadequate' if fc < .5 else 'mixed')
        res[f'adequacy_{w}'] = dict(label=lab, consistent=sum(s == 'consistent' for s in used), used=len(used),
                                    reference_limited=sum(s == 'reference-limited' for s in st_))
    res['C_A'] = dict(raw=ca_raw, classifier=ca_cls, fired=bool(ca_raw and all(ca_raw) and all(ca_cls)))
    res['C_E'] = dict(n=len(ce), fired=bool(ce and all(ce)))
    res['G_D1'] = bool(all(r_['wall_max'] == 0. for r_ in rows))
    return res


def kendall_tau_b(x, y):
    n = len(x)
    c = d = tx = ty = 0
    for i, j in itertools.combinations(range(n), 2):
        a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if a == 0 and b == 0:
            continue
        if a == 0:
            tx += 1
        elif b == 0:
            ty += 1
        elif a == b:
            c += 1
        else:
            d += 1
    den = np.sqrt((c + d + tx) * (c + d + ty))
    return float((c - d) / den) if den > 0 else None


def front_order(b7, T):
    """A2.4: common population of cases eligible in W1 and W2 for every member; ties within 2x the larger median eps."""
    mks = list(b7)
    common = None
    for mk in mks:
        ok = {x['case'] for x in b7[mk]['rows'] if x.get('acc_W1') and x.get('acc_W2') and x.get('G_R2')}
        common = ok if common is None else common & ok
    common = sorted(common or [])
    if len(common) < 8:
        return dict(label='inconclusive (population)', n=len(common))
    med = {}
    for mk in mks:
        rr = {x['case']: x for x in b7[mk]['rows']}
        med[mk] = {w: float(np.median([rr[c][f'e_{w}'] for c in common])) for w in ('W1', 'W2')}
        med[mk].update({f'eps_{w}': float(np.median([rr[c][f'eps_{w}'] for c in common])) for w in ('W1', 'W2')})
    resolved = reversed_ = 0
    for a, b in itertools.combinations(mks, 2):
        r1 = abs(med[a]['W1'] - med[b]['W1']) > 2 * max(med[a]['eps_W1'], med[b]['eps_W1'])
        r2 = abs(med[a]['W2'] - med[b]['W2']) > 2 * max(med[a]['eps_W2'], med[b]['eps_W2'])
        if r1 and r2:
            resolved += 1
            reversed_ += int(np.sign(med[a]['W1'] - med[b]['W1']) != np.sign(med[a]['W2'] - med[b]['W2']))
    lab = 'inconclusive' if resolved == 0 else ('preserved' if reversed_ == 0 else 'not preserved')
    return dict(label=lab, n=len(common), resolved_pairs=resolved, reversed_pairs=reversed_, medians=med,
                kendall_tau_b=kendall_tau_b([med[m]['W1'] for m in mks], [med[m]['W2'] for m in mks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--anaref', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    T = float(res['config']['b7']['horizon'])
    mesh = int(res['config']['mesh'])
    anaref = json.loads(Path(a.anaref).read_text())
    out = dict(run=str(run), mesh=mesh, horizon=T, b7={}, b8={},
               label='W1 against E0 R_ST (primary); windows against this lane\'s R_ST realisation; eps conditional')
    for mk, v in res['b7'].items():
        out['b7'][mk] = b7_member(run, mk, v['member'], mesh, a.refs, anaref, T)
        b8 = b8_member(run, mk, a.refs, T)
        if b8 is not None:
            out['b8'][mk] = b8
    out['front_order'] = front_order(out['b7'], T)
    dest = Path(a.out or HERE / 'checks' / f"b7b8-{run.parents[1].name}-{mesh}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps({mk: (v['qualified_at_T'], v['stability']['pass'], v['growth']['label']) for mk, v in out['b7'].items()}))


def load_newref3(refs, coh, j, T):
    stem = Path(refs) / '3d' / coh / f'T{T:g}' / f'c{int(j):03d}'
    if not Path(str(stem) + '.json').exists():
        return None
    meta = json.loads(Path(str(stem) + '.json').read_text())
    z = np.load(str(stem) + '.npz')
    return meta, np.asarray(z[meta['final'][2]])


if __name__ == '__main__':
    main()
