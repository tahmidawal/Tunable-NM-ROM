"""Offline 2D B7 (long horizon) and B8 (physical diagnostics, sampling adequacy) analysis (DESIGN 5, 6, A1.4, A1.5, A1.7,
A2.2-A2.4, A3.6; CPU, NumPy only, independent decode npcheck.decode257).

Inputs: a pulled b2run output directory (B7_*.npz, B8_*.npz, result.json); this lane's references (pulled lattice files
<refs>/2d/<cohort>/T<T>/c###.{npz,json}, verified against a sha256 manifest and their identity); the anaref json
(per case and window eligibility, cross-check tiers); the E0 store (W1 primary); jcp-qualify / P1 archives (G-L1);
optionally the pulled fine reference files of the check cases (--fine) for the offline recomputation of the in-job
fine-route numbers (A1.8).

Verdict tiers (A2.3): a window whose dimension tier is 'validated' gets verdicts; 'unvalidated' -> labels prefixed
'provisional:'; 'suspended' -> 'descriptive only'.

    python ana7.py --run <b2run output> --refs <dir with 2d/...> --manifest <sha256 file> --anaref <json> [--fine <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
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
SPANS = {}
MAN = {}
ARGS = None


def boot_median(v, n=10000):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None, None
    m = np.median(v[RNG.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_manifest(path):
    """'<sha>  <path>' lines; paths normalised to start at '<d>d/...' (refs manifest './2d/...' or job 'output/<step>/2d/...')."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        h, f = line.split(None, 1)
        f = f.strip()
        for tag in ('2d/', '3d/'):
            i = f.find(tag)
            if i >= 0:
                out[f[i:]] = h
                break
    return out


def load_newref(refs, cid, T, dim=2):
    """This lane's reference with sha256 (manifest) and identity checks; None if absent."""
    coh, c = cid.split('|') if isinstance(cid, str) else cid
    rel = f'{dim}d/{coh}/T{T:g}/c{int(c):03d}'
    stem = Path(refs) / rel
    if not Path(str(stem) + '.json').exists():
        return None
    for suf in ('.json', '.npz'):
        if MAN.get(rel + suf) != sha(str(stem) + suf):
            raise A2.IdentityError(f'reference file {rel + suf} not in the manifest or changed')
    meta = json.loads(Path(str(stem) + '.json').read_text())
    want = dict(dim=dim, cohort=coh, case=int(c), kind='ST', horizon=T)
    if {k: meta.get(k) for k in want} != want:
        raise A2.IdentityError(f'reference identity {[meta.get(k) for k in want]} != {want}')
    if not np.allclose(meta['times'], np.arange(int(round(T / .05)) + 1) * .05, rtol=0, atol=1e-12):
        raise A2.IdentityError('reference time list')
    z = np.load(str(stem) + '.npz')
    return meta, np.asarray(z[meta['final'][2]])


def dmax(Fa, Fb, n0, k0=1):
    d = [float(np.linalg.norm(Fa[k] - Fb[k])) / n0 for k in range(k0, len(Fa))]
    return float('inf') if not np.isfinite(d).all() else max(d)


def tiered(label, tier):
    if tier == 'validated':
        return label
    if tier == 'suspended':
        return 'descriptive only (cross-check suspended)'
    return f'provisional: {label}'


def elig_of(anaref, T, dim=2):
    """{(cid, window): eligibility record} for horizon T from the anaref json."""
    out = {}
    for c in anaref.get('cases', []):
        if c['dim'] == dim and abs(c['horizon'] - T) < 1e-12:
            for w, e in c.get('eligibility', {}).items():
                out[(f"{c['cohort']}|{c['case']}", w)] = e
    return out


def growth_of(rows, nout, minpop):
    growth = {}
    for wa, wb in (('W1', 'W2'), ('W2', 'W3')):
        if WIN[wb][1] >= nout:
            continue
        g, excl, tiers = [], 0, set()
        for x in rows:
            if all(f'e_{w}' in x for w in (wa, wb)) and x.get(f'elig_{wa}') and x.get(f'elig_{wb}') and \
                    x[f'eps_{wa}'] is not None and x[f'eps_{wb}'] is not None and x[f'e_{wa}'] > 2 * x[f'eps_{wa}'] \
                    and x[f'e_{wb}'] > 2 * x[f'eps_{wb}']:
                g.append(x[f'e_{wb}'] / x[f'e_{wa}'])
                tiers |= {x[f'tier_{wa}'], x[f'tier_{wb}']}
            else:
                excl += 1
        lo, hi = boot_median(g)
        if len(g) < minpop:
            lab = 'inconclusive (population)'
        elif hi <= 1:
            lab = 'no growth'
        elif lo > 1 and hi <= 2:
            lab = 'bounded growth'
        elif lo > 2:
            lab = 'accumulating'
        else:
            lab = 'between labels'
        tier = 'suspended' if 'suspended' in tiers else ('validated' if tiers == {'validated'} else 'unvalidated')
        growth[f'{wb}/{wa}'] = dict(n=len(g), excluded=excl, median=float(np.median(g)) if g else None, ci=[lo, hi],
                                    label=tiered(lab, tier), tier=tier, values=g)
    return growth


def stability_of(rows, all_valid):
    stab = dict(all_valid=all_valid,
                growth_cases=sum(any(e > (1 + 1e-3) * x['E_rom'][0] for e in x['E_rom'][1:]) for x in rows),
                monotone_violations_rom=sum(sum(x['E_rom'][k] > (1 + 1e-3) * x['E_rom'][k - 1] for k in range(1, len(x['E_rom'])))
                                            for x in rows),
                monotone_violations_ref=sum(sum(x['E_ref'][k] > (1 + 1e-3) * x['E_ref'][k - 1] for k in range(1, len(x['E_ref'])))
                                            for x in rows if 'E_ref' in x),
                note='at the recorded output times')
    stab['pass'] = bool(all_valid and stab['growth_cases'] == 0)
    return stab


def window_rows(row, F, Rn, meta, nout, EL, cid):
    n0n = float(np.linalg.norm(Rn[0]))
    et = [float(np.linalg.norm(F[k] - Rn[k])) / n0n for k in range(nout)]
    row['e_t_new'] = et
    row['e_rel_t'] = [float(np.linalg.norm(F[k] - Rn[k]) / np.linalg.norm(Rn[k])) for k in range(nout)]
    row['E_ref'] = [.5 * float(np.sum(Rn[k] ** 2)) for k in range(nout)]
    jw = meta.get('judge_windows') or {}
    for w, (a, b) in WIN.items():
        if b < nout:
            row[f'e_{w}'] = max(et[a:b + 1])
            row[f'eps_{w}'] = (jw.get(w) or {}).get('eps_ref')
            e = EL.get((cid, w), {})
            row[f'elig_{w}'] = bool(e.get('eligible'))
            row[f'tier_{w}'] = e.get('tier', 'unvalidated (no cross-check)')
    return n0n


def b7_member(run, mk, member, mesh, refs, EL, T, minpop):
    z = np.load(run / f"B7_{mk.replace('|', '_')}.npz")
    r = int(member['Rp'])
    nout = int(z['nout'])
    ck = [str(x) for x in z['case_ids']]
    arch, srcs = A2.archive_f257(mesh, member)
    dev, q2 = [str(x) for x in z['dev_case_ids']], [str(x) for x in z['q2_case_ids']]
    rows = []
    for i, cid in enumerate(ck):
        coh, c = cid.split('|')
        R0, n0, eps0, el0, rsha = A2.load_e0(cid)
        F = NC.decode257(z['W'][i], r)
        row = dict(case=cid, valid=bool(z['valid'][i]), gcert=bool(z['gcert'][i]), noncert=int(z['noncert'][i]),
                   steps=int(z['steps']), budget_exits=int(z['budget_exits'][i]))
        row['dist_tight'] = dmax(F, NC.decode257(z['W_tight'][i], r), n0) if z['valid_tight'][i] else float('inf')
        if cid in dev:
            k = dev.index(cid)
            Ft = NC.decode257(z['W_tight'][i], r)
            row['dist_tt'] = (dmax(Ft, NC.decode257(z['W_tighter'][k], r), n0)
                              if (z['valid_tighter'][k] and z['valid_tight'][i]) else float('inf'))
        if cid in q2:
            k = q2.index(cid)
            row['dist_qp'] = dmax(F, NC.decode257(z['W_qp'][k], r), n0) if z['valid_qp'][k] else float('inf')
            rr = np.asarray(z['rho'][k], float)
            row['rho_trivial'] = int(np.isnan(rr).sum())              # both zero: excluded and counted (qualify A2.4)
            row['rho_nonfinite'] = int(np.isinf(rr).sum())
            fin = rr[np.isfinite(rr)]
            row['rho_max'] = float(fin.max()) if fin.size else None
            if cid in dev:
                kd = dev.index(cid)
                row['dist_qp_qpp'] = (dmax(NC.decode257(z['W_qp'][k], r), NC.decode257(z['W_qpp'][kd], r), n0)
                                      if (z['valid_qp'][k] and z['valid_qpp'][kd]) else float('inf'))
        if cid in arch:
            row['G_L1'] = dmax(F[:6], arch[cid], n0)
        row['E_rom'] = [.5 * float(np.sum(F[k] ** 2)) for k in range(nout)]          # stability needs no reference
        s0 = A2.score_e0(F[:6], dict(case_id=cid, requested=(coh, c), time_tag=A2.TIME_TAG))
        row.update(e_W1_E0=s0['e'], eligible_W1_E0=s0['eligible'], eps_W1_E0=s0['eps_ref'])
        nr = load_newref(refs, cid, T)
        if nr is not None:
            meta, Rn = nr
            n0n = window_rows(row, F, Rn, meta, nout, EL, cid)
            row['ref_walls_zero'] = bool(np.all(Rn[:, 0, :] == 0) and np.all(Rn[:, -1, :] == 0) and np.all(Rn[:, :, 0] == 0)
                                         and np.all(Rn[:, :, -1] == 0))
            span = SPANS.setdefault(r, A2.Span(r))
            row['e_P_t'] = (np.linalg.norm(A2.inner(Rn) - span.proj(Rn), axis=1) / n0n).tolist()
        rows.append(row)
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
               rho_nonfinite=sum(x['rho_nonfinite'] for x in qr), rho_trivial=sum(x['rho_trivial'] for x in qr),
               worst_qp_qpp=max([x['dist_qp_qpp'] for x in qr if 'dist_qp_qpp' in x], default=None))
    ok = (q2r['worst_dist_qp'] is not None and q2r['worst_dist_qp'] <= BAR_Q2 and q2r['rho_nonfinite'] == 0 and
          q2r['rho_max'] is not None and q2r['rho_max'] <= BAR_RHO)
    conv = q2r['worst_qp_qpp'] is not None and q2r['worst_qp_qpp'] <= BAR_Q2
    q2r['label'] = 'pass' if (ok and conv) else ('unverified (finer rule not converged)' if ok else 'fail')
    gl1 = [x['G_L1'] for x in rows if 'G_L1' in x]
    return dict(rows=rows, Q1=q1, Q2=q2r, Q2_population=f'Q2 on {len(qr)} cases (not individual evidence for others)',
                qualified_at_T=bool(q1['pass'] and q2r['label'] == 'pass'),
                G_L1=dict(worst=max(gl1) if gl1 else None, n=len(gl1), expected=len(rows), sources=srcs,
                          passed=bool(gl1 and len(gl1) == len(rows) and max(gl1) <= 1e-8)),
                stability=stability_of(rows, q1['all_valid']), growth=growth_of(rows, nout, minpop))


def b8_member(run, mk, EL, T, minpop):
    p = run / f"B8_{mk.replace('|', '_')}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    ck = [str(x) for x in z['case_ids']]
    nout = z['e_fine'].shape[1]
    rows, ca_raw, ca_cls, ce = [], [], [], []
    for i, cid in enumerate(ck):
        if not z['has_ref'][i]:
            continue
        meta, Rn = load_newref(ARGS.refs, cid, T)
        jw, jf = meta.get('judge_windows') or {}, meta.get('judge_fine_windows') or {}
        row = dict(case=cid)
        for w, (a, b) in WIN.items():
            if b >= nout:
                continue
            e2, ef = float(np.max(z['e_257'][i][a:b + 1])), float(np.max(z['e_fine'][i][a:b + 1]))
            el, efp = (jw.get(w) or {}).get('eps_ref'), (jf.get(w) or {}).get('eps_ref')
            e = EL.get((cid, w), {})
            fine_ok = (jf.get(w) or {}).get('outcome') == 'accepted'
            row[w] = dict(e257=e2, efine=ef, s=ef / e2 if e2 > 0 else None, eps257=el, epsfine=efp,
                          eligible=bool(e.get('eligible') and fine_ok), tier=e.get('tier', 'unvalidated (no cross-check)'))
            if row[w]['eligible'] and el is not None and efp is not None:
                if e2 <= 2 * el:
                    row[w]['status'] = 'reference-limited'
                else:
                    row[w]['status'] = 'consistent' if abs(ef - e2) <= .10 * e2 + 2 * (el + efp) else 'inconsistent'
            if np.isfinite(z['alias_efine'][i][a:b + 1]).all() and el is not None and efp is not None:   # C-A (A3.6)
                ae2 = float(np.max(z['alias_e257'][i][a:b + 1]))
                aef = float(np.max(z['alias_efine'][i][a:b + 1]))
                ca_raw.append(bool(abs(ae2 - e2) <= 1e-10 * e2 and aef - e2 > .10 * e2 + 2 * (el + efp)))
                if e2 > 2 * el:
                    ca_cls.append(bool(abs(aef - ae2) > .10 * ae2 + 2 * (el + efp)))
        if np.isfinite(z['CE_meas'][i]).all():                                                    # C-E (A2.2)
            meas, pred, zero = z['CE_meas'][i], z['CE_pred'][i], z['CE_zero'][i]
            E0r = float(z['E_ref'][i][0])
            okz = bool(np.all(np.abs(zero) <= 1e-15 * E0r))
            okp = bool(np.all(np.abs(meas - pred) <= 1e-10 * np.abs(pred) + 1e-15 * E0r) and np.any(np.abs(pred) > 1e-15 * E0r))
            ce.append(okz and okp)
        E_r, E_f, D_r, D_f = z['E_rom'][i], z['E_ref'][i], z['D_rom'][i], z['D_ref'][i]
        row['E_ratio_minus1'] = (E_r / E_f - 1).tolist()
        row['D_ratio_minus1'] = (D_r / D_f - 1).tolist()
        Br = E_r[1:] - E_r[:-1] + .025 * (D_r[1:] + D_r[:-1])
        Bf = E_f[1:] - E_f[:-1] + .025 * (D_f[1:] + D_f[:-1])
        row['beta'] = ((Br - Bf) / E_f[0]).tolist()
        row['B_ref'] = (Bf / E_f[0]).tolist()
        u0max = float(z['umax_ref'][i][0])
        row.update(undershoot=float(max(0., -np.min(z['umin_rom'][i])) / u0max),
                   overshoot=float(max(0., np.max(z['umax_rom'][i]) - u0max) / u0max),
                   undershoot_ref=float(max(0., -np.min(z['umin_ref'][i])) / u0max),
                   overshoot_ref=float(max(0., np.max(z['umax_ref'][i]) - u0max) / u0max), wall_max=float(z['wall_max_rom'][i]))
        tot = z['all_err2'][i]
        row['band_density'] = [(float(z['band_err2'][i][k] / tot[k] / BAND_FRAC) if tot[k] > 0 else 'zero error')
                               for k in range(nout)]
        row['e0_ic'] = float(z['e0_ic'][i])
        row['e0_ic_ls'] = float(z['e0_ic_ls'][i]) if 'e0_ic_ls' in z.files and np.isfinite(z['e0_ic_ls'][i]) else None
        ep0 = float(z['eP_fine'][i][0]) if 'eP_fine' in z.files else None
        row['eP_fine_t0'] = ep0
        row['ic_over_floor'] = None if ep0 is None else (row['e0_ic'] / ep0 if ep0 > 0 else 'undefined (zero floor)')
        rows.append(row)
    res = dict(rows=rows)
    for w in ('W1', 'W2'):
        st_ = [r_[w] for r_ in rows if w in r_ and 'status' in r_[w]]
        used = [x for x in st_ if x['status'] != 'reference-limited']
        tiers = {x['tier'] for x in used}
        tier = 'suspended' if 'suspended' in tiers else ('validated' if tiers == {'validated'} else 'unvalidated')
        if len(used) < minpop:
            lab = 'inconclusive (population)'
        else:
            fc = sum(x['status'] == 'consistent' for x in used) / len(used)
            lab = 'adequate' if fc >= .9 else ('inadequate' if fc < .5 else 'mixed')
        res[f'adequacy_{w}'] = dict(label=tiered(lab, tier), raw_label=lab, tier=tier,
                                    consistent=sum(x['status'] == 'consistent' for x in used), used=len(used),
                                    reference_limited=sum(x['status'] == 'reference-limited' for x in st_))
    res['C_A'] = dict(raw=ca_raw, classifier=ca_cls, fired=bool(ca_raw and all(ca_raw) and all(ca_cls)))
    res['C_E'] = dict(n=len(ce), fired=bool(ce and all(ce)))
    res['G_D1'] = bool(rows and all(r_['wall_max'] == 0. for r_ in rows))
    return res


def ranking_adequacy(b8, members, w, minpop):
    """A1.4: per mesh, no resolved reversal of the members' median order between the routes (common population)."""
    common = None
    for mk in members:
        ok = {r_['case'] for r_ in b8[mk]['rows'] if w in r_ and r_[w]['eligible']}
        common = ok if common is None else common & ok
    common = sorted(common or [])
    if len(common) < minpop or len(members) < 2:
        return dict(label='inconclusive (population)', n=len(common))
    med = {}
    for mk in members:
        rr = {r_['case']: r_[w] for r_ in b8[mk]['rows']}
        med[mk] = {k: float(np.median([rr[c][k] for c in common])) for k in ('e257', 'efine', 'eps257', 'epsfine')}
    res_, rev = 0, 0
    for a, b in itertools.combinations(members, 2):
        r1 = abs(med[a]['e257'] - med[b]['e257']) > 2 * max(med[a]['eps257'], med[b]['eps257'])
        r2 = abs(med[a]['efine'] - med[b]['efine']) > 2 * max(med[a]['epsfine'], med[b]['epsfine'])
        if r1 and r2:
            res_ += 1
            rev += int(np.sign(med[a]['e257'] - med[b]['e257']) != np.sign(med[a]['efine'] - med[b]['efine']))
    lab = 'inconclusive' if res_ == 0 else ('adequate' if rev == 0 else 'inadequate')
    return dict(label=lab, n=len(common), resolved_pairs=res_, reversed_pairs=rev, medians=med)


def kendall_tau_b(x, y):
    c = d = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
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


def front_order(b7, mks, minpop):
    """A2.4: common population of cases eligible in W1 and W2 for every member; ties within 2x the larger median eps."""
    common = None
    for mk in mks:
        ok = {x['case'] for x in b7[mk]['rows'] if x.get('elig_W1') and x.get('elig_W2')}
        common = ok if common is None else common & ok
    common = sorted(common or [])
    if len(common) < minpop or len(mks) < 2:
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


def fine_recompute(run, members, fine_dir, T):
    """A1.8: recompute the in-job fine-route errors of the check cases from the pulled fine reference files and an
    independent NumPy decode at the 1025^2 nodes; bar 1e-9 relative; also the reference's fine walls."""
    P, Tm = NC.bank()
    x = np.arange(1025) / 1024.
    X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    wall = (X[:, 0] == 0) | (X[:, 0] == 1) | (X[:, 1] == 0) | (X[:, 1] == 1)
    feats = None
    out = []
    for mk, m in members.items():
        p, p7 = run / f"B8_{mk.replace('|', '_')}.npz", run / f"B7_{mk.replace('|', '_')}.npz"
        if not (p.exists() and p7.exists()):
            continue
        z, z7 = np.load(p), np.load(p7)
        ck7 = [str(c) for c in z7['case_ids']]
        r = int(m['Rp'])
        for i, cid in enumerate([str(c) for c in z['case_ids']]):
            coh, c = cid.split('|')
            ff = Path(fine_dir) / '2d' / coh / f'T{T:g}' / f'c{int(c):03d}_fine.npz'
            if not (ff.exists() and z['has_ref'][i]):
                continue
            meta = json.loads((Path(ARGS.refs) / '2d' / coh / f'T{T:g}' / f'c{int(c):03d}.json').read_text())
            if meta.get('fine_sha256') != sha(ff):
                raise A2.IdentityError(f'fine file {ff} differs from its recorded sha256')
            Rf = np.asarray(np.load(ff)['fine_' + meta['final'][2]])
            if feats is None:
                feats = np.concatenate([NC.NB.features_grad(P, X[s:s + 8192])[0] for s in range(0, len(X), 8192)])
                feats[wall] = 0.
            W = z7['W'][ck7.index(cid)]
            F = (W @ (feats @ Tm[:, :r]).T).reshape(len(W), 1025, 1025)
            n0f = float(np.linalg.norm(Rf[0]))
            ef = np.array([np.linalg.norm(F[k] - Rf[k]) / n0f for k in range(len(W))])
            rel = float(np.max(np.abs(ef - z['e_fine'][i]) / np.maximum(np.abs(z['e_fine'][i]), 1e-300)))
            walls = bool(np.all(Rf[:, 0, :] == 0) and np.all(Rf[:, -1, :] == 0) and np.all(Rf[:, :, 0] == 0) and np.all(Rf[:, :, -1] == 0))
            out.append(dict(member=mk, case=cid, e_fine_rel=rel, ref_fine_walls_zero=walls, passed=bool(rel <= 1e-9 and walls)))
    return dict(rows=out, passed=bool(out and all(r_['passed'] for r_ in out)))


def _band_frac(L=1024, width=1 / 32):
    x = np.arange(L + 1) / L
    m1 = np.minimum(x, 1 - x) <= width + 1e-15
    inter = np.zeros(L + 1, bool)
    inter[1:-1] = True
    band = (m1[:, None] | m1[None, :]) & inter[:, None] & inter[None, :]
    return float(band.sum() / (inter[:, None] & inter[None, :]).sum())


BAND_FRAC = _band_frac()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--manifest', required=True, help='sha256 manifest of the reference files (REFS-MANIFEST or OUTPUTS)')
    ap.add_argument('--anaref', required=True)
    ap.add_argument('--fine')
    ap.add_argument('--minpop', type=int, default=None)
    ap.add_argument('--out')
    ARGS = a = ap.parse_args()
    MAN.update(load_manifest(a.manifest))
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    T = float(res['config']['b7']['horizon'])
    mesh = int(res['config']['mesh'])
    minpop = a.minpop if a.minpop is not None else (6 if mesh == 4096 else 8)
    EL = elig_of(json.loads(Path(a.anaref).read_text()), T)
    out = dict(run=str(run), mesh=mesh, horizon=T, b7={}, b8={},
               label='W1 against E0 R_ST (primary); windows against this lane\'s R_ST realisation; eps conditional')
    members = {mk: v['member'] for mk, v in res['b7'].items()}
    for mk, m in members.items():
        out['b7'][mk] = b7_member(run, mk, m, mesh, a.refs, EL, T, minpop)
        b8 = b8_member(run, mk, EL, T, minpop)
        if b8 is not None:
            out['b8'][mk] = b8
    lspg = [mk for mk, m in members.items() if m['form'] == 'LSPG']
    out['front_order'] = dict(all=front_order(out['b7'], list(members), minpop), lspg=front_order(out['b7'], lspg, minpop))
    if out['b8']:
        out['ranking_adequacy'] = {w: dict(all=ranking_adequacy(out['b8'], list(out['b8']), w, minpop),
                                           lspg=ranking_adequacy(out['b8'], [m for m in lspg if m in out['b8']], w, minpop))
                                   for w in ('W1', 'W2')}
        lab = [out['b8'][m]['adequacy_W1']['raw_label'] for m in lspg if m in out['b8']]
        rk = out['ranking_adequacy']['W1']['lspg']['label']
        out['lane_statement_W1'] = dict(
            lspg_members=len(lab), all_adequate=bool(lab and all(x == 'adequate' for x in lab)), ranking=rk,
            statement=('the 257^2 lattice is adequate for the qualified 2D front (relative to the 1025^2 route)'
                       if lab and all(x == 'adequate' for x in lab) and rk == 'adequate' else 'not established'))
    if a.fine:
        out['fine_recompute'] = fine_recompute(run, members, a.fine, T)
    dest = Path(a.out or HERE / 'checks' / f"b7b8-{run.parents[1].name}-{mesh}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps({mk: (v['qualified_at_T'], v['stability']['pass'], {k: g['label'] for k, g in v['growth'].items()})
                            for mk, v in out['b7'].items()}))


def load_newref3(refs, coh, j, T):
    return load_newref(refs, (coh, j), T, dim=3)


if __name__ == '__main__':
    main()
