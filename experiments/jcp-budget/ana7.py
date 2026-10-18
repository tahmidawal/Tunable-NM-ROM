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
    lat = {2: 256, 3: 64}[dim]
    if meta.get('lattice') != lat or meta.get('metric_id') != f'{dim}|lattice={lat}|times=0:0.05:{T:g}|norm=fixed_t0|max_k>=1':
        raise A2.IdentityError('reference lattice / metric')
    if not same_case(meta, dim, coh, int(c)):
        raise A2.IdentityError('reference case descriptor differs from the cohort')
    rec = ANAREF_SHA.get((dim, coh, int(c), T))
    if rec is not None and (rec['sha256']['.npz'] != MAN.get(rel + '.npz') or rec['sha256']['.json'] != MAN.get(rel + '.json')
                            or rec['final'] != meta['final']):
        raise A2.IdentityError('reference differs from the one the anaref eligibility was established on')
    z = np.load(str(stem) + '.npz')
    return meta, np.asarray(z[meta['final'][2]])


_CSHA = {}
ANAREF_SHA = {}


def same_case(meta, dim, coh, c):
    """The reference's recorded case descriptor equals this machine's cohort descriptor to 1e-13 relative. (The cohort
    sha256 is computed from float64 draws that can differ in the last bit across numpy versions (np.exp of the
    log-uniform viscosity), so the cluster-written hash is compared only when numpy versions agree.)"""
    sys.path.insert(0, str(HERE / 'ref'))
    import cohorts as CH
    co = _COH.setdefault((dim, coh), CH.Cohort(dim, coh))
    want = co.desc(c)
    got = meta.get('desc') or {}
    def flat(x):
        return np.concatenate([np.ravel(np.asarray(v, float)) for k, v in sorted(x.items())]) if isinstance(x, dict) else np.asarray(x, float)
    try:
        return bool(sorted(got) == sorted(want) and np.allclose(flat(got), flat(want), rtol=1e-13, atol=0))
    except Exception:  # noqa: BLE001
        return False


_COH = {}


def COHORT_SHA(dim, coh):
    if (dim, coh) not in _CSHA:
        sys.path.insert(0, str(HERE / 'ref'))
        import cohorts as CH
        _CSHA[(dim, coh)] = CH.Cohort(dim, coh).sha256
    return _CSHA[(dim, coh)]


def dmax(Fa, Fb, n0, k0=1):
    d = [float(np.linalg.norm(Fa[k] - Fb[k])) / n0 for k in range(k0, len(Fa))]
    return float('inf') if not np.isfinite(d).all() else max(d)


def tiered(label, tier):
    """A2.3: validated -> verdict; unvalidated -> no verdict, provisional value only; suspended -> descriptive only."""
    if tier == 'validated':
        return label
    if tier == 'suspended':
        return 'descriptive only (cross-check suspended)'
    return f'no verdict (cross-check unvalidated); provisional: {label}'


def tier_of(tiers):
    tiers = set(tiers)
    return 'suspended' if 'suspended' in tiers else ('validated' if tiers == {'validated'} else 'unvalidated')


def elig_of(anaref, T, dim=2):
    """{(cid, window): eligibility record} for horizon T from the anaref json."""
    out = {}
    for c in anaref.get('cases', []):
        if c['dim'] == dim and abs(c['horizon'] - T) < 1e-12:
            ANAREF_SHA[(dim, c['cohort'], int(c['case']), T)] = dict(sha256=c['sha256'], final=c['final'])
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
        tier = tier_of(tiers)
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
    b7c = ARGS.cfg['b7']
    allc = A2.case_set(b7c.get('cases', ARGS.cfg['cases']))
    inv_ok = (ck == allc and [str(x) for x in z['dev_case_ids']] == [c for c in A2.case_set(b7c.get('tighter_cases', 'dev6')) if c in allc]
              and [str(x) for x in z['q2_case_ids']] == [c for c in A2.case_set(b7c.get('q2_cases', 'sub16')) if c in allc])
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
    q1['inventory_matches_configuration'] = inv_ok
    q1['pass'] = bool(inv_ok and q1['all_valid'] and q1['gcert'] and q1['noncert_fraction'] <= .01 and q1['worst_dist_tight'] <= BAR_Q1
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
                G_L1=dict(worst=max(gl1) if gl1 else None, n=len(gl1), expected=len(allc), sources=srcs,
                          passed=bool(inv_ok and gl1 and len(gl1) == len(allc) and max(gl1) <= 1e-8)),
                stability=stability_of(rows, q1['all_valid']), growth=growth_of(rows, nout, minpop))


def classify(e2, ef, el, efp):
    """A1.4 sampling classification of one case-window (the production classifier, also used by C-A)."""
    if e2 <= 2 * el:
        return 'reference-limited'
    return 'consistent' if abs(ef - e2) <= .10 * e2 + 2 * (el + efp) else 'inconsistent'


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
                row[w]['status'] = classify(e2, ef, el, efp)
            if np.isfinite(z['alias_efine'][i][a:b + 1]).all() and el is not None and efp is not None:   # C-A (A3.6)
                ae2 = float(np.max(z['alias_e257'][i][a:b + 1]))
                aef = float(np.max(z['alias_efine'][i][a:b + 1]))
                ca_raw.append(bool(abs(ae2 - e2) <= 1e-10 * e2 and aef - e2 > .10 * e2 + 2 * (el + efp)))
                lab_ = classify(ae2, aef, el, efp)                      # the production classifier on the injected case
                ca_cls.append(lab_)
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
        tier = tier_of({x['tier'] for x in used})
        if len(used) < minpop:
            lab = 'inconclusive (population)'
        else:
            fc = sum(x['status'] == 'consistent' for x in used) / len(used)
            lab = 'adequate' if fc >= .9 else ('inadequate' if fc < .5 else 'mixed')
        res[f'adequacy_{w}'] = dict(label=tiered(lab, tier), raw_label=lab, tier=tier,
                                    consistent=sum(x['status'] == 'consistent' for x in used), used=len(used),
                                    reference_limited=sum(x['status'] == 'reference-limited' for x in st_))
    resolved = [l for l in ca_cls if l != 'reference-limited']
    res['C_A'] = dict(raw=ca_raw, classifier=ca_cls, resolved=len(resolved), unresolved=len(ca_cls) - len(resolved),
                      fired=bool(ca_raw and all(ca_raw) and resolved and all(l == 'inconsistent' for l in resolved)))
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
    tier = tier_of({r_[w]['tier'] for mk in members for r_ in b8[mk]['rows'] if r_['case'] in common})
    return dict(label=tiered(lab, tier), raw_label=lab, tier=tier, n=len(common), resolved_pairs=res_, reversed_pairs=rev,
                medians=med)


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
    tier = tier_of({x[f'tier_{w}'] for mk in mks for x in b7[mk]['rows'] if x['case'] in common for w in ('W1', 'W2')})
    return dict(label=tiered(lab, tier), raw_label=lab, tier=tier, n=len(common), resolved_pairs=resolved,
                reversed_pairs=reversed_, medians=med,
                kendall_tau_b=kendall_tau_b([med[m]['W1'] for m in mks], [med[m]['W2'] for m in mks]))


def fine_recompute(run, members, fine_dir, T, check_ids):
    """A1.8: recompute EVERY in-job fine-route number of the check cases (with a reference) from the pulled fine
    reference files, the pulled lattice reference and an independent NumPy decode at the 1025^2 nodes: route errors and
    normalisers, energy / dissipation (ROM and reference), extrema, ROM wall maximum, near-wall and total error, IC
    errors (deployed and LS, from the job's LS coefficients), the fine floor per time, the injected alias errors and the
    C-E arrays. Bar 1e-9 relative (absolute floor 1e-14 of the compared scale). Coverage: every check case with a
    reference must have its fine file locally, else fail."""
    P, Tm = NC.bank()
    x = np.arange(1025) / 1024.
    X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    wall = (X[:, 0] == 0) | (X[:, 0] == 1) | (X[:, 1] == 0) | (X[:, 1] == 1)
    band = np.zeros((1025, 1025), bool)
    m1 = np.minimum(x, 1 - x) <= 1 / 32 + 1e-15
    inter = np.zeros(1025, bool)
    inter[1:-1] = True
    band = (m1[:, None] | m1[None, :]) & inter[:, None] & inter[None, :]
    allm = inter[:, None] & inter[None, :]
    h = 1 / 1024
    feats, GQ = None, {}
    phi = np.outer(np.sin(256 * np.pi * x), np.sin(256 * np.pi * x))
    out, missing = [], []

    def ED(F, nu):
        I = F[:, 1:-1, 1:-1]
        gx = (F[:, 2:, 1:-1] - F[:, :-2, 1:-1]) / (2 * h)
        gy = (F[:, 1:-1, 2:] - F[:, 1:-1, :-2]) / (2 * h)
        return .5 * h * h * np.sum(I ** 2, axis=(1, 2)), nu * h * h * np.sum(gx ** 2 + gy ** 2, axis=(1, 2))

    def close(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            return float('inf')
        sc = max(float(np.max(np.abs(b))), 1e-300)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-14 * sc)))

    # expected coverage, built independently of the produced B8 records: every member x every check case that has a
    # reference file in the local reference tree
    expected = {(mk, cid) for mk in members for cid in check_ids
                if (Path(ARGS.refs) / '2d' / cid.split('|')[0] / f'T{T:g}' / f"c{int(cid.split('|')[1]):03d}.json").exists()}
    seen = set()
    for mk, m in members.items():
        p, p7 = run / f"B8_{mk.replace('|', '_')}.npz", run / f"B7_{mk.replace('|', '_')}.npz"
        if not (p.exists() and p7.exists()):
            continue
        z, z7 = np.load(p), np.load(p7)
        ck7 = [str(c) for c in z7['case_ids']]
        r = int(m['Rp'])
        lsf = run / f'LS_R{r}.npz'
        LSC = {}
        if lsf.exists():
            zl = np.load(lsf)
            LSC = {str(c_): zl['C'][k] for k, c_ in enumerate(zl['case_ids'])}
        for i, cid in enumerate([str(c) for c in z['case_ids']]):
            if (mk, cid) not in expected:
                continue
            if not z['has_ref'][i]:
                missing.append(f'{mk}|{cid} (no in-job record)')
                continue
            coh, c = cid.split('|')
            ff = Path(fine_dir) / '2d' / coh / f'T{T:g}' / f'c{int(c):03d}_fine.npz'
            if not ff.exists():
                missing.append(f'{mk}|{cid} (no local fine file)')
                continue
            if cid not in LSC or not np.isfinite(z['e0_ic_ls'][i]):
                missing.append(f'{mk}|{cid} (no LS coefficients)')
                continue
            seen.add((mk, cid))
            meta, Rl = load_newref(ARGS.refs, cid, T)
            if meta.get('fine_sha256') != sha(ff):
                raise A2.IdentityError(f'fine file {ff} differs from its recorded sha256')
            Rf = np.asarray(np.load(ff)['fine_' + meta['final'][2]])
            if feats is None:
                feats = np.concatenate([NC.NB.features_grad(P, X[s:s + 8192])[0] for s in range(0, len(X), 8192)])
                feats[wall] = 0.
            if r not in GQ:
                Gr = feats @ Tm[:, :r]
                Gi = Gr.reshape(1025, 1025, r)[1:-1, 1:-1].reshape(-1, r)
                GQ[r] = (Gr, np.linalg.qr(Gi)[0])
            Gr, Qf = GQ[r]
            W = z7['W'][ck7.index(cid)]
            nout = len(W)
            F = (W @ Gr.T).reshape(nout, 1025, 1025)
            nu = float(meta['desc']['nu'])
            E = F - Rf
            n0f, n0l = float(np.linalg.norm(Rf[0])), float(np.linalg.norm(Rl[0]))
            rec = {}
            rec['n0_fine'] = close([n0f], [z['n0_fine'][i]])
            rec['n0_257'] = close([n0l], [z['n0_257'][i]])
            rec['e_fine'] = close(np.linalg.norm(E.reshape(nout, -1), axis=1) / n0f, z['e_fine'][i])
            e257 = np.linalg.norm((F[:, ::4, ::4] - Rl).reshape(nout, -1), axis=1) / n0l
            rec['e_257'] = close(e257, z['e_257'][i])
            Er, Dr = ED(F, nu)
            Ef, Df = ED(Rf, nu)
            for k_, v_ in (('E_rom', Er), ('D_rom', Dr), ('E_ref', Ef), ('D_ref', Df)):
                rec[k_] = close(v_, z[k_][i])
            for k_, v_ in (('umax_rom', F.max(axis=(1, 2))), ('umin_rom', F.min(axis=(1, 2))), ('umax_ref', Rf.max(axis=(1, 2))),
                           ('umin_ref', Rf.min(axis=(1, 2)))):
                rec[k_] = close(v_, z[k_][i])
            rec['wall_max_rom'] = 0. if float(np.max(np.abs(np.concatenate([F[:, 0, :], F[:, -1, :], F[:, :, 0], F[:, :, -1]], 1)))) \
                == float(z['wall_max_rom'][i]) else float('inf')
            rec['band_err2'] = close(np.sum(np.where(band, E ** 2, 0.), axis=(1, 2)), z['band_err2'][i])
            rec['all_err2'] = close(np.sum(np.where(allm, E ** 2, 0.), axis=(1, 2)), z['all_err2'][i])
            rec['e0_ic'] = close([np.linalg.norm(E[0]) / n0f], [z['e0_ic'][i]])
            Fl = (LSC[cid] @ Gr.T).reshape(1025, 1025)
            rec['e0_ic_ls'] = close([np.linalg.norm(Fl - Rf[0]) / n0f], [z['e0_ic_ls'][i]])
            RfI = Rf[:, 1:-1, 1:-1].reshape(nout, -1)
            rec['eP_fine'] = close(np.linalg.norm(RfI - (RfI @ Qf) @ Qf.T, axis=1) / n0f, z['eP_fine'][i])
            jw, jf = meta.get('judge_windows') or {}, meta.get('judge_fine_windows') or {}
            ae2, aef = np.full(nout, np.nan), np.full(nout, np.nan)
            ef = np.linalg.norm(E.reshape(nout, -1), axis=1)
            nphi = float(np.linalg.norm(phi))
            for k in range(1, nout):
                w_ = 'W1' if k <= 5 else ('W2' if k <= 10 else 'W3')
                el = (jw.get(w_) or {}).get('eps_ref') or 0.
                efp = (jf.get(w_) or {}).get('eps_ref') or 0.
                ak = (ef[k] + n0f * (1.2 * e257[k] + 2 * (efp + el) + 1e-6)) / nphi
                Fa = F[k] + ak * phi
                aef[k] = np.linalg.norm(Fa - Rf[k]) / n0f
                ae2[k] = np.linalg.norm(Fa[::4, ::4] - Rl[k]) / n0l
            rec['alias_efine'] = close(aef[1:], z['alias_efine'][i][1:])
            rec['alias_e257'] = close(ae2[1:], z['alias_e257'][i][1:])
            Bref = Ef[1:] - Ef[:-1] + .025 * (Df[1:] + Df[:-1])
            fac = 1 + .01 * np.arange(nout)
            Es, Ds = ED(Rf * fac[:, None, None], nu)
            rec['CE_meas'] = close(Es[1:] - Es[:-1] + .025 * (Ds[1:] + Ds[:-1]) - Bref, z['CE_meas'][i])
            worst = max(rec.values())
            out.append(dict(member=mk, case=cid, worst_rel=worst, per_array=rec,
                            ref_fine_walls_zero=bool(np.all(Rf[:, 0, :] == 0) and np.all(Rf[:, -1, :] == 0) and
                                                     np.all(Rf[:, :, 0] == 0) and np.all(Rf[:, :, -1] == 0)),
                            passed=bool(worst <= 1e-9)))
    missing += [f'{mk}|{cid} (no B8 file)' for mk, cid in sorted(expected - seen) if not any(x.startswith(f'{mk}|{cid}') for x in missing)]
    ok = bool(expected and not missing and seen == expected and all(r_['passed'] and r_['ref_fine_walls_zero'] for r_ in out))
    return dict(rows=out, expected=sorted(f'{a}|{b}' for a, b in expected), missing=missing, passed=ok)


def _band_frac(L=1024, width=1 / 32):
    x = np.arange(L + 1) / L
    m1 = np.minimum(x, 1 - x) <= width + 1e-15
    inter = np.zeros(L + 1, bool)
    inter[1:-1] = True
    band = (m1[:, None] | m1[None, :]) & inter[:, None] & inter[None, :]
    return float(band.sum() / (inter[:, None] & inter[None, :]).sum())


BAND_FRAC = _band_frac()


def main():
    """One or more b2run output directories of the SAME mesh and horizon (A8: the 1024^2 partitions b1024a + b1024b):
    members are merged (a member present in two runs is an error), coverage is checked per run against its own
    configuration, and the front order / ranking / lane statement are computed over the merged member set."""
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, action='append')
    ap.add_argument('--refs', required=True)
    ap.add_argument('--manifest', required=True, help='sha256 manifest of the reference files (REFS-MANIFEST or OUTPUTS)')
    ap.add_argument('--anaref', required=True)
    ap.add_argument('--fine')
    ap.add_argument('--minpop', type=int, default=None)
    ap.add_argument('--out')
    ARGS = a = ap.parse_args()
    MAN.update(load_manifest(a.manifest))
    runs = [Path(r) for r in a.run]
    ress = [json.loads((r / 'result.json').read_text()) for r in runs]
    Ts = {float(r_['config']['b7']['horizon']) for r_ in ress}
    meshes = {int(r_['config']['mesh']) for r_ in ress}
    assert len(Ts) == 1 and len(meshes) == 1, ('runs differ in horizon or mesh', Ts, meshes)
    T, mesh = Ts.pop(), meshes.pop()
    minpop = a.minpop if a.minpop is not None else (6 if mesh == 4096 else 8)
    EL = elig_of(json.loads(Path(a.anaref).read_text()), T)
    out = dict(runs=[str(r) for r in runs], mesh=mesh, horizon=T, b7={}, b8={}, coverage=dict(per_run={}),
               label='W1 against E0 R_ST (primary); windows against this lane\'s R_ST realisation; eps conditional')
    members, owner = {}, {}
    for run, res in zip(runs, ress):
        ARGS.cfg = res['config']
        mine = {f"R{m['Rp']}|{m['form']}|{m['scheme']}|{float(m['factor']):g}": m for m in res['config']['members']}
        dup = sorted(set(mine) & set(members))
        cov = dict(configured=sorted(mine), duplicates=dup, b7_missing=sorted(set(mine) - set(res['b7'])),
                   b8_missing=sorted(mk for mk in mine if not (run / f"B8_{mk.replace('|', '_')}.npz").exists()),
                   b7_extra=sorted(set(res['b7']) - set(mine)))
        cov['passed'] = not (dup or cov['b7_missing'] or cov['b8_missing'] or cov['b7_extra'])
        out['coverage']['per_run'][str(run)] = cov
        for mk, m in mine.items():
            if mk in members or mk not in res['b7']:
                continue
            members[mk], owner[mk] = m, run
            out['b7'][mk] = b7_member(run, mk, m, mesh, a.refs, EL, T, minpop)
            b8 = b8_member(run, mk, EL, T, minpop)
            if b8 is not None:
                out['b8'][mk] = b8
        if a.fine:
            cc = res['config']['b8'].get('check_cases', 'dev6')
            check_ids = set(A2.case_set(cc)) if isinstance(cc, str) else {f'{x[0]}|{x[1]}' for x in cc}
            fr = fine_recompute(run, mine, a.fine, T, check_ids)
            out.setdefault('fine_recompute_runs', {})[str(run)] = fr
    out['coverage']['passed'] = all(c['passed'] for c in out['coverage']['per_run'].values())
    out['coverage']['merged_members'] = sorted(members)
    lspg = [mk for mk, m in members.items() if m['form'] == 'LSPG']
    out['front_order'] = dict(all=front_order(out['b7'], list(members), minpop), lspg=front_order(out['b7'], lspg, minpop))
    if out['b8']:
        out['ranking_adequacy'] = {w: dict(all=ranking_adequacy(out['b8'], list(out['b8']), w, minpop),
                                           lspg=ranking_adequacy(out['b8'], [m for m in lspg if m in out['b8']], w, minpop))
                                   for w in ('W1', 'W2')}
        lab = [out['b8'][m]['adequacy_W1']['label'] for m in lspg if m in out['b8']]          # tiered labels only
        rk = out['ranking_adequacy']['W1']['lspg']['label']
        out['lane_statement_W1'] = dict(
            lspg_members=len(lab), all_adequate=bool(lab and all(x == 'adequate' for x in lab)), ranking=rk,
            statement=('the 257^2 lattice is adequate for the qualified 2D front (relative to the 1025^2 route)'
                       if lab and all(x == 'adequate' for x in lab) and rk == 'adequate' else 'not established'))
    if a.fine:
        frs = list(out['fine_recompute_runs'].values())
        out['fine_recompute'] = dict(rows=[r_ for f in frs for r_ in f['rows']], missing=[m_ for f in frs for m_ in f['missing']],
                                     passed=bool(frs and all(f['passed'] for f in frs)))
    dest = Path(a.out or HERE / 'checks' / f"b7b8-{runs[0].parents[1].name}-{mesh}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps({mk: (v['qualified_at_T'], v['stability']['pass'], {k: g['label'] for k, g in v['growth'].items()})
                            for mk, v in out['b7'].items()}))


def load_newref3(refs, coh, j, T):
    return load_newref(refs, (coh, j), T, dim=3)


if __name__ == '__main__':
    main()
