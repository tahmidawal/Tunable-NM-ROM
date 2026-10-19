"""jcp-regimes production analysis (DESIGN §6, §7, A1.5, A1.8, A2.3, A2.4, A3.2, A4). CPU only.

    JAX_PLATFORMS=cpu python analyze.py --rom runs/rom1024 [--store store]   -> checks/analysis.json

Per configuration (setting, arm) and cohort: Q1/Q2 on every sampled case of the cohort (rscore.qualify, subset = cohort),
E0-style errors on reference-eligible valid cases, label (precedence FAILS > NOT QUALIFIED > accuracy label, plus the
"not usable" flag), ratios with bootstrap intervals against the in-range baseline I = N0 + dev6 + val32 and against N0
alone, the paired A-/C1 contrast, distance trends, sigma_a separation checks, representation floors, coverage counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import npcheck as NC
import rcohorts as RG
import rscore as RS

HERE = Path(__file__).resolve().parent
PRIMARY_REGIMES = ('Ap', 'Am', 'C1', 'C2', 'B1', 'B2')
DESCRIPTIVE = ('N0s',)
DIAGNOSTIC_ARMS = {('R384', 'GAL_CN_1')}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def references(cfg):
    """{(cohort, case): reference dict} for every case the ROM config runs (union over arms; A3.3 reduced C2 included)."""
    ids = []
    for st in cfg['settings']:
        for a in st['arms']:
            ids += RS.expected_populations(cfg, a)['cases']
    out = {}
    for cid in dict.fromkeys(ids):
        coh, c = cid.split('|')
        out[(coh, int(c))] = RS.reference(coh, int(c))
    return out


SEP_RELABEL = dict(Ap='viscosity extrapolation confounded with steepness', Am='viscosity extrapolation confounded with steepness',
                   B1='family change confounded with steepness', B2='family change confounded with steepness',
                   C1='nominal steepness only', C2='nominal steepness only')


def coverage_counts(refs, coh):
    ms = [r_ for (c_, _), r_ in refs.items() if c_ == coh]
    k = dict(n=len(ms), eligible=sum(r_['eligible'] for r_ in ms))
    if coh in RG.REGIMES_2D:
        metas = [r_['meta'] for r_ in ms]
        k.update(outcomes={o: sum(m['outcome'] == o for m in metas) for o in sorted({m['outcome'] for m in metas})},
                 unresolved_under_allowance=sum(bool(m.get('tight_skipped') or m.get('refine_skipped')) for m in metas),
                 refined=sum(bool(m['refined']) for m in metas), tight_used=sum(bool(m['tight_used']) for m in metas))
    return k


def errors_of(R, qual, refs):
    """{cid: dict(e, valid, eligible, n0_match)} for every case of the R file."""
    out = {}
    Rp = int(R['Rp'])
    for j, cid in enumerate(str(x) for x in R['case_ids']):
        coh, c = cid.split('|')
        ref = refs[(coh, int(c))]
        pc = qual.get(cid, {})
        valid = bool(pc.get('valid'))
        e = RS.error(NC.decode257(R['W'][j], Rp), ref, R['n0'][j]) if valid else None
        out[cid] = dict(e=e, valid=valid, eligible=ref['eligible'], eps_ref=ref['eps_ref'],
                        n0_rel=abs(float(R['n0'][j]) - ref['n0']) / ref['n0'])
    return out


def usable(errs, coh):
    """Eligible, valid, finite, positive errors of one cohort (A2.4 exclusions), and the excluded count."""
    v = [(cid, x['e']) for cid, x in errs.items() if cid.split('|')[0] == coh]
    ok = [(cid, e) for cid, e in v if e is not None and np.isfinite(e) and e > 0 and errs[cid]['eligible'] and errs[cid]['valid']]
    return ok, len(v) - len(ok)


def label_cell(errs, qcoh, coh, base_e):
    ok, excl = usable(errs, coh)
    e = np.array([x for _, x in ok])
    cell = dict(n=sum(1 for cid in errs if cid.split('|')[0] == coh), n_used=len(e), excluded=excl,
                median=float(np.median(e)) if len(e) else None, worst=float(np.max(e)) if len(e) else None)
    invalid = sum(1 for cid, x in errs.items() if cid.split('|')[0] == coh and not x['valid'])
    cell['invalid_runs'] = invalid
    pt, lo, hi, dropped = RS.boot_ratio(e, base_e) if len(e) and len(base_e) else (float('nan'),) * 3 + (0,)
    acc = RS.acc_label(pt, lo, hi, dropped, len(e), len(base_e))
    cell.update(ratio_median=pt, ci=[lo, hi], boot_dropped=dropped, accuracy_label=acc,
                ratio_worst=(float(np.max(e) / np.max(base_e)) if len(e) and len(base_e) else None))
    cell['worst_degrades'] = bool(cell['ratio_worst'] is not None and cell['ratio_worst'] > RS.FACTOR)
    cell['not_usable'] = bool(cell['median'] is not None and cell['median'] > RS.UNUSABLE)
    nq = qcoh['numerically_qualified']
    cell['Q1'], cell['Q2'] = qcoh['Q1'], qcoh['Q2']
    cell['qualified'] = nq
    cell['label'] = 'FAILS' if invalid else ('NOT QUALIFIED' if not nq else acc)
    if invalid or not nq:
        cell['accuracy_label_unqualified'] = acc
    return cell


def paired(errs):
    a = dict(usable(errs, 'Am')[0])
    c = dict(usable(errs, 'C1')[0])
    pairs = [(int(k.split('|')[1]), c[f'C1|{k.split("|")[1]}'] / v) for k, v in a.items() if f'C1|{k.split("|")[1]}' in c]
    if len(pairs) < RS.MIN_ELIG:
        return dict(n_pairs=len(pairs), label='inconclusive (pair coverage)')
    q = np.array([x for _, x in pairs])
    pt, lo, hi, _ = RS.boot_ratio(q, np.ones_like(q), paired=True)
    from scipy.stats import wilcoxon
    lq = np.log(q)
    if np.all(lq == 0):
        p = 1.
    else:
        nz = np.abs(lq[lq != 0])
        exact = len(lq) <= 25 and len(np.unique(nz)) == len(nz) and np.all(lq != 0)
        p = float(wilcoxon(lq, zero_method='wilcox', alternative='two-sided', method='exact' if exact else 'approx').pvalue)
    return dict(n_pairs=len(pairs), median_q=pt, ci=[lo, hi], wilcoxon_p=p, label=RS.acc_label(pt, lo, hi, 0, len(q), len(q)),
                pairs=pairs)


def trends(errs, refs):
    def collect(cohs, key):
        xs, ys = [], []
        for cid, x in errs.items():
            coh, c = cid.split('|')
            if coh not in cohs:
                continue
            d = RG.desc_2d(coh, RG.descriptors_2d(coh)[int(c)])
            ok = x['e'] is not None and x['valid'] and x['eligible'] and np.isfinite(x['e']) and x['e'] > 0
            xs.append(key(d))
            ys.append(np.log(x['e']) if ok else np.nan)
        return RS.spearman_perm(xs, ys)
    return {'Ap|d_nu': collect(('Ap',), lambda d: d['d_nu']), 'Am|d_nu': collect(('Am',), lambda d: d['d_nu']),
            'Am+C1+C2|log10_s': collect(('Am', 'C1', 'C2'), lambda d: np.log10(d['s']))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rom', required=True)
    ap.add_argument('--store', default=str(HERE / 'store'))
    ap.add_argument('--out', default=str(HERE / 'checks' / 'analysis.json'))
    a = ap.parse_args()
    RS.STORE = Path(a.store)
    qd = next(Path(a.rom).glob('output/qacc_*'))
    rep = json.loads((qd / 'result.json').read_text())
    cohorts = rep['config']['cohorts']
    refs = references(rep['config'])
    # inventory and full-field verification are required (A6 production gate)
    want = sorted(f"R_{s_['name']}_{a_['form']}_{a_['scheme']}_{a_['factor']:g}.npz" for s_ in rep['config']['settings']
                  for a_ in s_['arms'])
    have = sorted(p.name for p in qd.glob('R_*.npz'))
    if have != want or not rep.get('complete'):
        raise SystemExit(f'ROM job incomplete: missing {sorted(set(want) - set(have))}, complete={rep.get("complete")}')
    import hashlib
    certified = json.loads((HERE / 'configs' / 'rom1024.json').read_text())
    if rep['config'] != certified:                   # the job ran the certified production config, not a variant
        raise SystemExit('ROM job config differs from the certified configs/rom1024.json')
    fv = json.loads(next(Path(a.rom).glob('output/fverify_*/fverify.json')).read_text())
    fvc = hashlib.sha256((HERE / 'configs' / 'fverify-rom1024.json').read_bytes()).hexdigest()
    if not (fv['all_pass'] and fv['config_sha256'] == fvc and sorted(fv['arms']) == sorted(w[2:-4] for w in want) and all(
            fv['arms'][k]['R_file_sha256'] == hashlib.sha256((qd / f'R_{k}.npz').read_bytes()).hexdigest()
            and fv['arms'][k]['f257_sha256'] == str(np.load(qd / f'R_{k}.npz')['f257_sha256']) for k in fv['arms'])):
        raise SystemExit('full-field verification missing, failed or not bound to these R files / config')
    deferred = {}                                   # A3.3: cases of a cohort not in the configured population
    for coh in cohorts:
        if coh in RG.REGIMES_2D:
            got = {c for (cc, c) in refs if cc == coh}
            miss = [c for c in range(len(RG.descriptors_2d(coh))) if c not in got]
            if miss:
                deferred[coh] = miss
    out = dict(deferred_cases=deferred, rom_attempt=str(a.rom), job=dict(job_id=rep.get('job_id'), gpu=rep.get('gpu'), backend=rep.get('backend'),
                                                 commit=rep.get('commit')),
               coverage={coh: coverage_counts(refs, coh) for coh in cohorts}, configs={})
    sig = {}
    for (coh, c), r_ in refs.items():
        if coh in RG.REGIMES_2D and r_['eligible'] and r_['sigma'] and r_['sigma']['sigma'] is not None:
            sig.setdefault(coh, []).append(r_['sigma']['sigma'])
    out['sigma'] = {coh: [dict(case=c, sigma=r_['sigma']) for (cc, c), r_ in refs.items() if cc == coh]
                    for coh in cohorts if coh in RG.REGIMES_2D}
    n0s_ok = len(sig.get('N0s', [])) >= 4
    sep_in = dict(sig) if n0s_ok else {k: v for k, v in sig.items() if k != 'N0s'}
    out['separation'] = RS.separation(sep_in, PRIMARY_REGIMES)
    out['separation_note'] = None if n0s_ok else 'stiff-part comparison unavailable (N0s < 4 eligible): N0 alone'
    out['regime_labels'] = {g: (SEP_RELABEL[g] if isinstance(out['separation'].get(g), dict)
                                and out['separation'][g].get('passed') is False else None) for g in PRIMARY_REGIMES}
    # calibration label (A2.2, A3.1): smoke probes, plus production probes if they were added
    sm = json.loads((HERE / 'checks' / 'smoke-accept.json').read_text())
    prod_probes = []
    for g in RG.REGIMES_2D:
        for rj in Path(a.store).parent.glob(f'runs/ref{g}/output/refjob_ref-{g}/result.json'):
            prod_probes += [p_['tight_coarse'] for p_ in json.loads(rj.read_text()).get('probes', []) if p_.get('tight_coarse')]
    agree = sm.get('calib_agree') and all(p_['verdict'] == 'agree' for p_ in prod_probes)
    out['calibration'] = dict(smoke_agree=sm.get('calib_agree'), production_probes=len(prod_probes),
                              label=None if agree else 'floor proxy not supported on these regimes')
    for p in sorted(qd.glob('R_*.npz')):
        R = np.load(p)
        name, arm = p.stem[2:].split('_', 1)
        setting = next(s for s in rep['config']['settings'] if s['name'] == name)
        rows = [r_ for r_ in rep['rows'] if r_['setting'] == name and r_['arm'] == arm]
        rt = (RS.rule_tables(setting['rule'], int(R['Rp']), R['kx'], R['ky'], 1024),
              RS.rule_tables(setting['qp'], int(R['Rp']), R['kx'], R['ky'], 1024))
        arm_cfg = next(a_ for a_ in setting['arms'] if f"{a_['form']}_{a_['scheme']}_{a_['factor']:g}" == arm)
        expect = RS.arm_expect(rep['config'], arm_cfg, setting)
        quals, per_case = {}, {}
        for coh in cohorts:
            q = RS.qualify(R, rows, expect, rt, subset=[coh])
            quals[coh] = q
            per_case.update(q['per_case'])
        errs = errors_of(R, per_case, refs)
        base = np.array([x for coh in ('N0', 'dev6', 'val32') for _, x in usable(errs, coh)[0]])
        base_n0 = np.array([x for _, x in usable(errs, 'N0')[0]])
        dv = np.array([x for coh in ('dev6', 'val32') for _, x in usable(errs, coh)[0]])
        cfg = dict(setting=name, arm=arm, Rp=int(R['Rp']), rule=setting['rule'], diagnostic=(name, arm) in DIAGNOSTIC_ARMS,
                   baseline=dict(n=len(base), median=float(np.median(base)) if len(base) else None,
                                 worst=float(np.max(base)) if len(base) else None),
                   in_range_consistency=dict(zip(('ratio', 'lo', 'hi', 'dropped'), RS.boot_ratio(base_n0, dv))),
                   cells={}, cells_vs_N0={}, per_case=errs, qualification={coh: dict(Q1=quals[coh]['Q1'], Q2=quals[coh]['Q2'],
                                                                                     qualified=quals[coh]['numerically_qualified'],
                                                                                     pooled_noncertified=quals[coh]['pooled_noncertified'],
                                                                                     pooled_steps=quals[coh]['pooled_steps'],
                                                                                     disagreements=quals[coh]['disagreements'][:5])
                                                                              for coh in cohorts})
        for coh in PRIMARY_REGIMES + DESCRIPTIVE:
            if coh in cohorts:
                cfg['cells'][coh] = label_cell(errs, quals[coh], coh, base)
                cfg['cells'][coh]['unresolved_under_allowance'] = out['coverage'][coh].get('unresolved_under_allowance')
                cfg['cells'][coh]['regime_label'] = out['regime_labels'].get(coh)
                cfg['cells_vs_N0'][coh] = label_cell(errs, quals[coh], coh, base_n0)
        cfg['paired_Am_C1'] = paired(errs)
        cfg['trends'] = trends(errs, refs)
        out['configs'][f'{name}|{arm}'] = cfg
        print('analysed', name, arm, flush=True)
    # representation floors (diagnostic), per regime case and bank width
    floors = {}
    for (coh, c), r_ in refs.items():
        if coh in RG.REGIMES_2D and r_['eligible']:
            for Rp in (128, 256, 384, 512):
                floors.setdefault(f'{coh}|R{Rp}', []).append(RS.projection_floor(r_['field'], Rp))
    for coh in ('dev6', 'val32'):
        for (cc, c), r_ in refs.items():
            if cc == coh:
                for Rp in (128, 256, 384, 512):
                    floors.setdefault(f'{coh}|R{Rp}', []).append(RS.projection_floor(r_['field'], Rp))
    out['projection_floor'] = {k: dict(median=float(np.median(v)), max=float(np.max(v)), n=len(v)) for k, v in floors.items()}
    # pseudo-spectral cross-check (diagnostic, DESIGN 4.3; E0 verdict.crosscheck with its A10-amended rule, label carried)
    import verdict as V
    out['ps'] = {}
    for pj in sorted(HERE.glob('runs/ps/output/refjob_ps-*/2d/*/PS/c*_a*.json')):
        ps = json.loads(pj.read_text())
        coh, j = ps['cohort'], int(ps['case'])
        arr = np.load(pj.with_suffix('.npz'))
        ref = refs.get((coh, j)) or RS.reference(coh, j)
        runs_ = {r_['id']: r_ for r_ in ps['runs']}
        full = next(i for i in ps['selected_spatial'] if runs_[i]['N'] == ps['Ns'][-1])
        rec = V.crosscheck(ref['meta']['judge'], ref['field'], dict(
            spatial=[(runs_[i]['N'], np.asarray(arr[i])) for i in ps['selected_spatial']], half=np.asarray(arr[ps['selected_half']]),
            full=np.asarray(arr[full]), finite=ps['finite'], gates_ok=ps['gates_ok'], stability_ok=ps['stability_ok']),
            ref['meta']['judge']['n0'])
        if rec['verdict'] == 'UNRESOLVED':
            rec['note'] = 'no escalation ladder budgeted in this lane (E0 A1.7 would owe one)'
        out['ps'][f'{coh}|{j}'] = rec
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(clean(out), indent=1) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
