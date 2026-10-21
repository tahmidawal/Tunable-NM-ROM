"""The 2x gate (DESIGN §9, A1.2, A1.16, A2.2, A2.16, A3.16, A4.10, A5.10, A5.12) and its T-gate control.

    python gate.py selftest                                   T-gate on synthetic inputs (must-fail control)
    python gate.py --timing <vtime result.json> --ladder <vladder result.json> --eligibility checks/eligibility.json
                   --rombox checks/rom-box.json --out checks/gate.json

G(tau, C) = cost_FOM / cost_ROM, both anchor-normalised (C2 convention: median over slots of the per-slot median of
t_B / mean(t_A1, t_A2), stage 2 only). FOM = the frozen stage-1 union winner for (tau, C); ROM = the frozen denominator of
rom-box.json for tau, re-measured in-job (fallback: inherited C2 ratio, labelled). Interval: anchor-paired slot bootstrap.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DECISIVE = (0.03, 0.10)
GATE_C = 100.0
REQUIRED = ('PSI', 'ETD', 'FDMG')
NBOOT = 2000


def target_outcome(G, status, unresolved_families):
    """One decisive target: 'pass' / 'fail' / 'inconclusive' (A2.2, A4.10)."""
    if status != 'ok' or G is None or not np.isfinite(G):
        return 'inconclusive'
    if G >= 2.0:
        return 'inconclusive' if unresolved_families else 'pass'
    return 'fail'


def decide(targets, ref_ok):
    """targets: {tau: dict(G, status, unresolved_families)} at C = 100 for the decisive taus. Returns the gate label."""
    if not ref_ok:
        return 'UNRESOLVED', {}
    outs = {t: target_outcome(v.get('G'), v.get('status'), v.get('unresolved_families') or []) for t, v in targets.items()}
    if any(o == 'pass' for o in outs.values()):
        return 'PASS-PROVISIONAL', outs
    if outs and all(o == 'fail' for o in outs.values()) and len(outs) == len(DECISIVE):
        return 'FAIL', outs
    return 'UNRESOLVED', outs


def selftest():
    ok = lambda G, fam=None: dict(G=G, status='ok', unresolved_families=fam or [])
    inc = dict(G=None, status='inconclusive', unresolved_families=[])
    cases = [
        ('3x pass', {0.03: ok(3.0), 0.10: ok(1.0)}, True, 'PASS-PROVISIONAL'),
        ('exactly 2 pass', {0.03: ok(2.0), 0.10: ok(1.0)}, True, 'PASS-PROVISIONAL'),
        ('1.9 fail', {0.03: ok(1.9), 0.10: ok(1.9)}, True, 'FAIL'),
        ('1x fail', {0.03: ok(1.0), 0.10: ok(1.0)}, True, 'FAIL'),
        ('missing ROM target', {0.03: inc, 0.10: ok(1.0)}, True, 'UNRESOLVED'),
        ('empty FOM set', {0.03: dict(G=None, status='no qualified', unresolved_families=[]), 0.10: ok(1.0)}, True, 'UNRESOLVED'),
        ('coverage < 24 / reference failure', {0.03: ok(3.0), 0.10: ok(3.0)}, False, 'UNRESOLVED'),
        ('one pass + other inconclusive', {0.03: ok(2.5), 0.10: inc}, True, 'PASS-PROVISIONAL'),
        ('one fail + other inconclusive', {0.03: ok(1.5), 0.10: inc}, True, 'UNRESOLVED'),
        ('pass blocked by unresolved family', {0.03: ok(3.0, ['ETD']), 0.10: ok(1.0)}, True, 'UNRESOLVED'),
        ('fail not blocked by unresolved family', {0.03: ok(1.2, ['ETD']), 0.10: ok(1.0, ['ETD'])}, True, 'FAIL'),
    ]
    rows = []
    for name, t, ref_ok, want in cases:
        got, _ = decide(t, ref_ok)
        rows.append(dict(case=name, want=want, got=got, fires=got == want))
    # success at C = 10 only: the gate reads C = 100 only, so a C = 10 pass with C = 100 fails is FAIL
    got, _ = decide({0.03: ok(1.1), 0.10: ok(1.2)}, True)
    rows.append(dict(case='success at C = 10 only (C = 100 fails)', want='FAIL', got=got, fires=got == 'FAIL'))
    return dict(rows=rows, all_fire=all(r['fires'] for r in rows))


CASES = [f'dev6|{s}' for s in range(6)]
REPS = (0, 1, 2)


def cost_by_slot(stage2, label):
    """{slot case id: median ratio over reps} for valid stage-2 triplets of `label`, and whether the coverage is exact:
    every registered slot dev6|0..5 with repetitions {0, 1, 2} once each, valid, identity-checked."""
    by, reps = {}, {}
    for t in stage2:
        if t['B'] == label and t.get('valid') and t.get('hashes_ok', True):
            by.setdefault(t['case'], []).append(t['ratio'])
            reps.setdefault(t['case'], []).append(t['rep'])
    exact = set(by) == set(CASES) and all(sorted(reps[c]) == list(REPS) for c in CASES)
    return {c: float(np.median(v)) for c, v in by.items()}, exact


def scored_ok(subj):
    rows = subj.get('scoring', [])
    return bool(sorted(r_['case'] for r_ in rows) == CASES and all(r_.get('valid') is True and r_.get('repro') is True
                                                                  for r_ in rows))


def diag_cost(T, lb, veto, kernel=False, per_slot=False):
    """Diagnostic subjects (KERNEL / SETUP / C = 1): own acceptance predicate -- exact scoring coverage, every run valid,
    kernel/query field agreement (KERNEL), complete valid timing, not vetoed; else an 'unavailable: reason' string."""
    subj = T['subjects'].get(lb)
    if subj is None:
        return 'unavailable: not a timing subject'
    rows = subj.get('scoring', [])
    if sorted(r_['case'] for r_ in rows) != CASES or not all(r_.get('valid') is True for r_ in rows):
        return 'unavailable: scoring coverage or validity'
    if kernel and not all(r_.get('kernel_query_ok') is True for r_ in rows):
        return 'unavailable: kernel output differs from the query output'
    if lb in veto:
        return 'unavailable: identity veto'
    cs_, ex_ = cost_by_slot(T['stage2'], lb)
    if not ex_:
        return 'unavailable: incomplete stage-2 coverage'
    return cs_ if per_slot else float(np.median([cs_[c] for c in CASES]))


def scored_ok_A(T):
    rows = T['subjects'].get('A', {}).get('scoring', [])
    return bool(sorted(r_['case'] for r_ in rows) == CASES and all(r_.get('valid') is True for r_ in rows))


def boot(fs, rs, slots, rng):
    out = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(slots), len(slots))
        f = np.median([fs[slots[i]] for i in idx])
        r = np.median([rs[slots[i]] for i in idx])
        out.append(f / r)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        r = selftest()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r['all_fire'] else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', required=True)
    ap.add_argument('--ladder', required=True)
    ap.add_argument('--eligibility', required=True)
    ap.add_argument('--rombox', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    T = json.loads(Path(a.timing).read_text())
    Lr = json.loads(Path(a.ladder).read_text())
    El = json.loads(Path(a.eligibility).read_text())
    RB = json.loads(Path(a.rombox).read_text())
    HERE = Path(__file__).resolve().parent
    b_ = T['config'].get('bindings') or {}
    chain = dict(timing_bindings_present=bool(b_) and set(b_.get('files', {})) == {
        'checks/eligibility.json', 'checks/candidates.json', 'checks/rom-box.json'})
    for name_, h_ in (b_.get('files') or {}).items():
        chain[name_] = bool(hashlib.sha256((HERE / name_).read_bytes()).hexdigest() == h_)
    chain['rombox_frozen'] = bool(hashlib.sha256(Path(a.rombox).read_bytes()).hexdigest() ==
                                  (HERE / 'checks/rom-box.FROZEN').read_text().split()[0])
    chain['eligibility_frozen'] = bool(hashlib.sha256(Path(a.eligibility).read_bytes()).hexdigest() ==
                                       Path(a.eligibility).with_suffix('.FROZEN').read_text().split()[0])
    chain['ladder_eligibility'] = bool(Lr.get('eligibility_sha256') == hashlib.sha256(Path(a.eligibility).read_bytes()).hexdigest())
    chain_ok = all(chain.values())
    cases = [f'{c}|{j}' for c, j in T['config']['cases']]
    assert cases == CASES and int(T['config']['reps']) == len(REPS), 'timing config is not the registered workload'
    veto = set(T.get('veto') or [])
    rng = np.random.default_rng(20261021)
    res = dict(sources={k: dict(path=v, sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest())
                        for k, v in dict(timing=a.timing, ladder=a.ladder, eligibility=a.eligibility, rombox=a.rombox).items()},
               timing_gates=T['gates'], timing_controls={k: v.get('fires') for k, v in T['controls'].items()},
               selftest=selftest(), targets={})
    for C in sorted({float(c) for c in El['contrasts']}):
        el = El['contrasts'][f'{C:g}']
        temporal = Lr['temporal'].get(f'{C:g}', {})
        groups = Lr['candidates'].get(f'{C:g}', {})
        for tau in (0.01, 0.03, 0.10):
            key = f'{tau:g}|C{C:g}'
            row = dict(tau=tau, C=C)
            rom = RB['cheapest_by_target'].get(f'{tau:g}')
            uid = f'union|{tau:g}|C{C:g}'
            sel = T['selection'].get(uid, {})
            fsel = sel.get('selected')
            fam_status = {}
            for f in REQUIRED:
                g = groups.get(f'{f}|{tau:g}|C{C:g}', {})
                tf = temporal.get(f, {}).get('final', 'accepted')
                runs_f = [r for r in Lr['runs'] if r['key'].split('|')[0].upper() == f and float(r['C']) == C]
                unstable = bool(runs_f) and not any(r['valid'] for r in runs_f)
                res_ = 'unresolved' if (g.get('status') != 'ok' or tf == 'unresolved') else 'resolved'
                fam_status[f] = dict(status=res_, all_runs_invalid=unstable, temporal=tf)   # A4.10: no unstable exemption
            row['families'] = fam_status
            unresolved = [f for f, s in fam_status.items() if s['status'] == 'unresolved']
            row['unresolved_families'] = unresolved
            if rom is None:
                row.update(status='inconclusive', reason='no qualified ROM at this target (inherited population)')
                res['targets'][key] = row
                continue
            row['rom'] = dict(label=rom['label'], worst=rom['worst'], inherited_ratio=rom['ratio_to_A'],
                              inherited_seconds=rom['seconds'], qualification='inherited (jcp-qualify FRONT-2D), prior evidence')
            rs, rexact = cost_by_slot(T['stage2'], rom['label'])
            rom_ok = bool(rom['label'] not in veto and scored_ok(T['subjects'].get(rom['label'], {})) and rexact)
            row['rom']['in_job'] = rom_ok
            if not rom_ok:
                rs = {c: rom['ratio_to_A'] for c in cases}
                row['rom']['note'] = 'in-job re-measurement failed: inherited C2 ratio used (same GPU type, same convention)'
            row['rom']['cost'] = float(np.median([rs[c] for c in cases]))
            if rom_ok:
                row['rom']['vs_inherited'] = row['rom']['cost'] / rom['ratio_to_A']
                row['rom']['consistent_20pct'] = bool(0.8 <= row['rom']['vs_inherited'] <= 1.2)
            else:
                row['rom']['vs_inherited'] = row['rom']['consistent_20pct'] = 'not assessed (fallback)'
            if not fsel:
                row.update(status='inconclusive', reason=f'no union winner: {sel.get("status")}')
                res['targets'][key] = row
                continue
            fs, fexact = cost_by_slot(T['stage2'], fsel)
            qual = Lr['qualification'].get(fsel, {}).get('qualified') is True
            f_ok = bool(fsel not in veto and scored_ok(T['subjects'].get(fsel, {})) and fexact and qual)
            # A3.14 decomposition and the paired C = 1 reading (reported, not gating)
            dec_ = {}
            for pre in ('KERNEL', 'SETUP'):
                lb = f'{pre}|{fsel}'
                dec_[pre.lower()] = diag_cost(T, lb, veto, kernel=(pre == 'KERNEL'))
            row['fom'] = dict(key=fsel, family=fsel.split('|')[0].upper(),
                              cost=float(np.median([fs[c] for c in cases])) if f_ok else None,
                              stage2_complete=f_ok, per_family={g_.split('|')[0]: T['selection'][g_].get('selected')
                                                                for g_ in T['selection'] if g_.endswith(f'|{tau:g}|C{C:g}')
                                                                and not g_.startswith('union')}, decomposition=dec_)
            if not f_ok:
                row.update(status='inconclusive', reason='union winner vetoed or incomplete stage 2 (A2.26)')
                res['targets'][key] = row
                continue
            G = row['fom']['cost'] / row['rom']['cost']
            ci = boot(fs, rs, cases, rng)
            # absolute-seconds ratio, reported beside (A2.16)
            tb = lambda lb: float(np.median([np.median([t['tB'] for t in T['stage2'] if t['B'] == lb and t['case'] == c and t.get('valid')])
                                             for c in cases]))
            fsec, rsec = tb(fsel), (tb(rom['label']) if rom_ok else rom['seconds'])
            row.update(status='ok', G=G, G_ci95=ci, marginal=bool(G >= 2 and ci[0] < 2),
                       fom_seconds=fsec, rom_seconds=rsec, seconds_ratio=fsec / rsec,
                       label='anchor-normalised cost ratio; against the strongest tested competitor; finite-ladder upper '
                             'bound on the ROM speed-up')
            res['targets'][key] = row
        # paired PSI reading (DESIGN §9 secondary; A6.3): the C = 100 PSI-family winner against the same configuration at C = 1
        for tau in (0.01, 0.03, 0.10):
            if C != GATE_C:
                continue
            psel = T['selection'].get(f'PSI|{tau:g}|C{C:g}', {}).get('selected')
            if not psel:
                continue
            c1 = psel.replace('@C100', '@C1')
            a_, ea_ = cost_by_slot(T['stage2'], psel)
            b_ = diag_cost(T, c1, veto, per_slot=True)
            if ea_ and isinstance(b_, dict):
                r_ = float(np.median([a_[c] / b_[c] for c in cases]))
                res['targets'][f'{tau:g}|C{C:g}']['psi_C100_over_C1'] = dict(config=psel, ratio=r_,
                                                                            note='median over slots of cost(C=100)/cost(C=1)')
            else:
                res['targets'][f'{tau:g}|C{C:g}']['psi_C100_over_C1'] = dict(config=psel, ratio=None, note=str(b_))
    dec = {t: res['targets'][f'{t:g}|C{GATE_C:g}'] for t in DECISIVE}
    timing_ok = bool(T.get('complete') is True and T.get('backend') == 'gpu' and T['gates'].get('A-valid') is True
                     and 'A' not in veto and scored_ok_A(T)
                     and all(v.get('fires') for v in T['controls'].values()) and set(T['controls']) >= {'T-dbl', 'T-fault', 'T-id'}
                     and 'H200' in str(T.get('gpu')))
    res['timing_valid'] = timing_ok
    res['hash_chain'] = chain
    ref_ok = bool(El['contrasts'][f'{GATE_C:g}']['decidable']) and timing_ok and chain_ok
    gate, outs = decide({t: dict(G=v.get('G'), status=v.get('status'), unresolved_families=v.get('unresolved_families'))
                         for t, v in dec.items()}, ref_ok)
    res['gate'] = dict(outcome=gate, target_outcomes={f'{t:g}': o for t, o in outs.items()}, reference_decidable=ref_ok,
                       contrast=GATE_C, rule='DESIGN §9 + A2.2 + A4.10 + A5.12')
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res['gate'], indent=1))


if __name__ == '__main__':
    main()
