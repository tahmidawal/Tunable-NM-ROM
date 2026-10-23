"""imp-box stop rules (DESIGN §9 verbatim, operational reading A2.8 + A3.3 + A4), evaluated from collected cell JSONs.

  python stoprule.py <runs/cells dir> [--out stoprule.json] [--deferred arm:cell,...] [--fixtures]

Brief (verbatim): "Stop the line if every reduced model has S < 0.5 at every target on the wide family (W >= 0.28
cells). Also stop the bank arm if its projection floor at its own equal-cost size r_x is above tau in every cell."

Entry status per (arm, cell, tau, stat): match (S measured) | arm-censored | arm-unqualified | FOM-censored | missing.
SPEED-STOP: NOT FIRED iff some match has S >= 0.5; FIRES iff no match has S >= 0.5, nothing missing or FOM-censored,
and every arm has >= 1 match; else UNRESOLVED. Scoped reading reported beside it. BANK-STOP: NOT FIRED iff any entry is
below; FIRES iff every required entry is defined and above; else UNRESOLVED.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAUS = (0.01, 0.03, 0.1)
STATS = ('median', 'worst')
WIDE = ('W28', 'W40')
LEVELS = ('A1', 'A2', 'A3', 'A4')
REDUCED = ('podg', 'ecsw', 'trans', 'bank_lspg', 'bank_e4')
SCHEDULE = {'podg': WIDE, 'ecsw': WIDE, 'trans': ('W40',), 'bank_lspg': ('W40',), 'bank_e4': ('W40',)}
S_BAR = 0.5


def is_deferred(arm, cell, deferred):
    return f'{arm}:{cell}' in deferred or f'*:{cell}' in deferred or f'{arm}:*' in deferred


def required(deferred=()):
    req = []
    for arm in REDUCED:
        for W in SCHEDULE[arm]:
            for lev in LEVELS:
                cell = f'{W}_{lev}'
                if is_deferred(arm, cell, deferred):
                    continue
                for tau in TAUS:
                    for stat in STATS:
                        req.append((arm, cell, tau, stat))
    return req


INCOMPLETE = ('search-incomplete', 'missing-construction')


def entry_status(final, arm, tau, stat, accepted=True):
    """From a cell's final.json: (status, S). accepted = scorable references and every mandatory control passed."""
    if final is None or not accepted or not final.get('scorable', False):
        return 'missing', None
    sel = final['selection']
    k = f'{arm}|{tau}|{stat}'
    fom = [sel.get(f'{f}|{tau}|{stat}') for f in ('spec', 'specad')]
    if any(x is None for x in fom) or k not in sel:
        return 'missing', None
    st = [x['status'] for x in fom]
    if any(x in INCOMPLETE or x == 'finalist-partial' for x in st):
        return 'missing', None                                     # either FOM search incomplete (code audit 1)
    if not any(x['finalists'] for x in fom):
        return 'FOM-censored', None
    a = sel[k]
    if a['status'] in INCOMPLETE:
        return 'missing', None
    if not a['finalists']:
        return a['status'], None          # arm-censored | arm-unqualified | operator-check-failed
    s = final['S'].get(k)
    if not s or s.get('S') is None:
        return 'missing', None                                     # timing invalid / incomplete matrix
    if (a['status'] == 'finalist-partial' or s.get('excluded')) and float(s['S']) < S_BAR:
        return 'missing', None   # a partial search / timing-excluded finalist can only show S >= 0.5 (S = max over b)
    return 'match', float(s['S'])


def speed_stop(entries):
    """entries: {(arm, cell, tau, stat): (status, S)} over the required set."""
    matches = {k: v[1] for k, v in entries.items() if v[0] == 'match'}
    missing = sorted(str(k) for k, v in entries.items() if v[0] in ('missing', 'FOM-censored'))
    arms = sorted({k[0] for k in entries})
    arms_matched = sorted({k[0] for k in matches})
    winners = sorted((str(k), s) for k, s in matches.items() if s >= S_BAR)
    if winners:
        out = 'NOT FIRED'
    elif not missing and set(arms_matched) == set(arms) and matches:
        out = 'FIRES'
    else:
        out = 'UNRESOLVED'
    sub = {k: v for k, v in entries.items() if k[0] in arms_matched}
    sc_missing = [str(k) for k, v in sub.items() if v[0] in ('missing', 'FOM-censored')]
    scoped = 'NOT FIRED' if winners else ('FIRES' if (matches and not sc_missing) else 'UNRESOLVED')
    never = sorted(a for a in arms if all(v[0] in ('arm-censored', 'arm-unqualified', 'operator-check-failed')
                                          for k, v in entries.items() if k[0] == a))
    return dict(outcome=out, winners=winners, n_match=len(matches), max_S=max(matches.values()) if matches else None,
                missing=missing, arms=arms, arms_with_match=arms_matched,
                scoped=dict(outcome=scoped, arms=arms_matched, missing=sc_missing,
                            arms_infeasible_everywhere=never,
                            note='scoped reading: arms with >= 1 match only; not a substitute for the registered rule'))


def bank_stop(entries):
    """entries: {(cell, tau, stat): verdict in {above, below, unresolved, no affordable tested rank, missing}}."""
    vals = list(entries.values())
    if any(v == 'below' for v in vals):
        out = 'NOT FIRED'
    elif vals and all(v == 'above' for v in vals):
        out = 'FIRES'
    else:
        out = 'UNRESOLVED'
    return dict(outcome=out, entries={str(k): v for k, v in entries.items()},
                not_affordable=[str(k) for k, v in entries.items() if v == 'no affordable tested rank'])


def evaluate(cells_dir, deferred=()):
    cells_dir = Path(cells_dir)
    finals, diags, acc = {}, {}, {}
    for W in WIDE:
        for lev in LEVELS:
            cell = f'{W}_{lev}'
            cd_ = cell + 't' if (cells_dir / (cell + 't') / 'out' / 'cell' / 'final.json').exists() else cell  # A16
            f = cells_dir / cd_ / 'out' / 'cell' / 'final.json'
            finals[cell] = json.loads(f.read_text()) if f.exists() else None
            cj = cells_dir / cd_ / 'out' / 'cell' / 'controls.json'
            acc[cell] = bool(cj.exists() and json.loads(cj.read_text()).get('all_fire', False))
            g = cells_dir / cd_ / 'out' / 'cell' / 'diag_bank_lspg.json'
            diags[cell] = json.loads(g.read_text()) if g.exists() else None
    ent = {}
    for (arm, cell, tau, stat) in required(deferred):
        ent[(arm, cell, tau, stat)] = entry_status(finals[cell], arm, tau, stat, acc[cell])
    sp = speed_stop(ent)
    bent = {}
    for lev in LEVELS:
        cell = f'W40_{lev}'
        if is_deferred('bank_lspg', cell, deferred) or is_deferred('bank_e4', cell, deferred):
            continue
        for tau in TAUS:
            for stat in STATS:
                d = diags[cell]
                e = None if d is None else d.get('r_x', {}).get('entries', {}).get(f'{tau}|{stat}')
                ok = acc.get(cell, False) and finals.get(cell) is not None and finals[cell].get('scorable', False)
                if ok:                                   # the FOM search behind r_x must be complete (code audit 5)
                    fs = [finals[cell]['selection'].get(f'{f}|{tau}|{stat}', {}).get('status') for f in ('spec', 'specad')]
                    ok = all(x is not None and x not in INCOMPLETE and x != 'finalist-partial' for x in fs)
                bent[(cell, tau, stat)] = 'missing' if (e is None or 'verdict' not in e or not ok) else e['verdict']
    labels = {}
    for W in WIDE:
        for lev in LEVELS:
            cd = cells_dir / f'{W}_{lev}' / 'out' / 'cell'
            if (cd / 'ref.json').exists():
                rj = json.loads((cd / 'ref.json').read_text())
                dj = json.loads((cd / 'data.json').read_text()) if (cd / 'data.json').exists() else {}
                labels[f'{W}_{lev}'] = dict(
                    eligible={k: f"{len(v['eligible']) - v['n_ineligible']}/{len(v['eligible'])}"
                              for k, v in rj['splits'].items()},
                    training_data_qualified={k: v['gate']['passed'] for k, v in dj.items()
                                             if isinstance(v, dict) and 'gate' in v},
                    controls_accepted=acc.get(f'{W}_{lev}', False))
    return dict(speed_stop=sp, bank_stop=bank_stop(bent), deferred=list(deferred), cell_labels=labels,
                entries={str(k): v for k, v in ent.items()})


def fixtures():
    """MF-STOP (A2.9, A4)."""
    k = lambda a, s='median': (a, 'W40_A1', 0.03, s)
    t = {}
    t['all_below_fires'] = speed_stop({k('podg'): ('match', 0.3), k('ecsw'): ('match', 0.2)})['outcome'] == 'FIRES'
    t['one_0.6_not_fired'] = speed_stop({k('podg'): ('match', 0.6), k('ecsw'): ('match', 0.2)})['outcome'] == 'NOT FIRED'
    t['exact_0.5_not_fired'] = speed_stop({k('podg'): ('match', 0.5), k('ecsw'): ('match', 0.2)})['outcome'] == 'NOT FIRED'
    t['0.4999_fires'] = speed_stop({k('podg'): ('match', 0.4999), k('ecsw'): ('match', 0.2)})['outcome'] == 'FIRES'
    t['missing_unresolved'] = speed_stop({k('podg'): ('match', 0.3), k('ecsw'): ('missing', None)})['outcome'] == 'UNRESOLVED'
    t['censored_arm_unresolved'] = speed_stop({k('podg'): ('match', 0.3), k('ecsw'): ('arm-censored', None)})['outcome'] \
        == 'UNRESOLVED'
    t['all_censored_unresolved'] = speed_stop({k('podg'): ('arm-censored', None)})['outcome'] == 'UNRESOLVED'
    t['bank_above_undefined_unresolved'] = bank_stop({1: 'above', 2: 'no affordable tested rank'})['outcome'] == 'UNRESOLVED'
    t['bank_below_undefined_not_fired'] = bank_stop({1: 'below', 2: 'no affordable tested rank'})['outcome'] == 'NOT FIRED'
    t['bank_all_above_fires'] = bank_stop({1: 'above', 2: 'above'})['outcome'] == 'FIRES'
    t['bank_missing_unresolved'] = bank_stop({1: 'above', 2: 'missing'})['outcome'] == 'UNRESOLVED'
    fx = dict(scorable=True, selection={'spec|0.03|median': dict(status='finalist', finalists=[1]),
                                        'specad|0.03|median': dict(status='finalist', finalists=[1]),
                                        'podg|0.03|median': dict(status='finalist', finalists=[1, 2])},
              S={'podg|0.03|median': dict(S=0.3, excluded=['b2'])})
    t['timing_excluded_missing'] = entry_status(fx, 'podg', 0.03, 'median')[0] == 'missing'
    fx['S']['podg|0.03|median'] = dict(S=0.7, excluded=['b2'])
    t['timing_excluded_winner_match'] = entry_status(fx, 'podg', 0.03, 'median') == ('match', 0.7)
    fin = dict(scorable=True, selection={'spec|0.03|median': dict(status='finalist', finalists=[1]),
                                         'specad|0.03|median': dict(status='search-incomplete', finalists=[]),
                                         'podg|0.03|median': dict(status='finalist', finalists=[1])},
               S={'podg|0.03|median': dict(S=0.3)})
    t['fom_incomplete_missing'] = entry_status(fin, 'podg', 0.03, 'median')[0] == 'missing'
    t['unaccepted_missing'] = entry_status(dict(fin, selection=dict(fin['selection'], **{
        'specad|0.03|median': dict(status='finalist', finalists=[1])})), 'podg', 0.03, 'median', accepted=False)[0] \
        == 'missing'
    t['bank_entry_fom_incomplete_missing'] = _bank_fixture()
    t['passed'] = all(t.values())
    return t


def _bank_fixture():
    """Through evaluate(): an incomplete adaptive-FOM search makes the bank entry missing, not 'below'."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cd = Path(td) / 'W40_A1' / 'out' / 'cell'
        cd.mkdir(parents=True)
        sel = {f'{f}|{t}|{s}': dict(status='finalist', finalists=[1]) for f in ('spec', 'specad') for t in TAUS
               for s in STATS}
        sel['specad|0.03|median'] = dict(status='finalist-partial', finalists=[1])
        (cd / 'final.json').write_text(json.dumps(dict(scorable=True, selection=sel, S={})))
        (cd / 'controls.json').write_text(json.dumps(dict(all_fire=True)))
        ents = {f'{t}|{s}': dict(verdict='below') for t in TAUS for s in STATS}
        (cd / 'diag_bank_lspg.json').write_text(json.dumps(dict(r_x=dict(entries=ents))))
        r = evaluate(td)
        return r['bank_stop']['entries'].get(str(('W40_A1', 0.03, 'median'))) == 'missing' and \
            r['bank_stop']['entries'].get(str(('W40_A1', 0.1, 'worst'))) == 'below'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cells', nargs='?')
    ap.add_argument('--out', default=None)
    ap.add_argument('--deferred', default='')
    ap.add_argument('--fixtures', action='store_true')
    a = ap.parse_args()
    if a.fixtures or not a.cells:
        print(json.dumps(fixtures(), indent=1))
        return
    res = evaluate(a.cells, tuple(x for x in a.deferred.split(',') if x))
    res['MF_STOP'] = fixtures()
    s = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(json.dumps(dict(speed=res['speed_stop']['outcome'], scoped=res['speed_stop']['scoped']['outcome'],
                          bank=res['bank_stop']['outcome'], max_S=res['speed_stop']['max_S'],
                          winners=res['speed_stop']['winners'][:10]), indent=1))


if __name__ == '__main__':
    main()
