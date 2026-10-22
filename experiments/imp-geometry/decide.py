"""imp-geometry decision table (DESIGN A4.2, A3.3, A1.11 zero-field exclusion): reads each cell's ref.json, final.json,
test.json, inter.json (and the box cells') and writes runs/decision.json. Pure NumPy/JSON.

  python decide.py --cells <dir with <dom>|<fam>|<lev> subdirectories, each holding the cell outputs>
"""
import argparse
import json
from pathlib import Path

import score as SC

TAUS = SC.TAUS


def ref_ok(rich, allowance, tau):
    labs = [r['label'] for r in rich]
    return all(l in ('E0', 'corner') for l in labs) and allowance['U'] <= tau / 5


def side(finalists, key):
    return 'matched' if finalists.get(key) else 'not matched'


def cell_label(cdir, tau, stat='worst'):
    ref = json.loads((cdir / 'ref.json').read_text())
    fin = json.loads((cdir / 'final.json').read_text())
    if not (ref.get('MF_MG_finest', {}).get('fires') and ref.get('MF_REF_real')):
        return dict(label='no evidence (control did not fire)')
    dg = cdir / 'diag.json'
    if dg.exists():
        dj = json.loads(dg.read_text())
        if not dj.get('G_FLOOR_ok', True) or dj.get('sem_crosscheck', {}).get('flagged'):
            return dict(label='no evidence (diagnostic failure)')
    else:
        return dict(label='no evidence (diagnostics missing)')
    if ref['zero_baseline']['worst'] <= tau:
        return dict(label='excluded (zero field meets tau)')
    if not ref_ok(ref['splits']['val']['rich'], ref['allowance'], tau):
        return dict(label='no evidence (reference)')
    sm = fin['summary'].get(f'{tau}|{stat}') or {}
    ss = fin['side_status']
    b = ss.get(f'bank|{tau}|{stat}', 'invalid')
    comps = [v for k, v in ss.items() if k.endswith(f'|{tau}|{stat}') and not k.startswith('bank')]
    c = 'matched' if 'matched' in comps else ('censored' if comps and all(x == 'censored' for x in comps) else 'other')
    if b == 'matched' and c == 'matched' and sm.get('S') is not None:
        lab = 'candidate win' if sm['S'] >= 2 else 'loss'
    elif b == 'censored' and c == 'matched':
        lab = 'loss'
    else:
        lab = 'no evidence'
    out = dict(label=lab, S_val=sm.get('S'), best_competitor=sm.get('best_competitor'))
    if lab == 'candidate win':
        tp = cdir / 'test.json'
        if not tp.exists():
            out['confirmation'] = 'pending'
            return out
        t = json.loads(tp.read_text())
        if not ref_ok(t['rich_test'], t['allowance_test'], tau):
            out['confirmation'] = 'unconfirmable'
            return out
        ts = t['summary'].get(f'{tau}|{stat}') or {}
        st = ts.get('S')
        out['S_test'] = st
        out['test_keys'] = dict(bank=ts.get('bank_key'), competitor=ts.get('competitor_key'))
        # test summary pairs are already gated on: both finalists meet tau on test, valid solves, bank Q1/Q2 (A6)
        if st is None and ts.get('reason'):
            out['confirmation'] = 'unconfirmable'          # timing invalid / incomplete: no evidence, not a loss
        else:
            out['confirmation'] = 'confirmed' if (st is not None and st >= 2) else 'not confirmed'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', required=True)
    a = ap.parse_args()
    root = Path(a.cells)
    res = {}
    for geom in ('L', 'H'):
        levels = {}
        for lev in ('S1', 'S2'):
            cdir = root / f'{geom}|wide|{lev}'
            if not (cdir / 'final.json').exists():
                levels[lev] = dict(outcome='no evidence (missing)')
                continue
            per = {tau: cell_label(cdir, tau) for tau in TAUS}
            conf = [t for t, v in per.items() if v.get('confirmation') == 'confirmed']
            loss = [t for t, v in per.items() if v['label'] == 'loss' or v.get('confirmation') == 'not confirmed']
            outc = 'win' if conf else ('loss' if loss else 'no evidence')
            levels[lev] = dict(outcome=outc, per_tau=per, confirmed=conf)
        if all(levels[l]['outcome'] == 'win' for l in levels):
            g = 'WIN'
        elif any(levels[l]['outcome'] == 'loss' for l in levels):
            g = 'WITHDRAWN (measured loss)'
        else:
            g = 'WITHDRAWN (insufficient evidence)'
        res[geom] = dict(levels=levels, outcome=g)
        # attribution (A4.2) is evaluated only for WIN, from the box cells' summaries and the D_bnd gates
        if g == 'WIN':
            att = {}
            for lev, lv in levels.items():
                bdir = root / f'B|wide|{lev}'
                kept = []
                for tau in lv['confirmed']:
                    sg, sgt = lv['per_tau'][tau]['S_val'], lv['per_tau'][tau].get('S_test')
                    blab = cell_label(bdir, tau) if (bdir / 'final.json').exists() else dict(label='no evidence (missing)')
                    if blab['label'].startswith(('no evidence', 'excluded')):
                        att[f'{lev}|{tau}'] = 'unestablished (box: ' + blab['label'] + ')'
                        continue
                    btp = bdir / 'test.json'
                    if btp.exists():
                        bt_j = json.loads(btp.read_text())
                        if not ref_ok(bt_j['rich_test'], bt_j['allowance_test'], tau):
                            att[f'{lev}|{tau}'] = 'unestablished (box test reference unqualified)'
                            continue
                    try:
                        bf = json.loads((bdir / 'final.json').read_text())['summary'][f'{tau}|worst']['S']
                        bt = json.loads((bdir / 'test.json').read_text())['summary'][f'{tau}|worst']['S']
                    except Exception:                       # noqa: BLE001
                        bf = bt = None
                    inter = json.loads((root / f'{geom}|wide|{lev}' / 'inter.json').read_text())['val']
                    tinter = json.loads((root / f'{geom}|wide|{lev}' / 'test.json').read_text()).get('inter_test') or {}
                    if (bf is not None and bf >= sg / 2) or (bt is not None and sgt and bt >= sgt / 2):
                        state = 'withdrawn'
                    elif bf is not None and bt is not None and inter['gate'] and tinter.get('gate'):
                        state = 'kept'
                    else:
                        state = 'unestablished'
                    att[f'{lev}|{tau}'] = state
                    if state == 'kept':
                        kept.append(tau)
                att[lev] = bool(kept)
            res[geom]['attribution'] = att
            res[geom]['outcome'] = 'WIN-attributed' if all(att.get(l) for l in levels) else 'WIN-unattributed'
    order = ['WIN-attributed', 'WIN-unattributed', 'WITHDRAWN (insufficient evidence)', 'WITHDRAWN (measured loss)']
    res['E2'] = min((res[g]['outcome'] for g in ('L', 'H')), key=order.index)
    out = Path(__file__).resolve().parent / 'runs' / 'decision.json'
    out.write_text(json.dumps(res, indent=1, default=str))
    print(res['E2'])


if __name__ == '__main__':
    main()
