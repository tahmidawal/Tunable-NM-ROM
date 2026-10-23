"""imp-geometry A10 S1-only readout (kept outside the hashed lane top level so frozen code hashes stay valid).
  python analysis/decide_s1.py --cells <dir with 'L|wide|S1', 'H|wide|S1' subdirectories>"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decide import cell_label, TAUS          # noqa: E402  (the audited decision code)


def s1_readout(root):
    """A10 S1-only readout (rev. after audits/codex-results-1.md). Per geometry and target: the A4.2 label, the
    descriptive validation and test S (also where the label is 'no evidence'), the best competitor, confirmation.
    Per geometry: 'WITHDRAWN (measured loss)' if any S1 target is a loss or not confirmed (A4.2: a level with a loss
    withdraws the geometry); otherwise 'S1-only, wide family; WIN not decidable without S2 and box controls'.
    A10: 'conditional met' iff some geometry has a confirmed candidate win; 'incomplete' if a cell or its test is
    missing; otherwise 'not supported at S1; S2 untested'."""
    res, met, incomplete = {}, False, False
    for geom in ('L', 'H'):
        cdir = root / f'{geom}|wide|S1'
        if not (cdir / 'final.json').exists() or not (cdir / 'test.json').exists():
            res[geom] = dict(status='missing cell or test')
            incomplete = True
            continue
        fin = json.loads((cdir / 'final.json').read_text())
        tst = json.loads((cdir / 'test.json').read_text())
        per = {}
        for tau in TAUS:
            lab = cell_label(cdir, tau)
            if lab.get('confirmation') == 'pending':
                incomplete = True
            row = dict(lab)
            for stat in ('worst', 'median'):
                v = fin['summary'].get(f'{tau}|{stat}') or {}
                t = tst['summary'].get(f'{tau}|{stat}') or {}
                row[f'descriptive_S_val_{stat}'] = v.get('S')
                row[f'descriptive_S_test_{stat}'] = t.get('S')
                row[f'best_competitor_{stat}'] = v.get('best_competitor')
            per[str(tau)] = row
        loss = any(v['label'] == 'loss' or v.get('confirmation') == 'not confirmed' for v in per.values())
        met |= any(v.get('confirmation') == 'confirmed' for v in per.values())
        res[geom] = dict(per_tau=per, geometry_label=('WITHDRAWN (measured loss)' if loss else
                                                      'S1-only, wide family; WIN not decidable without S2 and box controls'))
    res['A10'] = ('incomplete' if incomplete else
                  'conditional met: S >= 2 confirmed at S1 (S2 and box controls to be funded)' if met else
                  'not supported at S1; S2 untested')
    out = Path(__file__).resolve().parent.parent / 'runs' / 'decision_S1.json'
    out.write_text(json.dumps(res, indent=1, default=str))
    print(res['A10'], {g: res[g].get('geometry_label') for g in ('L', 'H')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', required=True)
    s1_readout(Path(ap.parse_args().cells))
