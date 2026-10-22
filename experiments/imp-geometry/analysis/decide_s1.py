"""imp-geometry A10 S1-only readout (kept outside the hashed lane top level so frozen code hashes stay valid).
  python analysis/decide_s1.py --cells <dir with 'L|wide|S1', 'H|wide|S1' subdirectories>"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decide import cell_label, TAUS          # noqa: E402  (the audited decision code)


def s1_readout(root):
    """A10: S1-only readout. Per geometry and target: the A4.2 label, S (val, worst), S_test, best competitor.
    Conditional met iff some target on some geometry is a confirmed candidate win (S >= 2)."""
    res = {}
    met = False
    for geom in ('L', 'H'):
        cdir = root / f'{geom}|wide|S1'
        if not (cdir / 'final.json').exists():
            res[geom] = dict(status='missing')
            continue
        per = {tau: cell_label(cdir, tau) for tau in TAUS}
        res[geom] = per
        met |= any(v.get('confirmation') == 'confirmed' for v in per.values())
    res['A10'] = ('conditional met: S >= 2 confirmed at S1 (S2 and box controls to be funded)' if met else
                  'not supported at S1; S2 untested')
    res['label'] = 'S1-only, wide family; WIN not decidable without S2 and box controls'
    out = Path(__file__).resolve().parent.parent / 'runs' / 'decision_S1.json'
    out.write_text(json.dumps(res, indent=1, default=str))
    print(res['A10'])




if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', required=True)
    s1_readout(Path(ap.parse_args().cells))
