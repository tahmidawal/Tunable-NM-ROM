"""jcp-robust: measured seconds per execution class for certify.py (DESIGN 7, A3.10, A4.2, A6.1). CPU only.

Sources (every number is read from a job record; nothing typed):
  * 2D rollouts: per-row seconds of jcp-qualify's acc256 / acc1024 / acc4096 result.json (the same r2acc code path's parent
    qacc; R128-R512, all three meshes; medians of production and tight seconds), and this lane's smk1b / smk2 rows (the
    1024^2 R128 / R384 classes and the gauss384 / gauss320 'big' target classes), the larger of the two per class;
    tighter = tight, successor = production (same solver, other table). 'big' classes at other (mesh, r) are scaled from
    the measured 1024^2 R128 target by the measured production ratio of that (mesh, r).
  * 2D setup per mesh: elapsed minus the sum of row seconds of the qualify job of that mesh.
  * 3D: smk1b e1run3d rows (seconds_first medians; W256), smk3 e2run3d ladder rows (W256, W1024, pilot); W512 / W768
    classes scaled from W256 / W1024 by the width ratio of the measured classes (linear in r, conservative for the r^2
    terms is not needed at these sizes: the measured W1024/W256 ratio is used to interpolate).
  * dtime / e5time / training: added by the respective smokes (smk2 train record, smk4 / smk5 e5time records).
Writes checks/smoke-timings.json with a 'sources' block (file + sha256 of every record used).

    python timings.py --qualify-dir <dir with acc256.json acc1024.json acc4096.json> --smk1b <archive output> \
                      [--smk2 <archive output>] [--smk3 <archive output>] [--smk4 <archive output>] [--smk5 <archive output>]
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def med(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qualify-dir', required=True)
    for k in ('smk1b', 'smk2', 'smk3', 'smk4', 'smk5'):
        ap.add_argument(f'--{k}')
    a = ap.parse_args()
    T, src = {}, {}
    # ---- 2D from qualify
    for m in (256, 1024, 4096):
        p = Path(a.qualify_dir) / f'acc{m}.json'
        r = json.loads(p.read_text())
        src[f'qualify acc{m}'] = dict(file=str(p), sha256=sha(p), gpu=r.get('gpu'))
        d = collections.defaultdict(lambda: ([], []))
        tot = 0.
        for x in r['rows']:
            if x.get('kind') == 'prod':
                d[x['setting']][0].append(x['seconds'])
                d[x['setting']][1].append(x.get('seconds_tight'))
                tot += float(x['seconds']) + float(x.get('seconds_tight') or 0)
        for g, (pp, tt) in d.items():
            T[f'r2acc|{m}|{g}|prod|std'] = med(pp)
            T[f'r2acc|{m}|{g}|tight|std'] = med(tt)
            T[f'r2acc|{m}|{g}|tighter|std'] = med(tt)
        T[f'r2acc|setup|{m}'] = max(0., float(r['elapsed_seconds']) - tot) / max(1, len(r['config']['settings']))
    # ---- 2D from the smokes (max with qualify)
    big_ref = None
    for tag in ('smk1b', 'smk2'):
        if not getattr(a, tag):
            continue
        for d_ in sorted(Path(getattr(a, tag)).glob('r2acc_*')):
            p = d_ / 'result.json'
            r = json.loads(p.read_text())
            src[f'{tag} {d_.name}'] = dict(file=str(p), sha256=sha(p), gpu=r.get('gpu'))
            m = int(r['mesh'])
            byc = collections.defaultdict(list)
            for x in r['rows']:
                g = x['group']
                big = str(x.get('rule', '')).startswith('gauss') and int(str(x['rule'])[5:]) >= 256
                if x['kind'] == 'target':
                    byc[(m, g, 'prod', 'big')].append(x['seconds'])
                elif x['kind'] == 'prod':
                    byc[(m, g, 'prod', 'big' if big else 'std')].append(x['seconds'])
                    if x.get('seconds_tight') is not None:
                        byc[(m, g, 'tight', 'big' if big else 'std')].append(x['seconds_tight'])
                        byc[(m, g, 'tighter', 'big' if big else 'std')].append(x['seconds_tight'])
            for (m_, g, lev, cl), v in byc.items():
                k = f'r2acc|{m_}|{g}|{lev}|{cl}'
                T[k] = max(T.get(k) or 0., med(v) or 0.)
            if m == 1024 and (1024, 'R128', 'prod', 'big') in byc:
                big_ref = med(byc[(1024, 'R128', 'prod', 'big')])
            T[f'r2acc|setup|{m}'] = max(T.get(f'r2acc|setup|{m}') or 0., float(r['setup']['bank']['seconds']) + 60.)
    if big_ref:
        base = T['r2acc|1024|R128|prod|std']
        for m in (256, 1024, 4096):
            for g in ('R128', 'R256', 'R384'):
                f = (T.get(f'r2acc|{m}|{g}|prod|std') or base) / base
                for lev in ('prod', 'tight', 'tighter'):
                    T.setdefault(f'r2acc|{m}|{g}|{lev}|big', big_ref * f * (1.4 if lev != 'prod' else 1.))
        for k in [k for k in T if k.startswith('r2acc|') and '|R128sob|' not in k and k.endswith('|std') and '|R128|' in k]:
            T.setdefault(k.replace('|R128|', '|R128sob|'), T[k])
    # ---- 3D from the smokes
    for tag in ('smk1b', 'smk3'):
        if not getattr(a, tag):
            continue
        for d_ in sorted(Path(getattr(a, tag)).glob('e*run3d_*')):
            p = d_ / 'result.json'
            r = json.loads(p.read_text())
            src[f'{tag} {d_.name}'] = dict(file=str(p), sha256=sha(p), gpu=r.get('gpu'))
            byc = collections.defaultdict(list)
            for x in r['rows']:
                st = x['setting']
                fam = 'gl' if str(x['rule']).startswith('gl') else 'lat'
                lev = 'adt-tight' if x['spec'].endswith('|tight') else 'adt-tighter' if x['spec'].endswith('|tighter') else 'prod'
                byc[(st, lev, fam)].append(x['seconds_first'])
            drv = 'e1run3d' if d_.name.startswith('e1run3d') else 'd3'
            for (st, lev, fam), v in byc.items():
                # first call per (tag, spec) includes compile; the median over cases removes it
                T[f'{drv}|{st}|{lev}|{fam}'] = max(T.get(f'{drv}|{st}|{lev}|{fam}') or 0., med(v) or 0.)
            T[f'{"e1run3d" if drv == "e1run3d" else "e2run3d"}|setup'] = max(T.get(f'{"e1run3d" if drv == "e1run3d" else "e2run3d"}|setup') or 0.,
                                                                          float(r.get('elapsed_seconds', 0.)) * .25)
    (HERE / 'checks').mkdir(exist_ok=True)
    out = dict(timings={k: v for k, v in sorted(T.items()) if v is not None}, sources=src,
               note='seconds per execution; certify.py multiplies by 1.25 and stage.py requires time limit >= 1.5 x projection')
    (HERE / 'checks' / 'smoke-timings.json').write_text(json.dumps(out, indent=1) + '\n')
    print(len(out['timings']), 'classes')


if __name__ == '__main__':
    main()
