"""Frozen selection from validation results only (PROTOCOL.md sections 4 and 5.2).

  select_frozen.py --crom-grid DIR --colora-val DIR --out selection.json
"""
import argparse, glob, json, os

FAST_GATE = 3.85


def pick_crom(rows):
    sel = []
    for arm in ('E', 'I'):
        c64 = [r for r in rows if r['arm'] == arm and r['dtype'] == 'float64' and r['complete'] and r['nonfinite'] == 0]
        if not c64:   # PROTOCOL 5.2: partial (8-case) settings only if nothing complete exists for the arm
            c64 = [r for r in rows if r['arm'] == arm and r['dtype'] == 'float64' and r['nonfinite'] == 0]
        if not c64:
            continue
        best = min(r['worst_percent'] for r in c64)
        acc = min([r for r in c64 if r['worst_percent'] <= 1.05 * best], key=lambda r: r['rough_ms'])
        chosen = [dict(acc, role='accurate')]
        fast = [r for r in c64 if r['worst_percent'] <= FAST_GATE]
        if fast:
            f = min(fast, key=lambda r: r['rough_ms'])
            if (f['name'], f['setting']) != (acc['name'], acc['setting']):
                chosen.append(dict(f, role='fast'))
        for c in chosen:
            sel.append(c)
            tw = [r for r in rows if r['dtype'] == 'float32' and (r['name'], r['arm'], r['setting']) == (c['name'], c['arm'], c['setting'])]
            if tw and tw[0]['nonfinite'] == 0 and tw[0]['worst_percent'] <= 1.05 * c['worst_percent']:
                sel.append(dict(tw[0], role=c['role'] + '-float32'))
    return sel


def pick_colora(rows):
    sel = []
    k16 = [r for r in rows if r['name'].startswith('K16') and r['dtype'] == 'float64' and r['nonfinite'] == 0]
    if k16:
        b = min(k16, key=lambda r: r['worst_percent'])
        sel.append(dict(b, role='k16'))
        tw = [r for r in rows if r['name'] == b['name'] and r['dtype'] == 'float32']
        if tw and tw[0]['worst_percent'] <= 1.05 * b['worst_percent']:
            sel.append(dict(tw[0], role='k16-float32'))
    for d in ('D7', 'D7-600k'):
        r = [r for r in rows if r['name'] == d and r['dtype'] == 'float64']
        if r:
            sel.append(dict(r[0], role='authors-default'))
    return sel


def merge_chunks(parts):
    """Grid jobs cover case ranges (8-case chunks); merge per (decoder, arm, setting, precision) in case order."""
    import numpy as np
    g = {}
    for r in parts:
        g.setdefault((r['name'], r['arm'], r['setting'], r['dtype']), []).append(r)
    out = []
    for key, rs in sorted(g.items()):
        rs = sorted(rs, key=lambda r: r['cases'][0])
        cov, err, ms = 0, [], []
        for r in rs:
            if r['cases'][0] != cov:
                break
            err += r['errors']; ms += r['ms_per_case']; cov = r['cases'][1]
        err = np.asarray(err)
        fin = np.isfinite(err)
        out.append(dict(name=key[0], arm=key[1], setting=key[2], dtype=key[3], gn=rs[0]['gn'], cases=[0, cov], complete=cov >= 32,
                        worst_percent=float(100 * err.max()) if fin.all() else float('inf'), median_percent=float(100 * np.median(err)),
                        nonfinite=int((~fin).sum()), rough_ms=float(np.median(ms)), weights=os.path.dirname(rs[0]['decoder']),
                        jobs=sorted({r['job_id'] for r in rs}), gpus=sorted({r['gpu'] for r in rs})))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--crom-grid'); a.add_argument('--colora-val'); a.add_argument('--out')
    a = a.parse_args()
    crom = merge_chunks([json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.crom_grid, '*.json')))])
    colora = []
    for f in sorted(glob.glob(os.path.join(a.colora_val, 'val-*.json'))):
        for r in json.load(open(f))['rows']:
            r.pop('errors', None); colora.append(r)
    out = dict(rule='PROTOCOL.md 4 and 5.2', fast_gate_percent=FAST_GATE, crom=pick_crom(crom), colora=pick_colora(colora),
               all_crom_validation_rows=crom, all_colora_validation_rows=colora)
    open(a.out, 'w').write(json.dumps(out, indent=1) + '\n')
    for k in ('crom', 'colora'):
        for s in out[k]:
            print('SELECTED', k, json.dumps({q: s.get(q) for q in ('role', 'name', 'arm', 'setting', 'dtype', 'worst_percent', 'median_percent', 'rough_ms')}))


if __name__ == '__main__':
    main()
