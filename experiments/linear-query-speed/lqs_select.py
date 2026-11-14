"""Apply the selection rule of PROTOCOL.md mechanically to the development-only results and write the
frozen pair (F: full input, S: sampled input).  Reads only `--mode select` outputs (dev12)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', nargs='+', required=True, help='select-mode result.json files')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs = [json.loads(Path(p).read_text()) for p in a.results]
    assert all(r['mode'] == 'select' and r['complete'] for r in runs)
    big = max(runs, key=lambda r: r['intervals'])
    variants = {}
    for r in runs:
        for row in r['dev_gate']['rows']:
            if row['kind'] != 'variant':
                continue
            v = variants.setdefault(row['variant'], dict(passes=True, checks=[]))
            v['passes'] &= bool(row['passes'])
            v['checks'].append(dict(intervals=r['intervals'], arm=row['arm'], deviation=row['max_deviation_from_paper']))
    meta = {s['variant']: s for s in big['subjects'] if s['kind'] == 'variant'}
    time_ms = {s['variant']: s['median_fused_device_ms'] for s in big['summary']
               if s['kind'] == 'variant' and s['arm'] == 'accurate'}
    frozen = dict(rule='PROTOCOL.md Selection', selection_meshes=[r['intervals'] for r in runs],
                  timing_mesh=big['intervals'], jobs=sorted({str(r['job_id']) for r in runs}),
                  gpu=sorted({str(r['nvidia_smi']) for r in runs}), candidates=[])
    for name, v in sorted(variants.items()):
        frozen['candidates'].append(dict(variant=name, passes=v['passes'], accurate_ms_at_timing_mesh=time_ms.get(name),
                                         max_deviation=max(c['deviation'] for c in v['checks'])))
    for key, is_full in (('F', True), ('S', False)):
        pool = [n for n, v in variants.items() if v['passes'] and n in time_ms and (meta[n]['input'] == 'full') == is_full]
        if not pool:
            frozen[key] = None
            continue
        best = min(time_ms[n] for n in pool)
        tied = [n for n in pool if time_ms[n] <= 1.02 * best]
        pick = sorted(tied, key=lambda n: (meta[n]['precision'] != 'f64', meta[n]['tau']))[0]
        frozen[key] = dict(variant=pick, tau=meta[pick]['tau'], precision=meta[pick]['precision'],
                           input=meta[pick]['input'], accurate_ms_at_timing_mesh=time_ms[pick], tied=tied)
    if frozen['S'] is None:           # no sampled variant passed: S falls back to F and is reported so
        frozen['S'] = dict(frozen['F'], fallback='no sampled-input variant passed the development gate')
    text = json.dumps(frozen, indent=1)
    Path(a.out).write_text(text)
    print('FROZEN', json.dumps({k: frozen[k] and frozen[k]['variant'] for k in ('F', 'S')}),
          hashlib.sha256(text.encode()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
