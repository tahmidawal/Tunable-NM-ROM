"""Second rebuttal push, PROTOCOL.md A7: rank fe_ladder2 entries and print the cheapest passing ones.

Rule: completed entries meeting all six targets, ordered by median stage wall time; the first --top are printed
as a fe_ladder2 --entries list ([] when none passes).  --always adds entries (JSON list) that are run regardless.
Usage: fe_select2.py --ladder LADDER2.json [more.json] --top 5 --out SELECT.json [--always '[[...]]']
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ladder', required=True, nargs='+'); ap.add_argument('--out', required=True)
ap.add_argument('--top', type=int, required=True); ap.add_argument('--always', default='[]')
a = ap.parse_args()
rows, case = [], None
for f in a.ladder:
    d = json.loads(Path(f).read_text()); case = d['case']
    rows += [{'name': n, 'entry': e['entry'], 'status': e['status'], 'ndof': e.get('ndof'), 'targets_met': e.get('targets_met'),
              'warm_seconds_median': e.get('warm_seconds_median'), 'warm_seconds_min': e.get('warm_seconds_min'),
              'warm_seconds_max': e.get('warm_seconds_max'), 'e_u_rms': (e.get('e_u') or {}).get('rms'),
              'error': e.get('error')} for n, e in d['entries'].items()]
ok = sorted((r for r in rows if r['status'] == 'completed' and r['targets_met'] == 6), key=lambda r: r['warm_seconds_median'])
chosen = [r['entry'] for r in ok[:a.top]]
for e in json.loads(a.always):
    if e not in chosen:
        chosen.append(e)
Path(a.out).write_text(json.dumps({'case': case, 'rule': f'{a.top} cheapest completed entries meeting all six targets',
                                   'passing_ranked': ok, 'chosen': chosen, 'cheapest': ok[0] if ok else None,
                                   'entries': rows}, indent=2) + '\n')
print(json.dumps(chosen))
