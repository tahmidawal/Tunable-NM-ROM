"""Rebuttal push (PROTOCOL.md comparator rule, amendment A6): pick the accuracy-matched FE entry of a ladder.

Rule (unchanged): the cheapest (median stage wall time) completed ladder entry meeting all six targets on
that case.  Writes the choice and the full ranking to --out and prints the entry as a fe_ladder --entries
list; prints [] (and records it) when no entry meets all six.
Usage: select_fe.py --ladder LADDER-<case>.json --out SELECT.json
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ladder', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
d = json.loads(Path(a.ladder).read_text())
rows = [{'name': n, 'entry': [e['velocity_order'], e['dt_inverse'], e['refactor_every']], 'status': e['status'],
         'targets_met': e.get('targets_met'), 'warm_seconds_median': e.get('warm_seconds_median'),
         'e_u_rms': (e.get('e_u') or {}).get('rms'), 'error': e.get('error')} for n, e in d['entries'].items()]
ok = sorted((r for r in rows if r['status'] == 'completed' and r['targets_met'] == 6), key=lambda r: r['warm_seconds_median'])
sel = {'case': d['case'], 'rule': 'cheapest completed entry meeting all six targets', 'chosen': ok[0] if ok else None, 'entries': rows}
Path(a.out).write_text(json.dumps(sel, indent=2) + '\n')
print(json.dumps([ok[0]['entry']] if ok else []))
