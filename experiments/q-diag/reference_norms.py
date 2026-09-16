"""Stage 1b: the reference-field norms the same-grid metric is normalised by."""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--restore', required=True)
ap.add_argument('--job', default='cclad01')
ap.add_argument('--out', required=True)
a = ap.parse_args()
R = Path(a.restore) / a.job / 'output'
r = json.loads((R / 'result.json').read_text())
Path(a.out).write_text(json.dumps(
    {str(e['case']): float(np.linalg.norm(np.load(R / e['artifact'])['fields'][0]))
     for e in r['reference']}))
print('wrote', a.out)
