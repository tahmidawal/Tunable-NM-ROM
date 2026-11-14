"""Compact summary of this lane's finished jobs (run on the cluster login node; reads JSON only):
every <NS>/<attempt>/output*/result.json of wc.py and every <NS>/<attempt>/output/history.json of wctrain.py.
  python3 collect.py NS > summary.json"""
import glob, json, os, sys

ns = sys.argv[1]
out = dict(evaluations=[], banks=[])
for f in sorted(glob.glob(f'{ns}/*/output*/result.json')):
    r = json.load(open(f))
    out['evaluations'].append(dict(
        file=os.path.relpath(f, ns), attempt=r['config']['attempt'], job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'],
        complete=r['complete'], checkpoint_sha256=r['checkpoint_sha256'], rotation_sha256=r['rotation_sha256'],
        R=r['R'], cohort=r['cohort'], time_reps=r['config'].get('time_reps'), time_cases=r['config'].get('time_cases'),
        projection={k: {q: v[q] for q in v if q != 'per_time' and not isinstance(v[q], list)} for k, v in r['projection'].items()},
        arms={n: {k: a[k] for k in a if k != 'rows'} for n, a in r['arms'].items()}))
for f in sorted(glob.glob(f'{ns}/*/output/history.json')):
    h = json.load(open(f))
    if 'bank_sha256' not in h:
        continue  # unfinished or cancelled training: no bank
    a = h['args']
    out['banks'].append(dict(
        file=os.path.relpath(f, ns), attempt=f.split('/')[-3], job_id=h['job_id'], commit=h['commit'],
        bank_sha256=h['bank_sha256'], rotation_sha256=h['rotation_sha256'], steps_done=h['steps_done'],
        train_seconds=h['train_seconds'], seconds=h['seconds'], snapshots=h['snapshots'],
        args={k: a.get(k) for k in a}, last_eval=h['evals'][-1]))
json.dump(out, sys.stdout, indent=1)
