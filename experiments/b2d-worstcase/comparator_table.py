"""Comparators on the fresh cohort, from saved JSON only. Run on the login node:
     ssh tufts-login 'python3 - NS/b1' < comparator_table.py > final-comparators.json"""
import glob, json, os, sys
root = sys.argv[1]


def pct(v, q):
    v = sorted(v); x = q * (len(v) - 1); i = int(x); j = min(i + 1, len(v) - 1)
    return v[i] + (x - i) * (v[j] - v[i])


def flat(e):
    return [max(r) if isinstance(r, list) else r for r in e]


out = dict(operators=[], colora=[])
for f in sorted(glob.glob(os.path.join(root, 'out-ops', '*.json'))):
    r = json.load(open(f)); d = r['L256']
    out['operators'].append(dict(key=r['key'], arm=r['arm'], seed=r['seed'], gpu=r['gpu'], job_id=r['job_id'], cohort=r['cohort'],
                                 L256={k: (v if not isinstance(v, list) else None) for k, v in d.items()},
                                 p95_percent=pct(flat(d['per_case_per_time']), 0.95), max_check=max(flat(d['per_case_per_time']))))
r = json.load(open(os.path.join(root, 'out-colora', 'test-0-9999.json')))
for x in r['results']:
    e = x['errors']
    out['colora'].append(dict(name=x['selection']['name'], role=x['selection']['role'], gpu=r['gpu'], job_id=r['job_id'],
                              cohort_sha256=r['cohort_sha256'], test=x['test'], ms_full_median=x['ms_full']['median'],
                              errors_shape=[len(e), len(e[0]) if isinstance(e[0], list) else 0],
                              p95_of_listed_errors=pct(flat(e), 0.95), max_of_listed_errors=max(flat(e))))
json.dump(out, sys.stdout, indent=1)
