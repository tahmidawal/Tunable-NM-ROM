"""Final table of the lane, from saved JSON only. Run on the login node:
     ssh tufts-login 'python3 - NS/f1' < final_table.py > final-results.json"""
import glob, json, os, sys
root = sys.argv[1]
out = []
for f in sorted(glob.glob(os.path.join(root, 'out-*', 'result.json'))):
    r = json.load(open(f))
    for name, a in r['arms'].items():
        out.append(dict(dir=os.path.basename(os.path.dirname(f)), job_id=r['job_id'], gpu=r['gpu'], commit=r['commit'],
                        cohort=r['cohort']['name'], cohort_sha256=r['cohort']['sha256'], bank_sha256=r['checkpoint_sha256'],
                        arm=name, cases=a['cases'], failed=a['failed'], worst_pct=100 * a['worst'], median_pct=100 * a['median'],
                        p95_pct=100 * a['p95'], mean_pct=100 * a['mean'], ms=a['ms'], complete=r.get('complete')))
json.dump(out, sys.stdout, indent=1)
