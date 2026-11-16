"""Run on the cluster inside a job dir: python3 extract.py output > output/result-timing.json
Keeps provenance, driver gates and the six-case validation accuracy medians (sanity only)."""
import json
import statistics as S
import sys

r = json.load(open(sys.argv[1] + '/result.json'))
arm = r['config']['conv_arm']
fom = r['config']['foms'][0]['name']
rom = [x['ST']['e'] for x in r['rows'] if x['arm'] == arm]
fm = [x['ST']['e'] for x in r['fom_rows'] if x['fom'] == fom]
print(json.dumps(dict(job_id=r['job_id'], host=r['host'], gpu=r['gpu'], commit=r['commit'], complete=r['complete'],
                      cohort=r['config']['cohort'], cases=r['cases'], gates=r['gates'],
                      fom_converged=[x['converged'] for x in r['fom_rows'] if x['fom'] == fom],
                      accuracy=dict(rom_ST_median=S.median(rom), rom_ST_max=max(rom), fom_ST_median=S.median(fm),
                                    fom_ST_max=max(fm), n=len(rom))), indent=1))
