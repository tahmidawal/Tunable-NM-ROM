"""Headline cost table of amendment A6, computed only from the saved JSON of rb-ho1 and rb-devt."""
import json, sys
RULE = {'bank_r32': '0.0025', 'bank_p24': '0.0025', 'bank_p16': '0.00125',
        'pod_r32': '0.0025', 'pod_r24': '0.0025', 'pod_r16': '0.00125'}
out = {}
for a, c in (('rb-devt', 're95'), ('rb-ho1', 're105')):
    fe = json.load(open(f'artifacts/{a}/results/ladder_timed/LADDER-{c}.json'))['entries']
    (fn, f), = fe.items()
    tb = json.load(open(f'artifacts/{a}/results/timing/ROM-before.json'))
    ta = json.load(open(f'artifacts/{a}/results/timing/ROM-after.json'))
    rows = {}
    for arm, dt in RULE.items():
        b, e = tb['arms'][arm], ta['arms'][arm]
        warm = 0.5 * (b['steps'][dt]['warm_median'] + e['steps'][dt]['warm_median'])
        outp = 0.5 * (b['steps'][dt]['output_median'] + e['steps'][dt]['output_median'])
        rows[arm] = dict(dt=dt, rollout_seconds=warm, output_seconds=outp, e_u_rms=b['steps'][dt]['e_u_rms'],
                         build_seconds=0.5 * (b['build_seconds'] + e['build_seconds']),
                         full_quadrature_output_seconds=0.5 * (b['full_quadrature_output_seconds'] + e['full_quadrature_output_seconds']),
                         speedup_rollout_only=f['warm_seconds_median'] / warm,
                         speedup_with_output=f['warm_seconds_median'] / (warm + outp))
    out[c] = dict(attempt=a, host=tb['host'], cpu=tb['cpu_model'], threads=tb['threads'], fe_entry=fn,
                  fe_seconds_median=f['warm_seconds_median'], fe_seconds_min=f['warm_seconds_min'],
                  fe_seconds_max=f['warm_seconds_max'], fe_targets_met=f['targets_met'], rom=rows)
json.dump(out, open('artifacts/HEADLINE-A6.json', 'w'), indent=1)
for c, o in out.items():
    print(c, o['host'], o['cpu'], o['fe_entry'], f"FE {o['fe_seconds_median']:.2f} [{o['fe_seconds_min']:.2f},{o['fe_seconds_max']:.2f}]")
    for arm, r in o['rom'].items():
        print(f"  {arm:9s} dt {r['dt']:8s} roll {r['rollout_seconds']:.3f} out {r['output_seconds']:.3f} build {r['build_seconds']:.1f} fullq {r['full_quadrature_output_seconds']:.1f} e_u {r['e_u_rms']:.3e} x{r['speedup_rollout_only']:.1f} x_with_output {r['speedup_with_output']:.1f}")
