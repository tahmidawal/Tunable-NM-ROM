"""Apply the selection rule of PROTOCOL.md (and amendment A1) to the validation summary and write selection.json plus
the final-evaluation configs. Reads validation results only.   python freeze.py validation-summary.json
Rows:  primary    lowest validation worst, standard solve, admissible, time <= 1.25 x deployed (same job)
       in_budget  the same over banks trained on the original 4,352 trajectories only (written if primary used extra data)
       fast       lowest validation worst over the fast-solve arms (admissible; no time cap needed)
       fast_in_budget  the same over banks trained on the original 4,352 trajectories only (if fast used extra data; A3)
Ties: all configurations within 5 % of the lowest validation worst are tied; the fastest of them is taken."""
import json, sys
import make_config as mc

DEPLOYED = '18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'
NS = '/cluster/tufts/paralab/tawal01/b2d-worstcase-20261114'
VAL_SHA = '69fee961fb01d0d2c3e1499bff6d6df6cbc883af1a73e4eb4b981d0402249e2d'
S = json.load(open(sys.argv[1]))
banks = {b['bank_sha256']: b for b in S['banks']}
evs = [e for e in S['evaluations'] if e['complete'] and e['cohort']['sha256'] == VAL_SHA and e['attempt'] != 'diag']
ref_ms = {e['file'].split('/')[0]: e['arms']['R384_M1536_lat64_g0.01']['ms']
          for e in evs if e['checkpoint_sha256'] == DEPLOYED and e['attempt'] == 'ref'}
assert ref_ms
rows = []
for e in evs:
    job = e['file'].split('/')[0]
    dep = e['checkpoint_sha256'] == DEPLOYED
    b = None if dep else banks[e['checkpoint_sha256']]
    for name, a in e['arms'].items():
        s = a['spec']
        if s['Rp'] != e['R'] and not dep or a['cases'] != 256:
            continue   # candidates are evaluated at the native width on all 256 validation cases
        same = job in ref_ms
        rows.append(dict(
            arm=name, spec=s, gpu=e['gpu'], validation_file=e['file'], validation_job=e['job_id'], bank_sha256=e['checkpoint_sha256'],
            rotation_sha256=e['rotation_sha256'], bank='deployed' if dep else b['attempt'],
            extra_trajectories=0 if dep else ((int(b['args'].get('extra_count') or 0) or 2304) if b['args'].get('extra_pool') else 0),
            fast=bool(s.get('qs')), worst=a['worst'], median=a['median'], p95=a['p95'], ms=a['ms'], failed=a['failed'],
            deployed_ms=ref_ms[job] if same else min(ref_ms.values()), deployed_ms_same_job=same))
# a (bank, arm) validated in several jobs counts once: an A100 80GB job before any other GPU (A4), then the latest job
latest = {}
for r in rows:
    k = (r['bank_sha256'], r['arm'])
    rank = ('A100 80GB' in r['gpu'], int(r['validation_job']))
    if k not in latest or rank > latest[k][0]:
        latest[k] = (rank, r)
latest = {k: v[1] for k, v in latest.items()}
rows = list(latest.values())
for r in rows:
    r['admissible'] = bool(r['median'] <= 1e-3 and r['failed'] == 0)
    r['within_time'] = bool(r['ms'] <= 1.25 * r['deployed_ms'])


def pick(c):
    if not c:
        return None
    lo = min(r['worst'] for r in c)
    tied = [r for r in c if r['worst'] <= 1.05 * lo]
    return dict(min(tied, key=lambda r: r['ms']), lowest_worst=lo, tied=[(r['bank'], r['arm']) for r in tied])


std = [r for r in rows if r['admissible'] and r['within_time'] and not r['fast']]
sel = dict(primary=pick(std))
if sel['primary']['extra_trajectories']:
    sel['in_budget'] = pick([r for r in std if r['extra_trajectories'] == 0 and r['bank'] != 'deployed'])
fast = [r for r in rows if r['admissible'] and r['fast'] and r['bank'] != 'deployed']
sel['fast'] = pick(fast)
if sel['fast']['extra_trajectories']:      # amendment A3
    sel['fast_in_budget'] = pick([r for r in fast if r['extra_trajectories'] == 0])
out = dict(rule='PROTOCOL.md selection rule + amendment A1', validation_cohort_sha256=VAL_SHA, selected=sel,
           candidates=sorted(rows, key=lambda r: r['worst']),
           banks={k: dict(attempt=b['attempt'], job_id=b['job_id'], args=b['args'], seconds=b['seconds'],
                          rotation_sha256=b['rotation_sha256']) for k, b in banks.items()},
           test_cohorts_seen=False)
json.dump(out, open('selection.json', 'w'), indent=1)

# final configs: one per (cohort, bank); the deployed setting is run in the same job
by_bank = {}
for k, r in sel.items():
    if r:
        by_bank.setdefault(r['bank'], []).append(r)
plan = []
for cn, coh in (('test64', mc.TEST), ('fresh64', mc.FRESH)):
    json.dump(dict(mc.FINAL, attempt=f'final-{cn}-deployed', cohort=coh, projection_widths=[384], arms=[mc.arm(384)]),
              open(f'config-final-{cn}-deployed.json', 'w'), indent=1)
    plan.append(dict(cohort=cn, bank='deployed', config=f'config-final-{cn}-deployed.json', checkpoint='$CK', rotation='$ROT0'))
    for bk, rs in by_bank.items():
        R = rs[0]['spec']['Rp']
        arms, seen = [], set()
        for r in rs:
            if r['arm'] not in seen:
                seen.add(r['arm']); arms.append(r['spec'])
        json.dump(dict(mc.FINAL, attempt=f'final-{cn}-{bk}', cohort=coh, projection_widths=[R],
                       ladder=sorted(set(mc.LADDER[:8] + [R])), arms=arms), open(f'config-final-{cn}-{bk}.json', 'w'), indent=1)
        plan.append(dict(cohort=cn, bank=bk, config=f'config-final-{cn}-{bk}.json', checkpoint=f'{NS}/{bk}/output/bank.pkl',
                         rotation=f'{NS}/{bk}/output/rotation.npz', bank_sha256=rs[0]['bank_sha256'], arms=sorted(seen)))
json.dump(plan, open('final-plan.json', 'w'), indent=1)
for k, r in sel.items():
    print(k, None if r is None else (r['bank'], r['arm'], f"worst {100*r['worst']:.3f} median {100*r['median']:.4f} "
                                     f"p95 {100*r['p95']:.3f} ms {r['ms']:.2f} cap {1.25*r['deployed_ms']:.1f} tied {r['tied']}"))
