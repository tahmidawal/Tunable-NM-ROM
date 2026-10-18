"""jcp-heads smoke-derived cost certificates (DESIGN 8, A1.9, A2.7; qualify resources.py pattern, rewritten for this
lane). Writes checks/resources-<attempt>.json with projection_hours and time_limit_hours = ceil_0.25(1.5 x projection);
cluster/stage.py requires the time limit to cover 1.5 x projection and binds the certificate to the exact configs/runs.
ESTIMATES, not bounds: the scheduler limit is the enforced bound. Every constant is read from the smoke archives.

Measurements (smk1, A100-80G; smk2, H200):
  train   per-step fit time: the incumbent's own record (dn256b stage C, A100: 'mid' fit seconds / 200,000 steps), never
          below the smoke's short-fit seconds per step; evaluation seconds per arm = smoke arm seconds - its fit seconds;
          K factor = smoke K=32 fit / K=4 fit (>= 1)
  rule    seconds per arm-rung = max over the smoke's rule groups of group seconds / (arms x rungs tried); a rung's cost
          scales with max(1, m / m_ref), m_ref = the largest rule the smoke rule probe reached; WORST CASE: every group
          walks the whole selectable ladder (16 .. 192)
  acc     seconds per (arm, scheme, case) = max over smk1 arms of arm-scheme seconds / cases, scaled by
          max(1, m / m_measured) for the arm's selected rule, plus a per-(arm, scheme) compile allowance = the largest
          smk1 arm-scheme time on two cases
  time    from smk2 (H200): seconds per triplet (split + fused) and per certification case, per subject compile allowance
    python resources.py <attempt>
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
INC_HFIT = HERE.parent / 'separable-decoder/runs/dn256b/out/hfit_dn256b_full.json'
LADDER = [16, 24, 32, 48, 64, 96, 128, 160, 192]


def out(att, drv, stem):
    return RUNS / att / 'archive' / 'output' / f'{drv}_{stem}'


def jl(p):
    return json.loads(Path(p).read_text())


def m_of(rule):
    return int(rule[5:]) ** 2


def train_proj(cfg):
    tr = jl(out('smk1', 'htrain', 'smk1train') / 'result.json')
    inc = jl(INC_HFIT)['arms']['mid']['train']
    per_step = max(inc['seconds'] / inc['steps_done'],                    # conservative: the slower of the incumbent's
                   max(tr['arms'][a]['train']['seconds'] / tr['arms'][a]['train']['steps_done'] for a in ('smk_k4_s0', 'smk_k32_s0')))
    #                                                                     full-fit rate and the smoke's (compile-inclusive) rate
    kfac = max(1., tr['arms']['smk_k32_s0']['train']['seconds'] / tr['arms']['smk_k4_s0']['train']['seconds'])
    evals = max(tr['arms'][a]['seconds'] - tr['arms'][a]['train']['seconds'] for a in ('smk_k4_s0', 'smk_k32_s0'))
    ident = tr['arms']['identity']['seconds']
    tot, det = 120., {}
    for arm in cfg['arms']:
        s = ident if arm.get('identity') else arm.get('steps', 200000) * per_step * (kfac if arm['k'] >= 16 else 1.) + evals
        det[arm['name']] = s
        tot += s
    return tot, dict(per_step=per_step, k_factor=kfac, eval_seconds=evals, setup=120., per_arm=det)


def rule_proj(cfg):
    ru = jl(out('smk1', 'hrule', 'smk1rule') / 'result.json')
    per = max(v['seconds'] / (len(v['arms']) * len(v['trail'])) for v in ru['groups'].values())
    m_ref = max(m_of(t['rule']) for v in ru['groups'].values() for t in v['trail'])
    # worst case: whole selectable ladder; each rung's cost scaled by its SUCCESSOR's point count (the q+ / q++ solves
    # are the largest of the rung), times ALLOW for dimension (r = 384, M = 1536) and trained-head iteration counts
    # ALLOW measured: in smk1's accuracy job the costliest two-case arm-scheme (any K or r, incl. r = 384 / M = 1536)
    # over the head arm the rule probe timed (Hk4)
    a = jl(out('smk1', 'hacc', 'smk1acc') / 'result.json')
    two = {n: v['arms']['BE']['seconds'] for n, v in a['arms'].items() if v['spec']['cases'] != 'all38'}
    ALLOW = max(1., max(two.values()) / two['Hk4'])
    full = LADDER + [256, 320]
    succ = [max(m_of(f'gauss{full[i + 1]}'), m_of(f'gauss{full[i + 2]}')) for i in range(len(LADDER))]   # q+ and q++
    rung_cost = sum(max(1., m / m_ref) for m in succ)
    tot = 300. + ALLOW * sum(len(g['arms']) * per * rung_cost for g in cfg['groups'])
    return tot, dict(seconds_per_arm_rung=per, m_ref=m_ref, worst_case_rung_units=rung_cost, allowance=ALLOW, setup=300.)


def acc_proj(cfg):
    a = jl(out('smk1', 'hacc', 'smk1acc') / 'result.json')
    per_case, comp, m_meas = 0., 0., 0
    for n, v in a['arms'].items():
        nc = 38 if v['spec']['cases'] == 'all38' else len(v['spec']['cases'])
        for sc, x in v['arms'].items():
            if nc == 38:                                   # real-size per-case cost (incl. tight, tighter, q+, q++)
                per_case = max(per_case, x['seconds'] / nc)
                m_meas = max(m_meas, v['m']['main'])
            else:                                          # two-case arms: compile-dominated -> compile allowance
                comp = max(comp, x['seconds'])
    tot = 300. + a['elapsed'].get('controls', 0.)
    det = {}
    for arm in cfg['arms']:
        nc = 38 if arm.get('cases', 'all38') == 'all38' else len(arm['cases'])
        f = max(1., m_of(arm['rule']) / m_meas)
        s_ = len(arm.get('schemes', ['BE'])) * (comp + nc * per_case * f)
        det[arm['name']] = s_
        tot += s_
    return tot, dict(seconds_per_case=per_case, compile_allowance=comp, m_measured=m_meas, per_arm=det)


def time_proj(cfg):
    t = jl(out('smk2', 'htime', 'smk2time') / 'result.json')
    trip = sorted(x['tA1'] + x['tA2'] + x['t_init'] + x['t_evolve'] + x['t_decode'] for x in t['invocations'])
    fus = sorted(x['tA1'] + x['tA2'] + x['t_fused'] for x in t['fused'])
    burn = 8 * float(t['config']['burn'])                     # 5 burns per split triplet + 3 per fused
    per_trip = trip[-1] + fus[-1] + burn                       # the slowest smoke triplet pair (retries included below)
    nsub = len(t['subjects'])
    cert_per_subject = (t['elapsed_seconds'] - sum(trip) - sum(fus)) / nsub   # certification + compile per subject (upper)
    reps = cfg['reps']
    cases = cfg['cases']
    from p1common import case_list, select  # noqa: E402  (JAX import is fine on CPU)
    cl = case_list({})
    idx = select(cl, cases)
    n_trip = sum(int(reps[cl[i][0]]) for i in idx) * len(cfg['subjects'])
    tot = 600. + (len(cfg['subjects']) + 1) * cert_per_subject + n_trip * per_trip * 1.5      # 1.5: drift retries
    return tot, dict(per_triplet_pair=per_trip, cert_per_subject=cert_per_subject, n_triplets=n_trip, setup=600.)


PROJ = dict(htrain=train_proj, hrule=rule_proj, hacc=acc_proj, htime=time_proj)


def main():
    att = sys.argv[1]
    runs = [r.split(':', 1) for r in sys.argv[2:]]
    tot, det, cfgs = 0., {}, {}
    for drv, cf in runs:
        cfg = jl(HERE / cf)
        s, d = PROJ[drv](cfg)
        tot += s
        det[f'{drv}:{cf}'] = dict(seconds=s, **d)
        cfgs[cf] = hashlib.sha256((HERE / cf).read_bytes()).hexdigest()
    ph = tot / 3600
    cert = dict(attempt=att, runs=[f'{d}:{c}' for d, c in runs], configs=cfgs, projection_hours=ph,
                time_limit_hours=math.ceil(1.5 * ph * 4) / 4, detail=det,
                note='estimate from smoke archives (resources.py docstring); the scheduler limit is the enforced bound')
    p = HERE / 'checks' / f'resources-{att}.json'
    p.write_text(json.dumps(cert, indent=1) + '\n')
    print(json.dumps({k: v for k, v in cert.items() if k != 'detail'}, indent=1))


if __name__ == '__main__':
    main()
