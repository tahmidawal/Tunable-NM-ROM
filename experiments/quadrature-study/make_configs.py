"""Generate every job configuration of the lane from DESIGN.md sections 5-9 (one source of truth).

    python make_configs.py        # writes configs/*.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = '/cluster/tufts/paralab/tawal01/quad2d_20261001'
SETTINGS = ['acc', 'fast', 'head']
GREF, GREF_CHECK, GREF_BAR = 'gauss640', 'gauss768', 1e-5


def rollout_arms(setting, gref=GREF):
    a = [dict(name='dense', kind='dense'), dict(name='lat64', kind='mesh', rule='lat64')]
    if setting == 'head':
        a.append(dict(name='q0scaled', kind='mesh', rule='q0scaled'))
    a += [dict(name=f'gauss{p}', kind='point', rule=f'gauss{p}') for p in (32, 48, 64, 96, 128, 192, 256)]
    a += [dict(name=f'fib{n}', kind='point', rule=f'fib{n}') for n in (1597, 4181, 6765, 17711, 46368)]
    a += [dict(name='sobol4096', kind='point', rule='sobol4096'),
          dict(name='flux_gauss64', kind='flux', rule='gauss64'),
          dict(name='flux_fib6765', kind='flux', rule='fib6765'),
          dict(name='ctrl_smolyak8', kind='point', rule='smolyak8', control=True),
          dict(name='ctrl_gauss8', kind='point', rule='gauss8', control=True),
          dict(name='gref', kind='point', rule=gref)]
    return a


def rho_rules(setting, L):
    r = [dict(name='dense', kind='dense')]
    r += [dict(name=f'lat{s}', kind='mesh', rule=f'lat{s}') for s in (16, 32, 64, 128) if L % s == 0 and s < L]
    if setting == 'head':
        r.append(dict(name='q0scaled', kind='mesh', rule='q0scaled'))
    r += [dict(name=f'gauss{p}', kind='point', rule=f'gauss{p}')
          for p in (8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 200, 256, 320, 400, 512)]
    r += [dict(name=f'fib{n}', kind='point', rule=f'fib{n}')
          for n in (987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393)]
    r += [dict(name=f'sobol{m}', kind='point', rule=f'sobol{m}') for m in (1024, 4096, 16384, 65536)]
    r += [dict(name='halton4096', kind='point', rule='halton4096')]
    r += [dict(name=f'smolyak{l}', kind='point', rule=f'smolyak{l}') for l in (6, 7, 8, 9)]
    r += [dict(name=f'flux_{x}', kind='flux', rule=x) for x in ('gauss64', 'gauss128', 'gauss256', 'fib6765',
                                                                  'fib17711', 'sobol4096')]
    return r


def qstudy(attempt, L, cohorts, dense_cases, refs, timing_cohort, settings=SETTINGS, parity=(), audit_cases=(0, 2),
           gref=GREF, gref_check=GREF_CHECK, case_subset=None, timing_cases=6, reps=3, extra=None):
    c = dict(attempt=attempt, mesh=L, cohorts=cohorts, settings=list(settings), gtol=1e-3, step_budget=600,
             arms={s: rollout_arms(s, gref) for s in settings},
             dense_cases=dense_cases, population_arm='lat64', gref=gref, gref_check=gref_check, gref_bar=GREF_BAR,
             rho_rules={s: rho_rules(s, L) for s in settings},
             refs=refs, fom=dict(truth=dict(name='lean_tight', ntol=1e-6, ltol=1e-8),
                                 timed=[dict(name='lean_tight', ntol=1e-6, ltol=1e-8),
                                        dict(name='lean_nt3e-3_l3e-3_dt005', ntol=3e-3, ltol=3e-3)]),
             timing=dict(cohort=timing_cohort, cases=timing_cases, reps=reps, burn=.1, seed=20261001,
                         arms={s: [a['name'] for a in rollout_arms(s, gref) if not a.get('control')
                                   and not (a['name'] == 'dense' and L > 1024)] for s in settings}),
             audit=dict(cohort=cohorts[0], cases=list(audit_cases), full_max_mesh=256, full_case0_max_mesh=1024,
                        full_arms=['dense', 'lat64', 'gauss64', 'fib6765', 'gref']),
             tangent_chunks=dict(acc=24, fast=8, head=1), dense_chunk=8, point_chunk=32768)
    for x in parity:
        c['arms'][x['setting']].append({k: v for k, v in x.items() if k != 'setting'})
    c['timing']['extra_mesh_arms'] = {s: [n for n in XMESH_ARMS + (['q0scaled'] if s == 'head' else [])] for s in settings}
    if case_subset:
        c['case_subset'] = case_subset
    if extra:
        c.update(extra)
    return c


def xmesh():
    return dict(timing_meshes=[256, 1024])


def parity_256():
    return [dict(setting='acc', name='lat64_g1e-2', kind='mesh', rule='lat64', gtol=1e-2, cohorts=['dev6']),
            dict(setting='fast', name='lat64_g1e-2', kind='mesh', rule='lat64', gtol=1e-2, cohorts=['dev6'])]


PARITY_TARGETS = {  # worst evolved same-grid %, dev6 (DESIGN G3)
    256: dict(acc=('lat64_g1e-2', 0.16561743047161853), fast=('lat64_g1e-2', 1.5967685047599642)),
    1024: dict(acc=('lat64', 0.21081980982218235), fast=('lat64', 1.8280675823814023),
               head=('q0scaled', 2.288356103816248)),
    4096: dict(acc=('lat64', 0.2239874024132454), fast=('lat64', 1.8940229488900593)),
}


def refs(att):
    return f'{NS}/{att}/output'


XMESH_ARMS = ['lat64', 'gauss64', 'gauss128', 'gauss256', 'fib6765', 'fib17711', 'fib46368']


def refjob(attempt, cohorts, audit, mesh=8192, case_limit=None):
    c = dict(attempt=attempt, mesh=mesh, cohorts=cohorts,
             refs=dict(ST=dict(dt=.005 / 16, ntol=1e-11, ltol=1e-9, accept_residual=2e-11),
                       S=dict(dt=.005, ntol=1e-11, ltol=1e-9, accept_residual=2e-11)),
             audit_cases=audit)
    if case_limit:
        c['case_limit'] = case_limit
    return c


def main():
    out = HERE / 'configs'
    out.mkdir(exist_ok=True)
    cfgs = {
        'refdv': refjob('refdv', ['dev6', 'val32'], dict(dev6=[0, 2])),
        'reft': refjob('reft', ['test64'], dict(test64=[0, 1])),
        'dv256': qstudy('dv256', 256, ['dev6', 'val32'], dict(dev6='all', val32='all'), refs('refdv'), 'dev6',
                        parity=parity_256()),
        'dv1024': qstudy('dv1024', 1024, ['dev6', 'val32'], dict(dev6='all', val32='all'), refs('refdv'), 'dev6'),
        'dv4096': qstudy('dv4096', 4096, ['dev6', 'val32'], dict(dev6='all'), refs('refdv'), 'dev6', extra=xmesh()),
        't256': qstudy('t256', 256, ['test64'], dict(test64='all'), refs('reft'), 'test64'),
        't1024': qstudy('t1024', 1024, ['test64'], dict(test64='all'), refs('reft'), 'test64'),
        't4096': qstudy('t4096', 4096, ['test64'], dict(test64=list(range(6))), refs('reft'), 'test64', extra=xmesh()),
        # local smoke (GB10): tiny mesh, two dev cases, a few arms, a cheap continuum rule
        'smk128': qstudy('smk128', 128, ['dev6'], dict(dev6='all'), {}, 'dev6', parity=parity_256(),
                         gref='gauss256', gref_check='gauss320', case_subset=dict(dev6=[0, 2]), timing_cases=1, reps=1,
                         extra=dict(allow_missing_refs=True, local_smoke_waives_cohort_hash=True, timing_meshes=[64])),
        'refsmk': refjob('refsmk', ['dev6'], dict(dev6=[0]), mesh=512, case_limit=1),
    }
    # G6 rollout-level check (DESIGN 7): a Gauss-768 rollout beside gref, dev6, at 1024^2 only
    for s_ in SETTINGS:
        cfgs['dv1024']['arms'][s_].append(dict(name='gref_check', kind='point', rule=GREF_CHECK, cohorts=['dev6']))
    # smoke: thin the arm and rho lists
    keep = {'dense', 'lat64', 'gauss64', 'fib4181', 'flux_gauss64', 'ctrl_smolyak8', 'ctrl_gauss8', 'gref', 'lat64_g1e-2'}
    s = cfgs['smk128']
    s['arms'] = {k: [a for a in v if a['name'] in keep] for k, v in s['arms'].items()}
    s['rho_rules'] = {k: [r for r in v if r['name'] in ('dense', 'lat16', 'lat32', 'lat64', 'gauss32',
                                                         'gauss64', 'gauss128', 'fib4181', 'sobol4096', 'smolyak8',
                                                         'flux_gauss64')] for k, v in s['rho_rules'].items()}
    s['timing']['arms'] = {k: [n for n in v if n in keep] for k, v in s['timing']['arms'].items()}
    s['timing']['extra_mesh_arms'] = {k: ['lat64', 'gauss64'] for k in s['settings']}
    s['case_subset'] = dict(dev6=[2])
    for name, c in cfgs.items():
        (out / f'{name}.json').write_text(json.dumps(c, indent=1) + '\n')
    (out / 'parity_targets.json').write_text(json.dumps({str(k): v for k, v in PARITY_TARGETS.items()}, indent=1) + '\n')
    print(sorted(cfgs))


if __name__ == '__main__':
    main()
