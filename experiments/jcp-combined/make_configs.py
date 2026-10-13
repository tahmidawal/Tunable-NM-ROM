"""Write the job configs of jcp-combined (DESIGN 3, A1.0). python make_configs.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_CONTRACT = {"mesh": 8192, "refs": {"ST": {"dt": 0.0003125, "ntol": 1e-11, "ltol": 1e-09, "accept_residual": 2e-11},
                                       "S": {"dt": 0.005, "ntol": 1e-11, "ltol": 1e-09, "accept_residual": 2e-11}}}
SETTINGS2 = [dict(name='fast', Rp=128, rule='fib1597'), dict(name='R128', Rp=128, rule='fib4181'),
             dict(name='R256', Rp=256, rule='gauss96'), dict(name='acc', Rp=384, rule='gauss96'),
             dict(name='R384', Rp=384, rule='gauss128'), dict(name='R512', Rp=512, rule='gauss192')]
FS = [['LSPG', 'BE'], ['LSPG', 'CN'], ['LSPG', 'CNR'], ['LSPG', 'BDF2'], ['GAL', 'CN']]
DTF = [0.5, 1, 5 / 3, 2, 2.5, 5, 10]
FOM = dict(schemes=['BE', 'CN', 'CNR', 'BDF2'], dt_factors=[0.5, 1, 2, 5], ntols=[1e-4, 1e-6, 1e-8],
           order_cases=[['dev6', 0], ['dev6', 2]])


def c2(attempt, mesh=1024, settings=SETTINGS2, case_subset=None, dtf=DTF, fom=FOM, reps=3, refs='REFS_DIR', **kw):
    c = dict(attempt=attempt, driver='c2grid', mesh=mesh, cohorts=['dev6', 'val32'], settings=settings,
             grid=dict(forms_schemes=FS, dt_factors=dtf, anchor_factors=[1 / 8, 1 / 16]), refs=refs,
             ref_contract=REF_CONTRACT, fom=fom, common_baseline='acc',
             timing=dict(cohort='dev6', cases=6, reps=reps, burn=0.1, seed=20261013))
    if case_subset:
        c['case_subset'] = case_subset
    c.update(kw)
    return c


REF3 = 'REF3D'
LADDER = ['lat8192', 'lat16384', 'lat32768', 'lat65536', 'lat131072', 'gl24', 'gl32', 'gl40', 'gl48', 'lat256']
SETTINGS3 = [dict(name='M2_512', bank='M2', Rp=512, a=3), dict(name='W256', bank='W1024', Rp=256, a=6),
             dict(name='W512', bank='W1024', Rp=512, a=6), dict(name='W768', bank='W1024', Rp=768, a=6),
             dict(name='W1024', bank='W1024', Rp=1024, a=6)]


def c3(attempt, mode, settings, count=64, case_subset=None, refined_ref=REF3, timing=None, **kw):
    c = dict(attempt=attempt, driver='c3run', mode=mode, mesh=65, cohort_seed=923801, cohort_count=count,
             refined_ref=refined_ref, gtol=1e-3, trust_fraction=0.05, conv_bar=2.5e-5, settings=settings,
             timing=timing or dict(cases=8, reps=3, burn=0.1, seed=20261013, common_cases=4, common_reps=2),
             grid=dict(dt_factors=[1, 1.25, 5 / 3, 2.5, 5], fs_schemes=['CN', 'CNR', 'BDF2'], Ks=[1, 2],
                       gal_factors=[1, 2.5], anchor_factors=[1 / 8, 1 / 16]),
             common_baseline='W512', common_baseline_fallback='M2_512')
    if case_subset:
        c['case_subset'] = case_subset
    c.update(kw)
    return c


def esc_settings():
    out = []
    for s in SETTINGS3:
        big = s['Rp'] >= 768
        out.append(dict(s, esc_rules=(['gl48', 'gl56', 'gl64'] if big else ['gl56', 'gl64']), diagnostic=big,
                        ladder=LADDER))
    return out


def main():
    cfgs = {
        'c2grid': c2('c2grid'),
        # cluster smoke (A1.0): real mesh, two dev6 cases, every phase, all settings, reduced step ladder
        'smk2d': c2('smk2d', case_subset={'dev6': [0, 2], 'val32': []}, dtf=[0.5, 1, 2.5, 10], reps=1,
                    fom=dict(FOM, dt_factors=[1, 2]), timing_cases_override=2),
        # local machinery run (GB10, one jaxrun, development only): L=256, one case, two settings
        'local2d': c2('local2d', mesh=256, settings=[SETTINGS2[0], dict(name='acc', Rp=384, rule='gauss96')],
                      case_subset={'dev6': [0], 'val32': []}, dtf=[1, 2.5], reps=1,
                      fom=dict(FOM, schemes=['BE', 'CNR'], dt_factors=[1], order_cases=[['dev6', 0]]),
                      refs=str((HERE / '../../../2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/'
                                'archive/output').resolve()), local_smoke_waives_cohort_hash=True),
    }
    cfgs['c3desc'] = c3('c3desc', 'esc', esc_settings(), timing=dict(cases=4, reps=2, burn=0.1, seed=20261013))
    loc3 = Path(HERE / '../../../2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output/'
                'ref_923801.npz').resolve()
    cfgs['local3desc'] = c3('local3desc', 'esc', [dict(esc_settings()[4], ladder=['lat32768', 'lat256'])],
                            case_subset=[0], refined_ref=str(loc3),
                            timing=dict(cases=1, reps=1, burn=0.1, seed=20261013))
    cfgs['local3grid'] = c3('local3grid', 'grid', [dict(SETTINGS3[4], rule='lat32768'), dict(SETTINGS3[2], rule='lat16384')],
                            case_subset=[0], refined_ref=str(loc3),
                            timing=dict(cases=1, reps=1, burn=0.1, seed=20261013, common_cases=1, common_reps=1))
    for k, c in cfgs.items():
        if 'timing_cases_override' in c:
            c['timing']['cases'] = c.pop('timing_cases_override')
        if k == 'local2d':
            c['timing']['cases'] = 1
        (HERE / 'configs').mkdir(exist_ok=True)
        (HERE / 'configs' / f'{k}.json').write_text(json.dumps(c, indent=1) + '\n')
        print(k)


if __name__ == '__main__':
    main()
