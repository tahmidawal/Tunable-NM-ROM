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
             timing=dict(cohort='dev6', cases=6, reps=reps, burn=0.1, seed=20261013, common_seed=20261013))
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
             timing=timing or dict(cases=8, reps=3, burn=0.1, seed=20261013, common_cases=4, common_reps=2, common_seed=20261013),
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
    es = esc_settings()
    cfgs['c3descW1024'] = c3('c3descW1024', 'esc', [es[4]], timing=dict(cases=4, reps=2, burn=0.1, seed=20261013))
    cfgs['c3descW768'] = c3('c3descW768', 'esc', [es[3]], timing=dict(cases=4, reps=2, burn=0.1, seed=20261013))
    cfgs['c3descB'] = c3('c3descB', 'esc', [es[0], es[1], es[2]], timing=dict(cases=4, reps=2, burn=0.1, seed=20261013))
    loc3 = Path(HERE / '../../../2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output/'
                'ref_923801.npz').resolve()
    cfgs['local3desc'] = c3('local3desc', 'esc', [dict(esc_settings()[1], ladder=['lat8192', 'lat256'])],
                            case_subset=[0], refined_ref=str(loc3),
                            timing=dict(cases=1, reps=1, burn=0.1, seed=20261013))
    lg = c3('local3grid', 'grid', [dict(SETTINGS3[1], rule='lat8192'), dict(SETTINGS3[0], rule='gl24')],
            case_subset=[0], refined_ref=str(loc3),
            timing=dict(cases=1, reps=1, burn=0.1, seed=20261013, common_cases=1, common_reps=1, common_seed=20261013))
    lg['grid'] = dict(dt_factors=[1, 2.5], fs_schemes=['CN'], Ks=[1, 2], gal_factors=[2.5], anchor_factors=[1 / 8])
    lg['common_baseline'] = 'W256'
    cfgs['local3grid'] = lg
    # cluster smokes (A1.0): 65^3, two cases, both modes, every phase, R' = 1024 included (memory and compile gate)
    cfgs['smk3grid'] = c3('smk3grid', 'grid', [dict(SETTINGS3[4], rule='lat32768'), dict(SETTINGS3[2], rule='lat16384')],
                          case_subset=[0, 1], timing=dict(cases=2, reps=1, burn=0.1, seed=20261013, common_cases=2,
                                                          common_reps=1, common_seed=20261013))
    cfgs['smk3esc'] = c3('smk3esc', 'esc', [dict(esc_settings()[4], ladder=['lat32768', 'gl48', 'lat256'])],
                         case_subset=[0, 1], timing=dict(cases=2, reps=1, burn=0.1, seed=20261013))
    # A11 (POST HOC): W1024 escalation extension (Gauss 80^3 -> 96^3), all 64 cases. A11.3: Gauss 112^3 is infeasible
    # at R' = 1024 on a 141 GB H200 with the vendored (unchunked) table builder (smk3ext 5215974: OOM allocating
    # 42.9 GiB), so the extension stops at 96^3; its smoke smk3ext2 runs both rules on two cases
    cfgs['c3extW1024'] = c3('c3extW1024', 'esc', [dict(SETTINGS3[4], ext_rules=['gl80', 'gl96'])],
                            timing=dict(cases=0, reps=0, burn=0.1, seed=20261013))
    cfgs['smk3ext2'] = c3('smk3ext2', 'esc', [dict(SETTINGS3[4], ext_rules=['gl80', 'gl96'])], case_subset=[0, 1],
                          timing=dict(cases=0, reps=0, burn=0.1, seed=20261013))
    cfgs['local3ext'] = c3('local3ext', 'esc', [dict(SETTINGS3[1], ext_rules=['gl24', 'gl32', 'gl40'])], case_subset=[0],
                           refined_ref=str(loc3), timing=dict(cases=0, reps=0, burn=0.1, seed=20261013))
    for k_ in ('c3extW1024', 'smk3ext2', 'local3ext'):
        cfgs[k_].pop('grid', None)
    # A14 (pre-registered conditional fallback qualification, A8.6): the R256 selected arms fail the finalist quadrature
    # check at gauss96; their first fallback gauss128 is qualified in c2fbgrid (accuracy + same-job timing against the
    # deployed R256 LSPG-BE at gauss96) and c2fbbud (quadrature, solver, budget). No FOM in the fallback grid.
    FB = [dict(name='R256', Rp=256, rule='gauss96'), dict(name='R256fb', Rp=256, rule='gauss128')]
    nofom = dict(schemes=[], dt_factors=[], ntols=[], order_cases=[])
    fbgrid = dict(forms_schemes=[['LSPG', 'BE'], ['LSPG', 'CNR']], dt_factors=[1, 2], anchor_factors=[1 / 8, 1 / 16])
    cfgs['c2fbgrid'] = c2('c2fbgrid', settings=FB, fom=nofom, common_baseline='R256')
    cfgs['c2fbgrid']['grid'] = fbgrid
    cfgs['local2dfb'] = c2('local2dfb', mesh=256, settings=FB, case_subset={'dev6': [0], 'val32': []}, reps=1, fom=nofom,
                           common_baseline='R256',
                           refs=str((HERE / '../../../2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/'
                                     'archive/output').resolve()), local_smoke_waives_cohort_hash=True)
    cfgs['local2dfb']['grid'] = dict(fbgrid, anchor_factors=[1 / 8])
    cfgs['local2dfb']['timing']['cases'] = 1
    # budget smokes (A1.0 step 6 drivers; arms are machinery choices, not a selection)
    sarms = [dict(form='LSPG', scheme='BE', f=1, role='baseline'), dict(form='LSPG', scheme='BDF2', f=2, role='test-selected',
                                                                        selected=True),
             dict(form='GAL', scheme='CN', f=1, role='diagnostic')]
    b2 = c2('smk2bud', settings=[dict(SETTINGS2[5], arms=sarms), dict(SETTINGS2[0], arms=sarms)],
            case_subset={'dev6': [0, 2], 'val32': [0]})
    b2.update(driver='c2bud', conv_rule='gauss320', check_rule='gauss640', successor_check='gauss768', control_rule='gauss8',
              conv_bar=2.5e-5, check_val32=1)
    for k_ in ('grid', 'fom', 'timing', 'common_baseline'):
        b2.pop(k_, None)
    cfgs['smk2bud'] = b2
    a3 = [dict(spec=['fs', 'BE', 0.01, 1], role='baseline'), dict(spec=['fs', 'CN', 0.025, 2], role='test-selected', selected=True),
          dict(spec=['fs', 'CN', 0.01, 1], role='diagnostic')]
    b3 = c3('smk3bud', 'bud', [dict(SETTINGS3[4], rule='lat32768', conv='gl56', finer='gl64', arms=a3)], case_subset=[0, 1, 2],
            timing=dict(cases=0, reps=0, burn=0.1, seed=20261013))
    b3.update(budget_cases=2, control_rule='lat256')
    b3.pop('grid', None)
    cfgs['smk3bud'] = b3
    for k, c in cfgs.items():
        if k.startswith(('smk', 'local')):
            c['smoke'] = True                     # A5.5: explicit smoke classification
        if 'timing_cases_override' in c:
            c['timing']['cases'] = c.pop('timing_cases_override')
        if k == 'local2d':
            c['timing']['cases'] = 1
        (HERE / 'configs').mkdir(exist_ok=True)
        (HERE / 'configs' / f'{k}.json').write_text(json.dumps(c, indent=1) + '\n')
        print(k)


if __name__ == '__main__':
    main()
