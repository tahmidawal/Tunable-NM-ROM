"""jcp-speed3d: write the run configs (DESIGN §2, §9, A1.9-A1.12). Production grid / fom configs are fixed by the design;
prof / mesh / rep configs are written by select.py from the frozen analysis.

    python make_configs.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETTINGS = {
    'M2_512': dict(name='M2_512', bank='M2', Rp=512, a=3, rule='gl32', reduced=['gl24', 'lat16384']),
    'W256': dict(name='W256', bank='W1024', Rp=256, a=6, rule='gl24', reduced=['gl20', 'gl16', 'lat8192']),
    'W512': dict(name='W512', bank='W1024', Rp=512, a=6, rule='lat32768', reduced=['lat16384', 'gl24']),
    'W768': dict(name='W768', bank='W1024', Rp=768, a=6, rule='lat32768', reduced=['lat16384']),
    'W1024': dict(name='W1024', bank='W1024', Rp=1024, a=6, rule='lat65536', reduced=['lat32768']),
}
BASE = dict(driver='s3run', cohort_seed=923801, cohort_count=64, trust_fraction=0.05, dts=[0.01, 0.0125, 0.05 / 3, 0.025])
TIMING = dict(cases=4, reps=2, burn=0.1, seed=20261014)
FOM = dict(fd=[['UB', 64, [0.01, 0.005, 0.0025]], ['UC', 64, [0.025, 0.0125, 0.01, 0.005]],
               ['CC', 64, [0.025, 0.0125, 0.01, 0.005]], ['UB', 128, [0.01, 0.005]], ['UC', 128, [0.0125, 0.01]],
               ['CC', 128, [0.0125, 0.01]]],
           ps=dict(Ns=[16, 24, 32, 48, 64], dts=[0.025, 0.01, 0.005]), calib_cases=[0, 1, 2, 3], save_cases=[0, 1])


def w(name, cfg):
    cfg = dict(attempt=name, **cfg)
    (HERE / 'configs' / f'{name}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print(name)


def main():
    w('g1', dict(BASE, mode='grid', settings=[SETTINGS['M2_512'], SETTINGS['W256']], timing=TIMING))
    w('g2', dict(BASE, mode='grid', settings=[SETTINGS['W512'], SETTINGS['W768']], timing=TIMING))
    w('g3', dict(BASE, mode='grid', settings=[SETTINGS['W1024']], timing=TIMING))
    w('f1', dict(BASE, mode='fom', settings=[], timing=TIMING, fom=FOM))
    # cluster smoke (real size, cases 0, 5, 7; every mode; DESIGN A1.10, A1.12)
    sm = dict(BASE, smoke=True, case_subset=[0, 5, 7], timing=dict(cases=2, reps=1, burn=0.05, seed=1))
    w('smkgrid', dict(sm, mode='grid', settings=[dict(SETTINGS['M2_512'], reduced=['gl24']), dict(SETTINGS['W1024'], reduced=[])],
                      dts=[0.01, 0.025]))
    w('smkfom', dict(sm, mode='fom', settings=[], fom=dict(fd=[['UB', 64, [0.01]], ['CC', 64, [0.0125]], ['UB', 128, [0.01]]],
                                                            ps=dict(Ns=[32], dts=[0.01]), calib_cases=[0, 5], save_cases=[0])))
    w('smkprof', dict(sm, mode='prof', settings=[SETTINGS['M2_512'], SETTINGS['W1024']], extra_rules=['gl12', 'gl56'],
                      prof=dict(configs=[dict(label='C_old', setting='M2_512', rule='gl32', spec='fs|BE|0.01|K1'),
                                         dict(label='qnCN', setting='M2_512', rule='gl32', spec='qn|CN|0.025|k4'),
                                         dict(label='W1024BE', setting='W1024', rule='lat65536', spec='fs|BE|0.01|K1'),
                                         dict(label='W1024qn', setting='W1024', rule='lat65536', spec='qn|CN|0.025|k4')],
                                mf=['qnCN', 'W1024qn'], k5_rule='gl12', profile=['C_old', 'qnCN', 'W1024BE'],
                                batch=dict(labels=['qnCN', 'C_old'], sizes=[1, 3]), confirm=[['C_old', 'qnCN']],
                                confirm_cases=2, confirm_reps=1, conv={'qnCN': 'gl56'}, reduced_adt=['qnCN'])))
    w('smkmesh', dict(sm, mode='mesh', settings=[SETTINGS['M2_512'], SETTINGS['W1024']],
                      mesh=dict(Ls=[64, 128, 384], roms=[dict(label='qnCN', setting='M2_512', rule='gl32', spec='qn|CN|0.025|k4'),
                                                         dict(label='W1024qn', setting='W1024', rule='lat65536', spec='qn|CN|0.025|k4')],
                                same=[['UB', 0.01], ['CC', 0.01]], matched=[['qnCN', 'ps|N32|0.01']],
                                calib_small=[0], calib_large=[0], acc_cases=2)))


if __name__ == '__main__':
    main()
