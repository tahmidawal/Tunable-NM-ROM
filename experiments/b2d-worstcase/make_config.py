"""Configs for wc.py (development cohorts only). python make_config.py -> config-*.json"""
import json

LADDER = [32, 48, 64, 96, 128, 192, 256, 384, 512]
VAL = dict(name='validation = P[4352:4608] = params_draw(1000,4032)[3776:4032]', draws=[[1000, 4032]],
           slice=[3776, 4032], disjoint_from=[[20260916, 64, 'test64'], [20261114, 64, 'fresh64'], [0, 576, 'bank train']])
BASE = dict(intervals=256, dt=.005, ladder=LADDER, reference=dict(name='fft_tight', dt=.005, ntol=1e-6, ltol=1e-8),
            rules=dict(lat64=dict(lattice=64), lat128=dict(lattice=128), lat32=dict(lattice=32)),
            cold_axis_points=48, step_budget=600, inner_damping=1e-10, time_reps=3, time_cases=16)


def arm(Rp, M=None, rule='lat64', gtol=1e-2, **kw):
    return dict(Rp=Rp, M=M or 4 * Rp, rule=rule, gtol=gtol, graphs=True, **kw)


def write(name, **kw):
    json.dump(dict(BASE, attempt=name, **kw), open(f'config-{name}.json', 'w'), indent=1)


# Step 1: diagnosis of the deployed checkpoint on validation
arms = [arm(R) for R in (192, 256, 384, 512)]
for R in (384, 512):
    arms += [arm(R, ic='grid'), arm(R, gtol=1e-4), arm(R, gtol=1e-6), arm(R, rule='lat128'),
             arm(R, M=2 * R), arm(R, M=8 * R), arm(R, rule='lat128', gtol=1e-6, ic='grid'),
             arm(R, exact_steps=50, cases=64), arm(R, exact_steps=50, gtol=1e-6, ic='grid', cases=64),
             arm(R, M=8 * R, exact_steps=50, gtol=1e-6, ic='grid', cases=64)]
arms += [arm(256, gtol=1e-6, ic='grid'), arm(192, gtol=1e-6, ic='grid')]
write('diag', cohort=VAL, projection_widths=[128, 192, 256, 384, 512], arms=arms)
write('smoke', cohort=dict(VAL, take=2), projection_widths=[32, 64], allow_cpu_smoke=True, intervals=64,
      arms=[dict(arm(32), graphs=False), dict(arm(64, ic='grid', gtol=1e-6, exact_steps=50), graphs=False)], time_cases=1)

# Step 2: validation of candidate banks (native width R; nested rungs below), next to the deployed setting in the same job
write('ref', cohort=VAL, projection_widths=[384], arms=[arm(384), arm(384, M=8 * 384)])
for R in (256, 384, 512):
    sub = [r for r in (192, 256, 384, 512) if r <= R][-3:]
    write(f'val{R}', cohort=VAL, projection_widths=sub,
          arms=[arm(r, M=m * r) for r in reversed(sub) for m in (4, 8)])

# fast solve of the query-speed lane (its frozen.json 'accurate' settings at widths 384 and 512, and its 'fast' one)
QS_SET = dict(qsk1f10=dict(k=1, s=50, form='fused', stencil32=True, kfirst=2, nfirst=10),
              qsk3=dict(k=3, s=50, form='fused', stencil32=True),
              qsk1P=dict(k=1, s=50, form='P', stencil32=True))

# native-width validation (the fine-tune is width-native; lower rungs are not candidates)
for R in (384, 448, 512):
    write(f'nat{R}', cohort=VAL, projection_widths=[R], ladder=sorted(set(LADDER[:8] + [R])),
          arms=[arm(R, M=m * R) for m in (4, 6, 8)]
          + [arm(R, M=m * R, qs=q, tag=t) for m in (4, 8) for t, q in QS_SET.items()]
          # amendment A2: more tests need the finer quadrature lattice (lat64 has 63^2 = 3969 nodes)
          + [arm(R, M=m * R, rule='lat128', qs=QS_SET[t], tag=t) for m in (8, 12, 16) for t in ('qsk3', 'qsk1P')])

# Final evaluation (run once, after selection.json is committed): arms are written by freeze.py
TEST = dict(name='test64 = params_draw(20260916, 64)', draws=[[20260916, 64]],
            expected_sha256='cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c')
FRESH = dict(name='fresh64 = params_draw(20261114, 64)', draws=[[20261114, 64]],
             disjoint_from=[[20260916, 64, 'test64'], [0, 576, 'pool a'], [1000, 4032, 'pool b']],
             expected_sha256='1116b7080643b2b2ea96662f6bd3592d5c51b963b3e2e3c45fc5fb22711db2f8')
FINAL = dict(BASE, time_reps=5, time_cases=64)
