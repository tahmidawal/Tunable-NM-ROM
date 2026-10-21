"""imp-fronts train jobs: timing record and the training projection (DESIGN Amendment 11).

    smoke:       python phase_times.py <datagen.json> <job> <arm> smoke <wall_s> <timer_s> <out.json>
    production:  python phase_times.py --production <run_seed outdir> <job> <arm> production <wall_s> <timer_s> <out.json>

Smoke: measured data-generation seconds per population (smoke_datagen.py: seed 0 x 576, seed 1000 x 4032) and the
training projection of Amendment 11:
    projection = K1 measured maximum production time + measured data-generation increment over K1's data generation
  K1 maximum = jcp-robust e2t2d seed 2 (job e2t2d, A100 80GB PCIe), run_record.json stage_seconds A 11099 + B 3901 +
  rotation 18 = 15018 s = 4.172 GPU-h (runs/e2t2d/archive/output/train2d_s2/run_record.json, sha256 d5259149...).
  K1 data generation in that run: Stage A data.seconds 124.96 s (576 trajectories: rollout + residual check + host
  copy, the work smoke_datagen.py times) and Stage B extraction gen+proj 906 s at 4,608/4,608 (extract.log; it also
  contains the projection, so using it as K1's generation time makes the increment conservative, i.e. larger).
  The arm's production generation = Stage A seed0 + Stage B (seed0 + seed1000); increment = (arm_A - 124.96) +
  (arm_B - 906), floored at 0 in the projection (a negative increment is reported but not credited).
  Flag NOT FUNDABLE if the projection > 4.6 GPU-h (Amendment 11 ceiling); UNCERTIFIED if the generation was incomplete.
Production: the per-phase record of a run_seed.sh output directory (stage seconds, Stage A rates, extraction and head
fit seconds), for the ledger; no projection.
"""
import json
import re
import sys
from pathlib import Path


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def num(p):
    try:
        return float(Path(p).read_text().strip())
    except Exception:
        return None


def production(argv):
    o, job, arm, mode, wall, timer, outp = argv
    o = Path(o)
    A = jload(o / 'stageA/train.json') or {}
    rec = dict(job=job, arm=arm, mode=mode, wall_limit_s=float(wall), self_timer_s=float(timer),
               device=(o / 'device.txt').read_text().splitlines() if (o / 'device.txt').exists() else None,
               stage_seconds={k: num(o / k / 'seconds.txt') for k in ('stageA', 'stageB', 'rotation')})
    d = A.get('data', {})
    g = d.get('R0_replacement') or {}
    gs = g.get('seconds')
    gs = sum(gs.values()) if isinstance(gs, dict) else gs
    rec['A'] = dict(gpu=A.get('gpu'), backend=A.get('backend'), precision=A.get('precision'),
                    generator_sha256=A.get('generator_sha256'), data_seconds=d.get('seconds'),
                    max_fom_rel_residual=d.get('max_fom_rel_residual'), r0_gate_passed=g.get('passed'),
                    r0_gate_seconds=gs, r0_gate_summary=g.get('summary'),
                    train_seconds=(A.get('train') or {}).get('seconds'), steps=A.get('steps'))
    stamps = []
    if (o / 'stageA/stageA.log').exists():
        for m in re.finditer(r'train2\[\w+\] step\s+(\d+)/(\d+).*?\[(\d+)s\]', (o / 'stageA/stageA.log').read_text()):
            stamps.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    rates = {}
    if stamps:
        steps = stamps[-1][1]
        full_last = min(int(A.get('recipe', {}).get('full_last', 10000)) if mode == 'production' else 500, steps)
        b = steps - full_last
        t = {s: x for s, _, x in stamps}
        if b in t and steps in t and b > 1 and 1 in t:
            rates = dict(sub_s_per_step=(t[b] - t[1]) / (b - 1), full_s_per_step=(t[steps] - t[b]) / full_last,
                         first_step_s=t[1], full_last=full_last, boundary_step=b)
    rec['A']['rates'] = rates
    ex = {}
    if (o / 'stageB/extract.log').exists():
        L = (o / 'stageB/extract.log').read_text()
        st = {int(m.group(1)): float(m.group(3)) for m in
              re.finditer(r'gen\+proj: trajectories (\d+)/(\d+).*?\[(\d+)s\]', L)}
        tot = re.search(r'EXTRACT done in (\d+)s', L)
        ex = dict(gen_seed0_s=st.get(576), gen_seed1000_s=(st[4608] - st[576]) if 4608 in st and 576 in st else None,
                  gen_all_s=st.get(max(st)) if st else None, n_traj_logged=max(st) if st else None,
                  extract_total_s=float(tot.group(1)) if tot else None)
    h = jload(o / 'stageB/hfit_full.json') or {}
    ex['hfit'] = {k: {kk: v['train'].get(kk) for kk in ('steps', 'steps_done', 'seconds')}
                  for k, v in (h.get('arms') or {}).items()}
    ex['hfit_total_s'] = h.get('total_seconds')
    rec['B'] = ex
    rec['rotation'] = dict(seconds=num(o / 'rotation/seconds.txt'))
    rec['exhausted'] = (o.parent / 'EXHAUSTED.json').exists()
    Path(outp).write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps(dict(stage_seconds=rec['stage_seconds'], A_rates=rates, B={k: ex.get(k) for k in
                          ('gen_seed0_s', 'gen_seed1000_s', 'extract_total_s')}), default=float))


K1 = dict(record='jcp-robust runs/e2t2d/archive/output/train2d_s2/run_record.json', record_sha256=
          'd525914941979abde3293e11b6df9e53e25d0efc5276364cfb23c56e239bf3bc', job='e2t2d', seed=2,
          gpu='NVIDIA A100 80GB PCIe', stage_seconds=dict(stageA=11099, stageB=3901, rotation=18),
          total_s=15018, stageA_data_s=124.96112990379333, stageB_genproj_4608_s=906.0)
CEILING_H = 4.6


def smoke(argv):
    dg, job, arm, mode, wall, timer, outp = argv
    d = json.loads(Path(dg).read_text())
    P = d['populations']
    s0, s1 = P['seed0'], P['seed1000']
    rec = dict(job=job, arm=arm, mode=mode, wall_limit_s=float(wall), self_timer_s=float(timer),
               backend=d.get('backend'), device=d.get('device'), datagen_passed=d.get('passed'),
               datagen_seconds={k: dict(m=v['m'], done=v['done'], seconds=v['seconds'], s_per_traj=v['s_per_traj'],
                                         finite=v['finite'], independent_residual=v['independent_residual'],
                                         rollout_rel_res=v['rollout_rel_res']) for k, v in P.items()},
               gate=d.get('C_B3') if arm == 'fid' else d.get('C_B4'), K1_reference=K1)
    complete = bool(d.get('complete'))
    arm_A = s0['seconds']
    arm_B = s0['seconds'] + s1['seconds']
    inc = (arm_A - K1['stageA_data_s']) + (arm_B - K1['stageB_genproj_4608_s'])
    proj_s = K1['total_s'] + max(0.0, inc)
    proj_h = proj_s / 3600
    status = 'UNCERTIFIED (generation incomplete)' if not complete else (
        'NOT FUNDABLE' if proj_h > CEILING_H else 'fundable')
    rec['projection'] = dict(arm_stageA_gen_s=arm_A, arm_stageB_gen_s=arm_B, increment_s=inc,
                             projection_s=proj_s, projection_gpu_h=proj_h, ceiling_gpu_h=CEILING_H, status=status,
                             same_gpu_type_as_K1=(d.get('device') == 'NVIDIA A100 80GB PCIe'))
    if not rec['projection']['same_gpu_type_as_K1']:
        rec['projection']['note'] = 'smoke GPU type differs from K1 record: the increment mixes GPU types'
    Path(outp).write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps(dict(datagen={k: (v['done'], round(v['seconds'], 1)) for k, v in P.items()},
                          projection=rec['projection']), default=float))


if __name__ == '__main__':
    if sys.argv[1] == '--production':
        production(sys.argv[2:9])
    else:
        smoke(sys.argv[1:8])
