"""imp-fronts train jobs: per-phase timing record of one run_seed.sh output directory (smoke or production).

    python phase_times.py <outdir> <job> <arm> <mode> <wall_limit_s> <self_timer_s> <out.json>

Reads only files run_seed.sh / smoothtrain / sep_coeff_extract / sep_hfit_run / make_rotation write (missing files ->
null, so a partial run still gets a record). Phases:
  A.data            Stage A data generation (576 seed-0 trajectories, train.json data.seconds; excludes the R0 gate)
  A.r0_gate         the arm's R0 replacement gate (fid: PS C-B3 + self-convergence seconds; cov: C-B4, ~0)
  A.train_sub/full  per-step rates of the point-subsampled and the final full-spatial phase (stageA.log stamps;
                    the first full-phase interval includes the full step's compilation)
  B.gen_seed0 / B.gen_seed1000   extraction generate+project time of the 576 canonical and the 4,032 appended
                    trajectories (extract.log stamps at 576 and 4,608 trajectories)
  B.extract_total, B.hfit (per arm steps, seconds), rotation, stage wall seconds (seconds.txt)
plus a production projection at the recipe sizes (Stage A 290,000 sub + 10,000 full steps; head fit as measured
per step x 200,000 steps capped at TIME_CAP = 1,500 s per arm; data and extraction as measured, they are full size).
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


def main():
    o, job, arm, mode, wall, timer, outp = sys.argv[1:8]
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
    proj = {}
    if rates and rec['A']['data_seconds'] is not None:
        proj['stageA_s'] = (rec['A']['data_seconds'] + (rec['A']['r0_gate_seconds'] or 0) + rates['first_step_s']
                            + 290000 * rates['sub_s_per_step'] + 10000 * rates['full_s_per_step'])
    if ex.get('extract_total_s') is not None and ex['hfit']:
        hf = sum(min(1500.0, v['seconds'] / max(v['steps_done'], 1) * 200000) for v in ex['hfit'].values())
        proj['stageB_s'] = ex['extract_total_s'] + hf + max(0.0, (ex.get('hfit_total_s') or 0)
                                                           - sum(v['seconds'] for v in ex['hfit'].values()))
        proj['note_hfit'] = 'per arm: min(TIME_CAP 1500 s, measured s/step x 200,000); oracle/diag overheads as measured (smoke sizes, a lower bound)'
    if rec['rotation']['seconds'] is not None:
        proj['rotation_s'] = rec['rotation']['seconds']
    if {'stageA_s', 'stageB_s', 'rotation_s'} <= set(proj):
        proj['total_h'] = (proj['stageA_s'] + proj['stageB_s'] + proj['rotation_s']) / 3600
        proj['total_h_x1.25'] = 1.25 * proj['total_h']
        proj['ceiling_h'] = 4.4
        proj['self_timer_h'] = 0.9 * 4.4
    rec['production_projection'] = proj
    rec['exhausted'] = (o.parent / 'EXHAUSTED.json').exists()
    Path(outp).write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps(dict(stage_seconds=rec['stage_seconds'], A_rates=rates, B={k: ex.get(k) for k in
                          ('gen_seed0_s', 'gen_seed1000_s', 'extract_total_s')}, projection=proj), default=float))


if __name__ == '__main__':
    main()
