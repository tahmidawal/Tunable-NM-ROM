"""imp-fronts train jobs: the training timing record (DESIGN Amendments 11, 15) in the schema resources.train_item()
reads, and the production per-phase record.

    smoke:       python phase_times.py smoke <job output dir> <SM-a|SM-b> <fid|cov> <slurm job id>
    SM-c:        python phase_times.py smokec <job output dir> <slurm job id>     (arm (c); see smokec())
    production:  python phase_times.py --production <run_seed outdir> <job> <arm> production <wall_s> <stop_s> <out.json>

Smoke inputs (all in <job output dir>): datagen.json (smoke_datagen.py), chain.json (TRAIN2D_DRYRUN chaining slice),
cb1.json (C-B1 guard rejection). Output: <job output dir>/timing.json, schema "imp-fronts-train-timing/3" (resources.py
docstring; resources.train_item schema imp-fronts-train-timing/3), written ONLY if the record is valid; otherwise timing_INVALID.json with the reasons, and exit 1.
  increment.{arm,incumbent}_datagen_seconds = seed0 seconds x 1152/576 (Stage A and Stage B each generate seed 0; Amendment 16) + seed1000 chunk seconds x 4032/576, both
      generators timed in the same process after a burn-in (Amendment 15: scaled to 576 + 4 032 trajectories);
  phases.datagen = the arm's two measured segments (units 576 -> production 1152 and 4032);
  phases.stageA / stageB / rotation = K1's measured production stage seconds (jcp-robust e2t2d seed 2, the 4.17 h
      record; Amendment 15: training-phase rates come from K1's records, identical training code), one segment each
      (units = production_units = 1). These are diagnostic only (resources' per-phase extrapolation; note that K1's
      stage seconds already contain K1's own data generation).
  projection = (15018 s + max(0, arm - incumbent)) / 3600 (resources.train_item), certified against the 4.567 h cutoff (4.6 h - 120 s).
Validity (all required): datagen passed (gates + complete paired timing + residuals), backend gpu, device kind equal
to K1's GPU type (NVIDIA A100 80GB PCIe), datagen job_id equal to this job's SLURM_JOB_ID, chaining slice complete
(SEED-COMPLETE), C-B1 fired.
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
          gpu='NVIDIA A100 80GB PCIe', stage_seconds=dict(stageA=11099, stageB=3901, rotation=18), total_s=15018)
CUTOFF_H = 4.6 - 120 / 3600.                                 # 4.567: run_seed.sh is stopped 120 s before the wall
SCALE_0 = 1152 / 576                                      # seed 0 is generated in Stage A AND Stage B (Amendment 16)
SCALE_1000 = 4032 / 576


def smoke(argv):
    od, job, arm, slurm = Path(argv[0]), argv[1], argv[2], argv[3]
    errs = []
    d = jload(od / 'datagen.json') or {}
    ch = jload(od / 'chain.json') or {}
    cb = jload(od / 'cb1.json') or {}
    T = d.get('timed') or {}
    need = ('incumbent/seed0', 'arm/seed0', 'incumbent/seed1000', 'arm/seed1000')
    if not d:
        errs.append('datagen.json missing')
    if d and d.get('passed') is not True:
        errs.append('datagen not passed (gate, completeness or residuals)')
    if any(k not in T or not T[k].get('complete') or not T[k].get('seconds', 0) > 0 or T[k].get('units') != 576
           for k in need):
        errs.append('paired timing incomplete')
    if d.get('backend') != 'gpu':
        errs.append(f"backend {d.get('backend')!r} is not gpu")
    if d.get('device') != K1['gpu']:
        errs.append(f"device {d.get('device')!r} differs from K1's GPU type {K1['gpu']!r}")
    if not slurm or d.get('job_id') != slurm:
        errs.append(f"datagen job_id {d.get('job_id')!r} != this job {slurm!r}")
    if ch.get('seed_complete') is not True or ch.get('chain_rc') != 0:
        errs.append(f'chaining slice not complete ({ch})')
    if cb.get('C_B1_guard_rejects') is not True:
        errs.append('C-B1 guard rejection not recorded')
    rec = dict(schema='imp-fronts-train-timing/3', job=job, job_id=slurm, backend=d.get('backend'), device=d.get('device'),
               gpu=d.get('device'), arm={'fid': 'a', 'cov': 'b'}[arm], errors=errs, K1_reference=K1,
               gates_passed=bool((d.get('gate') or {}).get('passed') is True),
               chaining_completed=bool(ch.get('seed_complete') is True and ch.get('chain_rc') == 0),
               burn_in=bool(isinstance(d.get('burn_in_seconds'), (int, float)) and d['burn_in_seconds'] > 0),
               cb1_fired=bool(cb.get('C_B1_guard_rejects') is True), datagen_record_sha256=None)
    if not rec['burn_in']:
        errs.append('burn-in not recorded')
    if not rec['gates_passed']:
        errs.append('arm gate not passed')
    if (od / 'datagen.json').exists():
        import hashlib
        rec['datagen_record_sha256'] = hashlib.sha256((od / 'datagen.json').read_bytes()).hexdigest()
    if not errs:
        sec = {k: float(T[k]['seconds']) for k in need}
        arm_s = sec['arm/seed0'] * SCALE_0 + sec['arm/seed1000'] * SCALE_1000
        inc_s = sec['incumbent/seed0'] * SCALE_0 + sec['incumbent/seed1000'] * SCALE_1000
        rec['phases'] = dict(
            datagen=dict(fixed_seconds=0., source='this smoke (arm generator, paired with the incumbent)', segments=[
                dict(name='arm/seed0', seconds=sec['arm/seed0'], units=576, production_units=1152),
                dict(name='arm/seed1000', seconds=sec['arm/seed1000'], units=576, production_units=4032)]),
            **{st: dict(fixed_seconds=0., source=f"K1 record {K1['record']} (Amendment 15)",
                        segments=[dict(name=f'K1/{st}', seconds=float(K1['stage_seconds'][st]), units=1,
                                       production_units=1)]) for st in ('stageA', 'stageB', 'rotation')})
        rec['timing'] = dict(populations=dict(
            seed0=dict(trajectories=576, production_trajectories=1152, incumbent_s=sec['incumbent/seed0'],
                       arm_s=sec['arm/seed0']),
            seed1000=dict(trajectories=576, production_trajectories=4032, incumbent_s=sec['incumbent/seed1000'],
                          arm_s=sec['arm/seed1000'])))
        rec['all_done'] = True
        rec['increment'] = dict(arm_datagen_seconds=arm_s, incumbent_datagen_seconds=inc_s, measured=sec,
                                chunk_seconds={k: T[k].get('chunk_seconds') for k in need},
                                scale_seed0=SCALE_0, scale_seed1000=SCALE_1000, burn_in_seconds=d.get('burn_in_seconds'))
        proj = (K1['total_s'] + max(0., arm_s - inc_s)) / 3600.
        rec['projection'] = dict(projection_gpu_h=proj, increment_gpu_h=(arm_s - inc_s) / 3600., cutoff_gpu_h=CUTOFF_H,
                                 status='fundable' if proj <= CUTOFF_H else 'NOT FUNDABLE')
        (od / 'timing.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
        print(json.dumps(dict(increment_s=arm_s - inc_s, projection=rec['projection']), default=float))
        return 0
    (od / 'timing_INVALID.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print('TIMING INVALID: ' + '; '.join(errs))
    return 1

# ------------------------------------------------------------------ SM-c (arm (c) 'sine_even'), DESIGN Amendments 19-21
K1C = dict(record='jcp-robust runs/e2t2d/archive/output/train2d_s1/run_record.json',
           record_sha256='7dfb5856c81c02327d0afa5a0909916e657043432936be62e1284f40eab9cc74',
           stageA_log_sha256='4d0addc31169699dd97182798e4abe3bde977a83c4679897328163fe45502160',
           extract_log_sha256='5546517ff72b22fc79b53f61eb5ca671ee57d0e164cf861255c2389eb4c1ff40',
           gpu='NVIDIA A100 80GB PCIe', stageA_s=9766., stageB_s=3501., rotation_s=19., data_A_s=119.68222498893738,
           # stageA.log: step 1 at 7 s, step 290,000 at 6,884 s, step 300,000 at 9,612 s
           t_sub=(6884. - 7.) / 289999., t_full=(9612. - 6884.) / 10000., extract_s=876.)
N_SUB, N_FULL, CUTOFF_C_S = 290000, 10000, 16440.


def smokec(argv):
    """T = n_sub t_sub rho_sub + n_full t_full rho_full + T_B,gen+head + T_B,extract max(1, rho_ext) + T_rot
         + data_A + remA max(1, rho_full) + T_compile                                     (Amendment 21; certified iff T <= 16,440 s)
    K1 (seed 1) split: T_B,extract = 876 s (extract.log total; generation is interleaved with projection there and cannot be
    separated, so the whole extraction is scaled, never credited below 1); T_B,gen+head = 3,501 - 876 s unscaled;
    remA = Stage A total - data - n_sub t_sub - n_full t_full (first step, recon passes, LS rotation, save: full-grid
    feature evaluations, scaled by max(1, rho_full)); T_compile = the arm's measured step_sub + step_full compilation.
    certified iff projection_conservative_s = T + B_extract (extraction counted twice, audit codex-job-smC-2) <= 16,440 s."""
    od, slurm = Path(argv[0]), argv[1]
    errs = []
    t = jload(od / 'smc_timing.json') or {}
    idn = jload(od / 'identity.json') or {}
    ch = jload(od / 'chain.json') or {}
    cb = jload(od / 'cb1.json') or {}
    if t.get('passed') is not True:
        errs.append('SM-c timing not passed')
    if t.get('small'):
        errs.append('timing ran in --small mode')
    if t.get('backend') != 'gpu' or t.get('device') != K1C['gpu']:
        errs.append(f"timing device {t.get('device')!r}/{t.get('backend')!r} is not K1's GPU type")
    if not slurm or t.get('job_id') != slurm:
        errs.append(f"timing job_id {t.get('job_id')!r} != this job {slurm!r}")
    if idn.get('fails') != [] or idn.get('backend') != 'gpu':
        errs.append(f"identity tests not passed on GPU ({idn.get('fails')})")
    if ch.get('seed_complete') is not True or ch.get('chain_rc') != 0:
        errs.append(f'chaining slice not complete ({ch})')
    if cb.get('C_B1_guard_rejects') is not True:
        errs.append('C-B1 guard rejection not recorded')
    rec = dict(schema='imp-fronts-train-timing-c/1', job='SM-c', job_id=slurm, arm='c', device=t.get('device'),
               K1_reference=K1C, errors=errs, identity_stats=idn.get('stats'), bank_control=idn.get('bank_control'))
    if not errs:
        tr, ex = t['train'], t['extract']
        rho_sub = tr['arm']['sub_s_per_step'] / tr['incumbent']['sub_s_per_step']
        rho_full = tr['arm']['full_s_per_step'] / tr['incumbent']['full_s_per_step']
        rho_ext = ex['arm']['median_s'] / ex['incumbent']['median_s']
        remA = K1C['stageA_s'] - K1C['data_A_s'] - N_SUB * K1C['t_sub'] - N_FULL * K1C['t_full']
        comp = tr['arm']['compile_sub_s'] + tr['arm']['compile_full_s']
        parts = dict(sub=N_SUB * K1C['t_sub'] * rho_sub, full=N_FULL * K1C['t_full'] * rho_full, data_A=K1C['data_A_s'],
                     remA=max(0., remA) * max(1., rho_full), B_gen_head=K1C['stageB_s'] - K1C['extract_s'],
                     B_extract=K1C['extract_s'] * max(1., rho_ext), rotation=K1C['rotation_s'], compile=comp)
        T = sum(parts.values())
        # audit codex-job-smC-2: the 514-uniform-call extraction weighting is approximate (production: 498 full + 71
        # partial + 8 test calls); registered conservative bound = the extraction term counted twice
        Tc = T + parts['B_extract']
        rec.update(rho=dict(sub=rho_sub, full=rho_full, extract=rho_ext), parts_s=parts, projection_s=T,
                   projection_conservative_s=Tc, projection_gpu_h=T / 3600., projection_conservative_gpu_h=Tc / 3600.,
                   cutoff_s=CUTOFF_C_S, certified=bool(Tc <= CUTOFF_C_S),
                   status='fundable' if Tc <= CUTOFF_C_S else 'NOT FUNDABLE', all_done=True)
        (od / 'timing_c.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
        print(json.dumps(dict(rho=rec['rho'], projection_s=T, projection_conservative_s=Tc, status=rec['status']), default=float))
        return 0
    (od / 'timing_c_INVALID.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print('SM-c TIMING INVALID: ' + '; '.join(errs))
    return 1


if __name__ == '__main__':
    if sys.argv[1] == '--production':
        production(sys.argv[2:9])
    elif sys.argv[1] == 'smokec':
        sys.exit(smokec(sys.argv[2:4]))
    elif sys.argv[1] == 'smoke':
        sys.exit(smoke(sys.argv[2:6]))
    else:
        sys.exit(f'usage: {sys.argv[0]} smoke|--production ...')
