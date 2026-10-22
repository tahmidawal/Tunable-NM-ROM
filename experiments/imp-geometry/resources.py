"""imp-geometry cost certificate (DESIGN A1.10, A3.8): project the production GPU-hours per cell from the measured
production-size probe (A100-80GB, runs/smoke/smoke{4,5}/probe*) and the registered ladder sizes, x 1.5 headroom.
Writes runs/cost/projection.json. Every unmeasured unit is named and given a stated conservative estimate.

  python resources.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE / 'runs' / 'smoke' / 'smoke5' / 'probe'
HEAD = 1.5


def load(name):
    return json.loads((P5 / name).read_text())


def measured(cell_dir, meas_json):
    """A8 re-certification from MEASURED numbers: the full cell's stage seconds and peak memories plus the isolated
    bank-step measurement; remaining cells scaled by stated factors (S2 / S1 from the production probes for ref and data;
    ladder / time / diag x 1.5 for S2 and box = 0.6 x ladder without SEM: assumptions, labelled)."""
    cell = Path(cell_dir)
    st = {}
    peaks = {}
    for line in (cell / 'stage_seconds.jsonl').read_text().splitlines():
        r_ = json.loads(line)
        st[r_['stage']] = st.get(r_['stage'], 0) + r_['seconds']
        peaks[r_['stage']] = (r_.get('device_peak_bytes', 0), r_.get('host_peak_bytes', 0))
    meas = json.loads(Path(meas_json).read_text()) if meas_json else {}
    bank_h_S1 = st.get('bank', 0) / 3600
    bank_h_S2 = meas.get('bank_train_projection_h', bank_h_S1 * 1.5)       # isolated H|S2 measurement (meas6)
    ref_f, data_f = 108.3 / 21.6, 34.81 / 7.83           # probe S2 / S1 ratios (smoke4/5 logs)
    h = lambda k: st.get(k, 0) / 3600
    L1 = sum(h(k) for k in st)
    S2 = h('ref') * ref_f + h('data') * data_f + bank_h_S2 + 1.5 * (h('ladder') + h('time') + h('diag') + h('inter'))
    box = lambda f: f * (h('ref') + h('data')) + 0.6 * 1.5 * (h('ladder') + h('time'))
    remaining = dict(L_wide_S2=S2, H_wide_S1=L1, H_wide_S2=S2, B_wide_S1=box(1.0), B_wide_S2=box(ref_f),
                     test_jobs=6 * 0.3, box_banks_conditional=bank_h_S1 + bank_h_S2)
    tot = sum(remaining.values())
    spent = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())['total_gpu_hours']
    data_ratio = (768 / 512) ** 2
    mem = {k: dict(device_GB=v[0] / 1e9, host_GB=v[1] / 1e9, S2_device_GB_bound=v[0] / 1e9 * data_ratio,
                   S2_host_GB_bound=v[1] / 1e9 * data_ratio) for k, v in peaks.items()}
    out = dict(measured_stage_hours={k: v / 3600 for k, v in st.items()}, first_cell_total_h=L1, bank_meas=meas,
               remaining_cells_h=remaining, remaining_total_h=tot, remaining_with_headroom_h=tot * HEAD, spent_h=spent,
               cap=62, fits=bool(spent + tot * HEAD <= 62), peak_memory=mem,
               memory_ok=all(v['S2_device_GB_bound'] <= 80 * 0.9 and v['S2_host_GB_bound'] <= 240 for v in mem.values()),
               assumptions=['S2 ladder/time/diag/inter = 1.5 x S1', 'H cells bounded by the L cell (fewer SEM rows)',
                            'box = probe-scaled ref/data + 0.6 x 1.5 x (ladder + time)', 'S2 memory = S1 x (768/512)^2'])
    (HERE / 'runs' / 'cost').mkdir(parents=True, exist_ok=True)
    (HERE / 'runs' / 'cost' / 'projection_measured.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()
                      if k in ('first_cell_total_h', 'remaining_total_h', 'remaining_with_headroom_h', 'spent_h', 'fits',
                               'memory_ok')}))


def measured_v2(cell_dir, meas_dir):
    """A9 re-certification from the timed-out first cell: measured per-family ladder walls (registered rows only,
    after the A5.2 reductions), measured ref / data / bank stages, isolated bank-step probes; listed assumptions for
    the unmeasured parts. Writes runs/cost/projection_A9.json."""
    import collections
    cell = Path(cell_dir)
    rows = [json.loads(l) for l in (cell / 'ladder.jsonl').read_text().splitlines()]
    R = (16, 32, 64, 128, 256, 384)

    def reg(c):
        if c.get('fam') in ('podg', 'ecsw', 'bank') and c['r'] not in R:
            return False
        return not (c.get('fam') == 'sem' and c.get('g') not in (None, 0.5))
    fam_h = collections.defaultdict(float)
    for x in rows:
        if reg(x['cfg']):
            fam_h[x['cfg']['fam']] += x.get('wall', 0) / 3600
    st = {}
    for line in (cell / 'stage_seconds.jsonl').read_text().splitlines():
        r_ = json.loads(line)
        st[r_['stage']] = r_['seconds'] / 3600
    pS2 = json.loads((Path(meas_dir) / 'probe_H_S2_bankstep.json').read_text())
    A = dict(bank_ladder_remaining_L_S1_h=0.8, bank_ladder_full_h=1.3, time_stage_S1_h=4.0, S2_ladder_factor=1.5,
             H_sem_fraction=0.5, box_time_h=1.5, spec_ladder_h=0.3, test_job_h=0.3, tail_h=0.3)
    lad_L = fam_h['fdmg'] + fam_h['rk4'] + fam_h['sem'] + fam_h['podg'] + fam_h['ecsw'] + A['bank_ladder_full_h']
    lad_H = lad_L - (1 - A['H_sem_fraction']) * fam_h['sem']
    ref_f, data_f = 108.3 / 21.6, 34.81 / 7.83
    s1_fix = st['ref'] + st['data'] + st['bank']
    s2_fix = st['ref'] * ref_f + st['data'] * data_f + pS2['bank_train_projection_h']
    f2 = A['S2_ladder_factor']
    cells = dict(
        L_wide_S1_resume=A['bank_ladder_remaining_L_S1_h'] + A['time_stage_S1_h'] + A['tail_h'],
        L_wide_S2=s2_fix + f2 * (lad_L + A['time_stage_S1_h']) + A['tail_h'],
        H_wide_S1=s1_fix + lad_H + A['time_stage_S1_h'] + A['tail_h'],
        H_wide_S2=s2_fix + f2 * (lad_H + A['time_stage_S1_h']) + A['tail_h'],
        B_wide_S1=st['ref'] + st['data'] + fam_h['fdmg'] + fam_h['rk4'] + A['spec_ladder_h'] + fam_h['podg'] + fam_h['ecsw'] + A['box_time_h'],
        B_wide_S2=st['ref'] * ref_f + st['data'] * data_f + f2 * (fam_h['fdmg'] + fam_h['rk4'] + A['spec_ladder_h'] + fam_h['podg'] + fam_h['ecsw'] + A['box_time_h']))
    tests = {k: A['test_job_h'] for k in cells}
    spent = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())['total_gpu_hours']
    remaining_cap = 62 - spent

    def plan(names):
        raw = sum(cells[n] + tests[n] for n in names)
        return dict(cells=names, raw_h=raw, with_headroom_h=raw * HEAD, fits=bool(raw * HEAD <= remaining_cap))
    plans = dict(full_wide=plan(list(cells)),
                 minus_H_S2=plan([n for n in cells if n != 'H_wide_S2']),
                 geometry_cells_only_minus_H_S2=plan(['L_wide_S1_resume', 'H_wide_S1', 'L_wide_S2']),
                 one_cell_per_geometry=plan(['L_wide_S1_resume', 'H_wide_S1']))
    out = dict(measured_family_ladder_h_L_S1=dict(fam_h), measured_stage_h=st, bank_probe_S2_h=pS2['bank_train_projection_h'],
               assumptions=A, cells_h=cells, spent_h=spent, remaining_cap_h=remaining_cap, headroom=HEAD, plans=plans)
    (HERE / 'runs' / 'cost').mkdir(parents=True, exist_ok=True)
    (HERE / 'runs' / 'cost' / 'projection_A9.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: dict(raw=round(v['raw_h'], 1), headroom=round(v['with_headroom_h'], 1), fits=v['fits'])
                      for k, v in plans.items()}), 'remaining cap', round(remaining_cap, 1))
    print({k: round(v, 1) for k, v in cells.items()})


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--measured-v2':
        return measured_v2(sys.argv[2], sys.argv[3])
    if len(sys.argv) > 1 and sys.argv[1] == '--measured':
        return measured(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    s1, fdp, semp, bk = (load(f) for f in ('probe_L_S1_ref.json', 'probe_L_S2_fd.json', 'probe_L_S2_sem.json',
                                            'probe_L_S2_bank.json'))
    # S2 reference/data/POD numbers from smoke4/5 logs (recorded in the slurm logs, transcribed by parse below)
    log4 = (HERE / 'runs' / 'smoke' / 'smoke4' / 'probe_L_S2.log').read_text()
    log5 = (HERE / 'runs' / 'smoke' / 'smoke5' / 'slurm-6158976.out').read_text()

    def grab(txt, key):
        for line in txt.splitlines():
            if line.startswith(key + ':'):
                return float(line.split(':')[1].strip().rstrip('s'))
        raise KeyError(key)
    s2 = {k: grab(log4, k) for k in ('ref_n256_8', 'ref_n512_8', 'ref_n1024_8', 'data_8traj')}
    pod = {k: grab(log5, k) for k in ('pod_4032', 'pod_space', 'podg_r384_lspg_cnr_32', 'podg_r384_gal_etdrk4_32',
                                      'podg_r384_gal_cnr_32')}
    n_ref_cases = 8 + 32 + 32
    units = {}
    for lev in ('S1', 'S2'):
        if lev == 'S1':
            ref8 = s1['ref_n128_8'] + s1['ref_n256_8'] + s1['ref_n512_8']
            fin8 = s1['ref_n512_8']
            data8 = s1['data_8traj']
        else:
            ref8 = s2['ref_n256_8'] + s2['ref_n512_8'] + s2['ref_n1024_8']
            fin8 = s2['ref_n1024_8']
            data8 = s2['data_8traj']
        ref = n_ref_cases / 8 * ref8 + 3 * fin8 + fin8           # cohorts + time check (n, 2n) + 1/128 lattice
        data = 192 / 8 * data8 + pod['pod_4032'] + pod['pod_space']
        inter = 3 * 4 * fin8 / 4                                 # three FD runs x 32 cases at the middle level
        # ladders (rows x measured per-row seconds; compile included in every measured row)
        fd_rows = 33.0 * 0.5 * 15 * 2.33 + 18 * 10.0 + 200.0     # A3.7 all sizes: sum n^3 scaling of the n768 probe
        sem_L = 90 * 30.0 + 30 * 60.0 + 30 * 2 * 6 * 5.0 + 30 * 4 * 10.0 + 30 * 2 * 10.0   # A6: all feasible configs
        podg = 10 * 4 * 5 * (pod['podg_r384_lspg_cnr_32'] + pod['podg_r384_gal_etdrk4_32']) / 2 * 0.7
        ecsw = 10 * 2 * 90.0 + 10 * 3 * 4 * 5 * 5.0               # NNLS fits (2 per r, estimate) + rows (4 steppers, A6)
        bank_rows = 10 * 7 * 5 * bk['bank_r384_lspg_32'] * 1.2 + 10 * bk['bank_space_q12']   # 7 rules x ~5 rungs (A6/A7)
        ladder = fd_rows + sem_L + podg + ecsw + bank_rows
        time_stage = 1.0 * ladder                                # selection + Q1/Q2 + paired timing (estimate)
        f = 1.0                         # S1 ladders: smaller step counts (estimate)
        units[lev] = dict(ref=ref, data=data, inter=inter, ladder=ladder * f, time=time_stage * f, sem=sem_L * f,
                          bank_rows=bank_rows * f)
    # bank training: UNMEASURED steady-state step time (the 220-step probe includes compilation); estimate from the
    # flop count (about 1e11 flop per sub-sampled step, 9x per full-batch step) at 5 TFLOP/s effective f64 on A100,
    # i.e. 20 ms / 180 ms -> 290k x 0.02 + 10k x 0.18 = 7600 s; taken as 2.5 h to cover the pilot's 3 A100-h guess
    bank_h = 3.0
    cells = []
    for geom in ('L', 'H'):
        for fam in ('wide', 'narrow'):
            for lev in ('S1', 'S2'):
                u = units[lev]
                sem_cut = 0.0 if geom == 'L' else 0.55 * u['sem']          # H: 30 spatial rows instead of 90
                sec = u['ref'] + u['data'] + u['inter'] + u['ladder'] + u['time'] - sem_cut
                cells.append(dict(cell=f'{geom}|{fam}|{lev}', gpu_h=sec / 3600 + bank_h, bank_h=bank_h))
    for fam in ('wide', 'narrow'):
        for lev in ('S1', 'S2'):
            u = units[lev]
            sec = u['ref'] + u['data'] + u['ladder'] - u['sem'] - u['bank_rows'] + 300 + 0.6 * u['time']  # C-SPEC
            cells.append(dict(cell=f'B|{fam}|{lev}', gpu_h=sec / 3600, bank_h=0.0))
    test_h = 12 * 0.3
    box_banks_h = 4 * bank_h                                   # conditional: only for winning cells
    base = sum(c['gpu_h'] for c in cells) + test_h + box_banks_h
    wide = sum(c['gpu_h'] for c in cells if '|wide|' in c['cell']) + test_h / 2
    narrowS2 = sum(c['gpu_h'] for c in cells if '|narrow|S2' in c['cell'])
    narrowS1 = sum(c['gpu_h'] for c in cells if '|narrow|S1' in c['cell'])
    proj = dict(
        gpu='A100-80GB (probe type)', headroom=HEAD, units_seconds=units, bank_hours_estimate=bank_h,
        cells=cells, test_jobs_h=test_h, box_banks_conditional_h=box_banks_h,
        total_registered_h=base, total_registered_with_headroom_h=base * HEAD,
        after_defer_narrowS2_h=(base - narrowS2) * HEAD,
        after_defer_narrowS2_S1_h=(base - narrowS2 - narrowS1) * HEAD,
        wide_only_with_box_banks_h=(wide + 2 * bank_h) * HEAD,
        cap=62, spent_smoke_h=json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())['total_gpu_hours'],
        panel_selected='wide-family geometry cells (4) + their box cells (2) + test jobs + conditional box banks (2); '
                       'narrow-family cells deferred (A5); narrow S1 only if spent + re-certified remainder allows',
        panel_selected_with_headroom_h=(wide + 2 * bank_h) * HEAD,
        panel_plus_spent_h=(wide + 2 * bank_h) * HEAD + json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())['total_gpu_hours'],
        memory_bounds=dict(host_data_S2_GB=4032 * 334273 * 8 / 1e9, device_pod_chunk_GB=4032 * 32768 * 8 / 1e9,
                           device_pod_qr_GB=2 * 334273 * 512 * 8 / 1e9, device_ecsw_G_GB=66 * 512 * 32768 * 8 / 1e9,
                           device_tensor_chunk_GB=1024 * 384 * 384 * 8 / 1e9 + 512 * 384 * 384 * 8 / 1e9,
                           bank_U_GB=4032 * 36801 * 8 / 1e9, probe_peaks_GB='see runs/smoke/smoke5/probe/*.json'),
        unmeasured=['bank steady-state step time (bound used: 3.0 h per bank)', 'ECSW NNLS fit time at r = 384 (90 s per fit assumed)',
                    'selection/Q1/Q2/timing stage (1.0 x ladder assumed)', 'S1 ladders = S2 ladders (assumed)',
                    'SEM screen 30 s per row (two measured rows: 5 s, 65 s); 30 feasible spatial configs assumed of 90 (L)'],
        status='PROVISIONAL: the listed units are assumptions, not measured or proven bounds; peak memories are per-array, not simultaneous')
    out = HERE / 'runs' / 'cost'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'projection.json').write_text(json.dumps(proj, indent=1))
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in proj.items()
                      if k in ('total_registered_h', 'total_registered_with_headroom_h', 'after_defer_narrowS2_h',
                               'after_defer_narrowS2_S1_h', 'wide_only_with_box_banks_h', 'spent_smoke_h')}))


if __name__ == '__main__':
    main()
