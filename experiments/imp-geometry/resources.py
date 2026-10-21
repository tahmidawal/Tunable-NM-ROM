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


def main():
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
        sem_L = 90 * 15.0 + 8 * 30.0 + 8 * 2 * 6 * 5.0 + 8 * 4 * 10.0
        podg = 10 * 4 * 5 * (pod['podg_r384_lspg_cnr_32'] + pod['podg_r384_gal_etdrk4_32']) / 2 * 0.7
        ecsw = 10 * 2 * 90.0 + 10 * 3 * 2 * 5 * 5.0               # NNLS fits (estimate) + rows
        bank_rows = 10 * 21 * bk['bank_r384_lspg_32'] * 1.2 + 6 * bk['bank_space_q12']
        ladder = fd_rows + sem_L + podg + ecsw + bank_rows
        time_stage = 0.5 * ladder                                # selection + Q1/Q2 + paired timing (estimate)
        f = 1.0 if lev == 'S2' else 0.6                         # S1 ladders: smaller step counts (estimate)
        units[lev] = dict(ref=ref, data=data, inter=inter, ladder=ladder * f, time=time_stage * f, sem=sem_L * f,
                          bank_rows=bank_rows * f)
    # bank training: UNMEASURED steady-state step time (the 220-step probe includes compilation); estimate from the
    # flop count (about 1e11 flop per sub-sampled step, 9x per full-batch step) at 5 TFLOP/s effective f64 on A100,
    # i.e. 20 ms / 180 ms -> 290k x 0.02 + 10k x 0.18 = 7600 s; taken as 2.5 h to cover the pilot's 3 A100-h guess
    bank_h = 2.5
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
    base = sum(c['gpu_h'] for c in cells) + test_h
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
        unmeasured=['bank steady-state step time (estimated 2.5 h per bank)', 'ECSW NNLS fit time at r = 384',
                    'selection/Q1/Q2/timing stage (estimated 0.5 x ladder)', 'S1 ladder factor 0.6',
                    'SEM spatial screen average row cost 15 s (two measured rows: 5 s, 65 s)'],
        status='PROVISIONAL certificate: several units are estimates, not measurements (listed above)')
    out = HERE / 'runs' / 'cost'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'projection.json').write_text(json.dumps(proj, indent=1))
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in proj.items()
                      if k in ('total_registered_h', 'total_registered_with_headroom_h', 'after_defer_narrowS2_h',
                               'after_defer_narrowS2_S1_h', 'wide_only_with_box_banks_h', 'spent_smoke_h')}))


if __name__ == '__main__':
    main()
