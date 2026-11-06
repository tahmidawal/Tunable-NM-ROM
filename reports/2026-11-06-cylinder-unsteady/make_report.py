"""Build the G5 unsteady-cylinder report from saved JSON/NPZ only (no hand-typed numbers).

Usage: /home/tahmid/Dev/.venv/bin/python make_report.py
Reads the three G5 lanes' collected artifacts, copies figures/animation, and writes
2026-11-06-cylinder-unsteady.md, report-numbers.json and job-ledger.json next to this file.
"""
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees')
CO = W / '2026-11-03-cylinder-unsteady/experiments/cylinder-unsteady'
REF = W / '2026-11-03-cylinder-unsteady-ref/experiments/cylinder-unsteady-ref/artifacts'
ROM = W / '2026-11-03-cylinder-unsteady-rom/experiments/cylinder-unsteady-rom/artifacts'


def J(p):
    return json.loads(Path(p).read_text())


def e(x, d=2):
    return '—' if x is None else f'{x:.{d}e}'


def ledger():
    out = subprocess.run(['ssh', 'tufts-login', "sacct -u $USER -S 2026-11-03 -X -n -P -o JobID,JobName,State,Elapsed,NodeList"],
                         capture_output=True, text=True, timeout=60).stdout
    rows = []
    for line in out.strip().splitlines():
        jid, name, state, el, node = line.split('|')
        if not ('cylu' in name or 'cylr' in name):
            continue
        h, m, s = (el.split('-')[-1]).split(':')
        days = int(el.split('-')[0]) if '-' in el else 0
        sec = days * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
        rows.append({'job': jid, 'name': name, 'state': state, 'elapsed_s': sec, 'node': node})
    return rows


def main():
    fig = HERE / 'figures'; fig.mkdir(exist_ok=True)
    for f in ('g5_reference.png', 'g5_rom_dev.png', 'g5_rom_dev_anim.gif', 'g5_rom_dev_anim_frame.png'):
        shutil.copyfile(CO / 'figures' / f, fig / f)
    led = ledger(); (HERE / 'job-ledger.json').write_text(json.dumps(led, indent=2) + '\n')
    gpu_h = sum(r['elapsed_s'] for r in led) / 3600
    r1 = J(REF / 'R1DE.json'); gates = J(REF / 'r1c-space02/results/compare/R1BC-GATES.json')
    cmp = {f'{c}_{2 * c}': J(REF / f'r1b-compare01/results/compare/compare_{c}_{2 * c}.json') for c in (800, 1600, 3200)}
    space = J(REF / 'r1c-space02/results/compare/space_3200.json')
    fl = J(ROM / 'r2-export01/results/R2-FLOORS.json')
    tr = J(ROM / 'r3-train01/results/train/TRAINING.json')
    dev = J(ROM / 'r4-bank-qcheck01/results/r4/R4-re95.json')
    ho = J(ROM / 'r5-heldout01/results/r4_heldout/R4-re105.json')
    qc = J(ROM / 'r4-qcheck02/results/QCHECK.json')
    tm = J(ROM / 'r5-heldout01/results/timing/R5-TIMING.json')
    bl = J(ROM / 'r5-heldout01/results/BL_magnitude.json')
    nums = {'gpu_hours': gpu_h, 'R1': {k: r1[k] for k in ('R1b', 'R1c', 'R1d_gate', 'R1e_gate', 'dt0_denominator')},
            'R1d': r1['R1d'], 'uncertainty': r1['uncertainty'], 'R2': fl, 'dev': dev, 'heldout': ho, 'qcheck': qc,
            'timing_summary': tm['summary'], 'BL': bl}
    (HERE / 'report-numbers.json').write_text(json.dumps(nums, indent=2, default=str) + '\n')
    arms = [('pod_r8', 'POD r=8'), ('pod_r16', 'POD r=16'), ('pod_r24', 'POD r=24'),
            ('bank_r16', 'retrained neural bank r=16'), ('frozen_bank_r4', 'frozen steady bank r=4')]

    def arm_table(R):
        L = ['| Arm | unknowns (vel + p) | e_u RMS | e_u max | e_p RMS | St err (rel) | C_D,max err (rel) | C_L,max err (abs) | divergence max | targets met |',
             '|---|---|---|---|---|---|---|---|---|---|']
        for k, lab in arms:
            a = R['arms'][k]
            if a['status'] != 'completed':
                L.append(f'| {lab} | failed: {a.get("error")} |||||||||'); continue
            ce = a.get('cycle_errors', {})
            met = sum(1 for v in a['targets'].values() if v['met'])
            L.append(f"| {lab} | {a['velocity_dimension']} + {a['pressure_dimension']} | {e(a['e_u']['rms'])} | {e(a['e_u']['max'])} | "
                     f"{e(a['e_p']['rms'])} | {e(ce.get('St_rel'))} | {e(ce.get('CD_max_rel'))} | {e(ce.get('CL_max_abs'))} | "
                     f"{e(a['divergence']['max'])} | {met}/6 |")
        return '\n'.join(L)
    S = tm['summary']
    tim = ['| Method | warm s per 0.1 s workload (median of 7) | cold s (median) | warm speed-up vs FE | velocity accuracy witness (max) |', '|---|---|---|---|---|',
           f"| FE reference stepper (P4/P3, dt0, 16 threads) | {S['fe_warm_median_s']:.2f} | {S['fe_cold_median_s']:.2f} | 1 | "
           f"{e(max(x['accuracy_velocity_rel_L2_vs_record_max'] for x in tm['fe']))} (restart vs record) |"]
    for k, lab in arms:
        r = S['rom'][k]
        tim.append(f"| {lab} | {r['warm_median_s']:.4f} | {r['cold_median_s']:.2f} | {r['warm_speedup_vs_fe_warm']:.0f}× | {e(r['accuracy_e_u_max'])} |")
    fl_rows = ['| Basis | median floor | max floor | passes floor rule |', '|---|---|---|---|']
    for k in ['frozen_bank_r4'] + [f'pod_r{r}' for r in (2, 4, 8, 12, 16, 20, 24)]:
        a = fl['arms'][k]
        fl_rows.append(f"| {k} | {e(a['velocity_basis_only']['median'])} | {e(a['velocity_basis_only']['max'])} | {'yes' if a['passes_floor'] else 'no'} |")
    d = r1['R1d']
    dfg = ['| Quantity | this reference | Featflow 2014 finest | error | tolerance | pass |', '|---|---|---|---|---|---|']
    for k in ('CD_max', 'CL_max', 'St', 'dp_at_t0_plus_half_period'):
        dfg.append(f"| {k} | {d[k]['ours']:.6f} | {d[k]['featflow_finest']:.6f} | {e(d[k]['error'])} ({d[k]['kind']}) | {d[k]['tolerance']} | {'yes' if d[k]['pass'] else 'NO'} |")
    ladder = ['| pair (dt) | e_u | e_p | e_D | e_L (tol 2e-4) |', '|---|---|---|---|---|']
    for k, v in cmp.items():
        a, b = k.split('_')
        ladder.append(f"| 1/{a} vs 1/{b} | {e(v['e_u_max'])} | {e(v['e_p_max'])} | {e(v['e_D'])} | {e(v['e_L'])} |")
    ladder.append(f"| space: P3/P2 vs P4/P3 at 1/3200 | {e(space['e_u_max'])} | {e(space['e_p_max'])} | {e(space['e_D'])} | {e(space['e_L'])} |")
    qrows = ['| Arm | max operator change (tol 1e-6) | largest operator | floor change (tol 1e-3) | R4 metric change (tol 1e-2) |', '|---|---|---|---|---|']
    for k, lab in arms:
        a = qc['arms'][k]; big = max(a['relative_change'], key=a['relative_change'].get)
        qrows.append(f"| {lab} | {e(a['operator_max'])} | {big} | {e(a['floor_relative_change'])} | {e(max(a['r4_metric_relative_change'].values()))} |")
    md = HERE / '2026-11-06-cylinder-unsteady.md'
    md.write_text(TEMPLATE.format(gpu_h=gpu_h, ladder='\n'.join(ladder), dfg='\n'.join(dfg), floors='\n'.join(fl_rows),
                                  dev=arm_table(dev), ho=arm_table(ho), timing='\n'.join(tim), qtable='\n'.join(qrows),
                                  dt0=r1['dt0_denominator'], unc_u=e(r1['uncertainty']['field_and_force_over_horizon']['e_u_max']),
                                  unc_p=e(r1['uncertainty']['field_and_force_over_horizon']['e_p_max']),
                                  bl12=e(bl['order12']['norm_BL']), bl14=e(bl['order14']['norm_BL']), blB=e(bl['order12']['norm_B']),
                                  parent=e(tr['parent']['objective']), child=e(tr['child']['objective']),
                                  njobs=len(led)))
    print('written', md)


TEMPLATE = open(HERE / 'report_template.md').read() if (HERE / 'report_template.md').exists() else ''

if __name__ == '__main__':
    main()
