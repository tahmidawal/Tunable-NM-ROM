"""Assemble the comparison table and figures from saved JSON / series files (nothing retyped).

Usage: make_report.py --final <dir with final_eval.json> --train label=<dir with summary.json> [...]
                      --rom-dev <R4-re95.json> [--rom-heldout <R4-re105.json> ...] --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ('e_u_rms', 'e_u_max', 'e_p_rms', 'St_rel', 'CD_max_rel', 'CL_max_abs')


def row_from_rec(r):
    if r.get('status') != 'completed':
        return {'status': r.get('status', 'failed'), 'targets_met': 0}
    out = {k: r['targets'][k]['value'] for k in KEYS}
    out.update(status='completed', targets_met=int(sum(bool(r['targets'][k]['met']) for k in KEYS)),
               last_period_e_u=r['e_u']['last_period_mean'])
    ph = r.get('phase') or r.get('phase_aligned') or {}
    out['lag_saves'] = ph.get('lag_saves')
    return out


def fmt(v):
    return '-' if v is None else (f'{v:.2e}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', required=True); ap.add_argument('--train', action='append', default=[])
    ap.add_argument('--rom-dev', required=True); ap.add_argument('--rom-heldout', action='append', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fin = json.loads((Path(a.final) / 'final_eval.json').read_text())
    train = {s.split('=', 1)[0]: json.loads((Path(s.split('=', 1)[1]) / 'summary.json').read_text()) for s in a.train}
    table = {'re95': {}, 're105': {}, 'sources': {'rom_dev': a.rom_dev, 'rom_heldout': a.rom_heldout, 'final': a.final}}
    for label, rec in fin['arms'].items():
        for case, r in rec['cases'].items():
            row = row_from_rec(r)
            row['seconds_per_window'] = r['inference']['median_seconds']
            row['seconds_min_max'] = [r['inference']['min_seconds'], r['inference']['max_seconds']]
            row['hardware'] = r['inference']['device']
            row['config'] = rec['config'].get('name'); row['params'] = rec['params']
            if label in train:
                t = train[label]
                row['train_gpu_hours'] = t['training_seconds_total'] / 3600
                row['train_gpu'] = t['gpu']
                row['plateau'] = t['final']['train'].get('plateau')
                row['relative_drop_last_20pct'] = t['final']['train'].get('relative_drop_last_20pct')
                row['validation'] = {n: c.get('val_e_u_rms') for n, c in t['candidates'].items()}
            table[case][label] = row
    rom = json.loads(Path(a.rom_dev).read_text())
    for name, r in rom['arms'].items():
        row = row_from_rec(r); row['seconds_per_window'] = r.get('rollout_seconds'); row['hardware'] = 'CPU (ROM lane)'
        table['re95'][name] = row
    for p in a.rom_heldout:
        j = json.loads(Path(p).read_text())
        for name, r in j['arms'].items():
            row = row_from_rec(r); row['seconds_per_window'] = r.get('rollout_seconds'); row['hardware'] = 'CPU (ROM lane)'
            row['source'] = p
            table['re105'][name] = row
    table['reference_readout'] = {c: {k: v for k, v in r.items() if k != 'representation_floors'} for c, r in fin['reference_readout'].items()}
    (out / 'table.json').write_text(json.dumps(table, indent=2, default=float) + '\n')
    lines = []
    for case in ('re95', 're105'):
        if not table[case]:
            continue
        lines.append(f'\n### {case}\n')
        lines.append('| arm | e_u RMS | e_u max | e_p RMS | St rel | CD_max rel | CL_max abs | met | lag (saves) | s/window | hardware |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
        lines.append('| target | 1e-2 | 3e-2 | 2e-2 | 1e-2 | 2e-2 | 2e-2 | 6 | <=2 | | |')
        for name, r in table[case].items():
            lines.append(f'| {name} | ' + ' | '.join(fmt(r.get(k)) for k in KEYS) + f" | {r['targets_met']}/6 | {fmt(r.get('lag_saves'))} | "
                         f"{fmt(r.get('seconds_per_window'))} | {r.get('hardware', '')} |")
    (out / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'node': '#1f77b4', 'grid': '#d62728', 'don': '#2ca02c'}
    romdir = Path(a.rom_dev).parent
    for case in fin['reference_readout']:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for label in fin['arms']:
            f = Path(a.final) / f'series_{label}_{case}.npz'
            if not f.exists():
                continue
            s = np.load(f)
            ax[0].semilogy(s['t'] - s['t'][0], s['e_u'], color=colors.get(label), label=f"{label} ({table[case][label].get('config')})")
            if 'CL' in s:
                ax[1].plot(s['t'] - s['t'][0], s['CL'], color=colors.get(label), lw=1, label=label)
                ref = (s['t'] - s['t'][0], s['CL_ref'])
        if case == 're95':
            for name, c in (('bank_r32', 'k'), ('bank_p16', '0.5'), ('pod_r16', '#9467bd')):
                f = romdir / f'series_re95_{name}.npz'
                if f.exists():
                    s = np.load(f)
                    ax[0].semilogy(s['t_save'] - s['t_save'][0], s['e_u'], color=c, ls='--', label=f'ROM lane {name}')
        ax[0].axhline(1e-2, color='gray', lw=0.6, ls=':'); ax[0].set_xlabel('time in window [s]'); ax[0].set_ylabel('relative velocity error e_u(t) on Q')
        ax[0].legend(fontsize=7); ax[0].set_title(f'{case}: rollout error from the first state')
        try:
            ax[1].plot(*ref, 'k--', lw=0.8, label='reference')
        except NameError:
            pass
        ax[1].set_xlabel('time in window [s]'); ax[1].set_ylabel('C_L'); ax[1].legend(fontsize=7); ax[1].set_title('lift')
        fig.tight_layout(); fig.savefig(out / f'rollout_{case}.png', dpi=160); plt.close(fig)
    print('figures in', out)


if __name__ == '__main__':
    main()
