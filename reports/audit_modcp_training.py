"""Persist an independent audit of the nine trusted local pilot checkpoints."""
import argparse
import json
from pathlib import Path

from modcp_audit import audit_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    folders = [('burgers2d', ROOT/'worktrees/2026-09-10-modcp-burgers2d/experiments/modcp-eq/runs/train01/out/checkpoints')]
    for boundary in ('reflective', 'absorbing'):
        folders.append((f'wave_{boundary}', ROOT/f'worktrees/2026-09-10-modcp-wave2d/experiments/modcp-eq/runs/train_{boundary}01/cluster/out/training/checkpoints'))
    rows = []
    for case, folder in folders:
        for architecture in ('cp', 'modcp', 'film'):
            row = audit_checkpoint(folder/f'{architecture}.pkl')
            if row['completed_updates'] != 16000:
                raise ValueError('Pilot checkpoint does not meet the approved training budget')
            cfg = row['configuration']
            expected = (16, 1, 'dirichlet', [384, 16]) if case == 'burgers2d' else (
                32, 2, 'free' if case == 'wave_absorbing' else 'dirichlet', [3136, 32])
            if (cfg['k'], cfg['outputs'], cfg['boundary'], row['code_shape']) != expected:
                raise ValueError(f'Wrong pilot training population or model state: {case}')
            if cfg['architecture'] != architecture or cfg['intervals'] != 256 or cfg['rank'] != 64:
                raise ValueError('Wrong pilot architecture, training mesh, or CP rank')
            row.update(case_name=case, path=str(Path(row['path']).relative_to(ROOT)))
            rows.append(row)
    out = ROOT/'reports'/f'{args.date}-modified-cp-training-audit.json'
    out.write_text(json.dumps({'scope': 'checkpoint integrity and training completion only; no rollout accuracy or speed conclusion',
                               'checkpoints': rows}, indent=2, allow_nan=False)+'\n')
    print(f'PASS {len(rows)} finite f64 completed checkpoints; {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
