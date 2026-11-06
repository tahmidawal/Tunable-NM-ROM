"""Identity-bound reference projections onto a retrained bank (G5-A5, Codex finding 4).

Usage: rom_project.py --manifest <json> --export <dir> --bank-eval <trained_bank_eval.npz>
       --tag trained --case <name> --out-dir <dir> [--allow-heldout]
Writes <out-dir>/proj_<case>_<tag>.npz with uv_<tag>, the bank-eval SHA-256 and the export manifest SHA-256.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rom_export as rx


def main():
    ap = argparse.ArgumentParser()
    for k in ('--manifest', '--export', '--bank-eval', '--tag', '--case', '--out-dir'):
        ap.add_argument(k, required=True)
    ap.add_argument('--allow-heldout', action='store_true'); ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    import ngsolve as ng
    ng.SetNumThreads(a.threads)
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    if case['split'] == 'train' or (case['split'] == 'heldout' and not a.allow_heldout):
        raise PermissionError('projection of this split is not allowed here')
    ex = Path(a.export)
    q = np.load(ex / 'quadrature.npz'); P = np.load(ex / 'basis.npz')['P']
    bank = np.load(a.bank_eval)
    with ng.TaskManager():
        fe = rx.FE(ng, man['mesh_path'], man['mesh_sha256'], k=man['velocity_order'],
                   qorder=man['quadrature_order'])
        pts, w = fe.rule()
        if pts.shape != q['points'].shape or np.max(np.abs(pts - q['points'])) > 1e-12:
            raise ValueError('quadrature differs from the export')
        rows = rx.project(fe, w, P, case, {f'_{a.tag}': bank['V']})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)   # never write into another job's export
    np.savez(out / f'proj_{a.case}_{a.tag}.npz', **rows, bank_eval_sha256=np.array(rx.sha(a.bank_eval)),
             export_manifest_sha256=np.array(rx.sha(ex / 'EXPORT.json')))


if __name__ == '__main__':
    main()
