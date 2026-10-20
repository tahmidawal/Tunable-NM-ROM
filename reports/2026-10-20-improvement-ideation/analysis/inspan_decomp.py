"""Per-case error decomposition for the frozen r = 384 LSPG-CN-R ROM on the jcp-regimes cohorts (CPU, NumPy only).

For each case: v(t) = decoded ROM field (saved coefficient outputs W, decoded on the 257^2 lattice by the deployed bank),
u(t) = E0-style reference (jcp-regimes store/2d/<regime>/ST/cNNN.npz, array a2_L8192), P = lattice least-squares projection
onto the first 384 bank functions. Since v lies in the span, ||v-u||^2 = ||(I-P)u||^2 + ||v-Pu||^2 at each time.
Reports, per case, max over t1..t5 of each term relative to ||u(t0)||, then medians per regime: total error, floor
(outside the span) and distance to the projected reference (inside the span; reduced-model, initialisation and
numerical errors together). Writes analysis/inspan_decomp.json.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
LANE = ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'
sys.path.insert(0, str(LANE))
import npcheck as NC  # noqa: E402
Z = np.load(LANE / 'runs/rom1024/output/qacc_rom1024/R_R384_LSPG_CNR_1.npz', allow_pickle=True)
A = json.loads((LANE / 'checks/analysis.json').read_text())['configs']['R384|LSPG_CNR_1']['per_case']
NC.decode257(np.zeros((1, 384)), 384)
G = NC._G257[384]
U_, s_, _ = np.linalg.svd(G, full_matrices=False)
Q = U_[:, s_ > s_[0] * 1e-13]
rows = {}
for i, cid in enumerate(Z['case_ids']):
    coh, c = str(cid).split('|')
    if coh in ('dev6', 'val32') or not Z['valid'][i] or not A.get(str(cid), {}).get('eligible', False):
        continue
    ref = np.load(LANE / f'store/2d/{coh}/ST/c{int(c):03d}.npz')['a2_L8192'].reshape(6, -1)
    v = (Z['W'][i] @ G.T)
    n0 = np.linalg.norm(ref[0])
    Pu = (Q @ (Q.T @ ref[1:].T)).T
    tot = np.linalg.norm(v[1:] - ref[1:], axis=1) / n0
    flo = np.linalg.norm(ref[1:] - Pu, axis=1) / n0
    ins = np.linalg.norm(v[1:] - Pu, axis=1) / n0
    rows.setdefault(coh, []).append(dict(case=str(cid), total=float(tot.max()), floor=float(flo.max()), inspan=float(ins.max()),
                                         lane_e=A[str(cid)]['e']))
out = dict(source=[str((LANE / 'runs/rom1024/output/qacc_rom1024/R_R384_LSPG_CNR_1.npz').relative_to(ROOT)), 'store/2d/<regime>/ST'],
           regimes={k: dict(n=len(v), median_total=float(np.median([x['total'] for x in v])), median_floor=float(np.median([x['floor'] for x in v])),
                            median_inspan=float(np.median([x['inspan'] for x in v])),
                            max_abs_diff_total_vs_lane=float(max(abs(x['total'] - x['lane_e']) for x in v)), rows=v) for k, v in rows.items()})
Path(__file__).with_name('inspan_decomp.json').write_text(json.dumps(out, indent=1))
for k, v in out['regimes'].items():
    print(k, v['n'], round(100 * v['median_total'], 3), round(100 * v['median_floor'], 3), round(100 * v['median_inspan'], 3), v['max_abs_diff_total_vs_lane'])
