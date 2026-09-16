"""Turn a completed training run into the solve job's model list and POD cohort.

Reads only the training `result.json`: the two pre-registered head primaries, their
correction bases, and the selected bank's training-source count, which is the cohort
the POD competitor is rebuilt from.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pbh_audit_np as N


def build_basis(checkpoint, intervals, count, draws, fit):
    """The `diagnose_head_corrections` construction, in pure NumPy.

    Right singular vectors of the normalised training residuals in the exact QR
    physical metric, nested prefixes, verified orthonormal in the field metric.
    """
    params, Z, _ = N.load(checkpoint)
    xy = N.coords(intervals)
    G = N.features(params, xy)
    Q, R = np.linalg.qr(G)
    U = np.stack([N.solve(intervals, p)[1:-1, 1:-1].ravel() for p in draws[fit]])
    T = U @ Q
    norm = np.linalg.norm(U, axis=1)
    residual = (T - N.head(params, Z) @ R.T) / norm[:, None]
    _, s, Vt = np.linalg.svd(residual, full_matrices=False)
    count = int(min(count, Vt.shape[0]))
    physical = Vt[:count].T
    directions = np.linalg.solve(R, physical)
    W = G @ directions
    orth = float(np.linalg.norm(W.T @ W - np.eye(count)))
    assert orth < 1e-8, orth
    return dict(coefficient_directions=directions, physical_metric_directions=physical,
                R=R, singular_values=s, training_latents=Z, training_norms=norm), orth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='collected training output directory')
    ap.add_argument('--stage', type=Path, required=True, help='where to copy the staged files')
    ap.add_argument('--config', type=Path, required=True, help='config-solve.json to update')
    a = ap.parse_args()
    d = json.loads((a.run / 'result.json').read_text())
    assert d['complete']
    a.stage.mkdir(parents=True, exist_ok=True)
    models = [dict(id='incumbent', role='control', checkpoint='r128_joint.pkl',
                   basis='basis.npz')]
    extra = []
    for sel in d['selection']['heads']:
        arm = next(x for x in d['head_arms'] if x['arm'] == sel['selected'])
        ck = next(x for x in d['checkpoints'] if x['id'] == sel['selected'])
        src = a.run / ck['path']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == ck['sha256'], ck['id']
        name = f"primary_K{sel['K']}.pkl"
        shutil.copy2(src, a.stage / name)
        bsrc = a.run / sel['basis']['path']
        assert hashlib.sha256(bsrc.read_bytes()).hexdigest() == sel['basis']['sha256']
        bname = f"primary_K{sel['K']}-basis.npz"
        shutil.copy2(bsrc, a.stage / bname)
        models.append(dict(id=f"new_K{sel['K']}", role='primary', checkpoint=name,
                           basis=bname, source_arm=sel['selected'], K=sel['K'],
                           R=arm['R'], sources=arm['S'], beta_weak=arm['beta_weak'],
                           beta_smooth=arm['beta_smooth'], bank=arm['bank'],
                           checkpoint_sha256=ck['sha256'], basis_sha256=sel['basis']['sha256']))
        extra += [str(a.stage / name), str(a.stage / bname)]
    # The selected bank arm carries its OWN jointly trained head. It is not a
    # pre-registered primary, but leaving it out would hide whether the frozen-bank
    # head sweep actually improved on the head the bank was trained with.
    bank = d['selection']['bank']
    arm = next(x for x in d['bank_arms'] if x['arm'] == bank['selected'])
    ck = next(x for x in d['checkpoints'] if x['id'] == bank['selected'])
    src = a.run / ck['path']
    assert hashlib.sha256(src.read_bytes()).hexdigest() == ck['sha256']
    shutil.copy2(src, a.stage / 'bankarm_head.pkl')
    draws = N.source_params(d['config']['train_seed'], arm['S'])
    fit = np.asarray(d['cohorts'][str(arm['S'])]['fit'])
    payload, orth = build_basis(a.stage / 'bankarm_head.pkl', d['config']['training_intervals'],
                                d['config']['correction_count'], draws, fit)
    np.savez_compressed(a.stage / 'bankarm_head-basis.npz', **payload)
    models.append(dict(id='bank_arm_head', role='bank arm own jointly trained head, reported '
                                                'but not a pre-registered primary',
                       checkpoint='bankarm_head.pkl', basis='bankarm_head-basis.npz',
                       source_arm=bank['selected'], K=arm['K'], R=arm['R'], sources=arm['S'],
                       basis_orthogonality_error=orth,
                       basis_construction='pure NumPy, built after collection from the '
                                          'checkpoint and regenerated training truth'))
    extra += [str(a.stage / 'bankarm_head.pkl'), str(a.stage / 'bankarm_head-basis.npz')]

    (a.stage / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
    extra.append(str(a.stage / 'models.json'))
    cfg = json.loads(a.config.read_text())
    S = d['selection']['bank']['S']
    for c in cfg['pod_cohorts']:
        if c['id'] == 'trainset':
            c['count'] = int(S)
            c['note'] = (f"the selected bank's own training cohort "
                         f"({d['selection']['bank']['selected']}), so the POD competitor sees "
                         f"exactly the snapshots the neural bank was trained on")
    a.config.write_text(json.dumps(cfg, indent=2) + '\n')
    print(' '.join(extra))
    print('pod trainset count ->', S)


if __name__ == '__main__':
    main()
