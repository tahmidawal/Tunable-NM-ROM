"""jcp-dims configuration-identity manifest (DESIGN A1.7). From the committed job configs it writes
configs/MANIFEST-identity.json: for every 2D (setting, arm) and every 3D (setting, rule, spec/form) the full numerical
identity — bank and rotation pins, r, M, mode-list policy and hash, rule name with draw, point count and rule sha256
(points + weights), q+ / q++ links, scheme, form, dt, tolerance levels (square-aware), budget, initialisation contract,
weighting, output times, metric id, population. CPU only (NumPy; no JAX tables).

    JAX_PLATFORMS=cpu python manifest_cells.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'
sys.path.insert(0, str(HERE / 'vendor' / 'quad2d' / 'vendor'))
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import p1core  # noqa: E402,F401  (sets the vendored 2D import paths)
import hops as H  # noqa: E402
import npcore as N3  # noqa: E402

import dcert as DC  # noqa: E402

ROOT = HERE.parents[1]
CKPT2 = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
ROT2 = HERE / 'vendor' / 'quad2d' / 'inputs' / 'rotation_R512.npz'
BANK3 = {'W1024': HERE / 'inputs' / 'model_W1024' / 'bank.pkl', 'M2': HERE / 'vendor' / 'quad3d' / 'inputs' / 'model_M2' / 'bank.pkl'}
OUT2 = dict(init='full-field input sampled at quadrature points; full-field decode (V4-4)', outputs='t=0:0.05:0.25',
            metric='2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1', budget=600)
SOLVER3 = dict(budget='adaptive: 600 LM iterations; fs: K sweeps after a adaptive start steps; qn: q_max 12',
               trust='0.05 x bank coefficient spread at r (speed3d)')
OUT3 = dict(init='mesh least-squares projection of the 65^3 field; lattice decode (speed3d mesh path)', outputs='t=0:0.05:0.25',
            metric='3|lattice=63^3|times=0:0.05:0.25|norm=fixed_t0|max_k>=1')


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


_ML = {}


def mode_hash2(policy, M):
    if (policy, M) not in _ML:
        if policy == 'lmf':
            k = np.arange(1, 1024)
            KX, KY = (a.ravel() for a in np.meshgrid(k, k, indexing='ij'))
            lam = 4 * 1024 ** 2 * (np.sin(np.pi * KX / 2048) ** 2 + np.sin(np.pi * KY / 2048) ** 2)
            o = np.lexsort((KX, lam, KX * KY))[:M]
            kx, ky = KX[o], KY[o]
        else:
            kx, ky, _ = H.modes_lean(1024, M)
        _ML[(policy, M)] = sha_arr(np.stack([np.asarray(kx, np.int64), np.asarray(ky, np.int64)]))
    return _ML[(policy, M)]


_RS = {}


def rule2(name):
    if name not in _RS:
        import rules2d
        X, w = rules2d.points(name)
        _RS[name] = dict(m=int(len(X)), sha256=sha_arr(np.concatenate([X.ravel(), w])))
    return _RS[name]


def rule3(name, pinned):
    if name not in _RS:
        import rules3d_names as RN
        X, w = RN.make(name, pinned)
        _RS[name] = dict(m=int(len(X)), sha256=RN.sha_rule(X, w))
    return _RS[name]


def main():
    out = dict(pins=dict(ckpt2=sha_file(CKPT2), rotation2=sha_file(ROT2), **{f'bank3_{k}': sha_file(v) for k, v in BANK3.items()}),
               entries2d=[], entries3d=[])
    for att in ('smk2', 'f2a', 'f2b', 'f2k', 'f2l', 'b2f', 'b2m'):
        cfg = json.loads((CFG / f'{att}.json').read_text())
        sets = cfg['settings']

        for s in sets:
            for a in s['arms']:
                sq = s['M'] == s['Rp']
                out['entries2d'].append(dict(
                    job=att, setting=s['name'], r=s['Rp'], M=s['M'], policy=s['policy'], mode_hash=mode_hash2(s['policy'], s['M']),
                    rule=s['rule'], rule_id=rule2(s['rule']), qp=s.get('qp'),
                    qpp=next((t.get('qp') for t in sets if t['rule'] == s.get('qp') and t['Rp'] == s['Rp'] and t['M'] == s['M']
                              and t['policy'] == s['policy']), None),
                    form=a['form'], scheme=a['scheme'], factor=a['factor'], dt=.005 * a['factor'], square=sq,
                    tol={lv: DC.tol_of(a['form'], lv, square=sq) for lv in ('prod', 'tight', 'tighter')},
                    weighting='LSPG D = 1/(a0 + dt b0 nu lam)' if a['form'] == 'LSPG' else 'GAL Q^T R = 0, A = Q Rm',
                    cases=a['cases'], tighter='dev6' if a.get('tighter', True) else None, rho_only=bool(a.get('rho_only')),
                    mesh=cfg['mesh'], edges=cfg['edges'], **OUT2))
    pinned = None
    import d3common as D3                 # imports JAX modules but builds nothing
    pinned = D3.load_rules_dims(None)
    for att in ('smk3', 'q3r4', 'm3a', 'm3b', 'k3', 'b3f', 'b3m'):
        cfg = json.loads((CFG / f'{att}.json').read_text())
        sets = {s['name']: s for s in cfg['settings']}
        for blk in cfg['blocks']:
            if blk['kind'] == 'rounds':
                for mm in blk['members'] + ([blk['control']] if blk.get('control') else []):
                    sn, rl = mm['key'].split('|')[:2]
                    out['entries3d'].append(dict(job=att, kind='round member', key=mm['key'], setting=sn, rule=rl,
                                                 rule_id=rule3(rl, pinned), qp=mm['qp'], qpp=mm.get('qpp'),
                                                 bank=sets[sn]['bank'], r=sets[sn]['Rp'], a=sets[sn]['a'],
                                                 M=len(N3.modes(65, 4 * sets[sn]['Rp'])[0]), spec='|'.join(mm['key'].split('|')[2:]),
                                                 cases='all64', qpp_cases=list(range(8)), **OUT3, **SOLVER3))
                continue
            if blk['kind'] not in ('ladder', 'adaptive'):
                continue
            s = sets[blk['setting']]
            Mreq = s.get('M') or None
            if Mreq is None:
                k, _ = N3.modes(65, 4 * s['Rp'])
                M = len(k)
            else:
                M = int(Mreq)
                k = N3.modes(65, M)[0][:M]
            specs = blk['specs'] if blk['kind'] == 'ladder' else [f"ad|{f}|{blk['dt']}|prod" for f in blk['forms']]
            for i, rl in enumerate(blk['rungs']):
                for sp in specs:
                    out['entries3d'].append(dict(
                        job=att, setting=s['name'], bank=s['bank'], r=s['Rp'], a=s['a'], M=M, mode_hash=sha_arr(np.asarray(k, np.int64)),
                        rule=rl, rule_id=rule3(rl, pinned), qp=blk['rungs'][i + 1] if i + 1 < len(blk['rungs']) else None,
                        qpp=blk['rungs'][i + 2] if i + 2 < len(blk['rungs']) else None, spec=sp, square=(M == s['Rp']),
                        cases=blk.get('cases', 'all64'), tighter_cases=blk.get('tighter_cases', list(range(8))),
                        tag_prefix=blk.get('tag_prefix', ''), trust_fraction=cfg['trust_fraction'],
                        tol=(DC.tol_of(sp.split('|')[1], 'prod', square=(M == s['Rp'])) if sp.startswith('ad|')
                             else 'fs/qn: gtol 1e-3 (speed3d); ADT anchors tight (1e-7, 0) / tighter (1e-8, 0)'),
                        weighting=('GAL Q^T R = 0' if sp.startswith('ad|GAL') else 'LSPG D = 1/(a0 + dt b0 nu lam)'),
                        **OUT3, **SOLVER3))
    (CFG / 'MANIFEST-identity.json').write_text(json.dumps(out, indent=1) + '\n')
    print('2D entries', len(out['entries2d']), '3D entries', len(out['entries3d']))


if __name__ == '__main__':
    main()
