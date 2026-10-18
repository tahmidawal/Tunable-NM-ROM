"""Extract the kernel fixtures (DESIGN 2.4, AM-3) from the combined archives into small lane inputs, with provenance.

P1a: W_R512.npz run 'LSPG|CNR|0.005|prod', dev6 cases 0-5, output indices 1 and 5 (R512, gauss192, 1024^2).
P1b: W_grid.npz key 'W1024|lat65536|fs|BE|0.01|K1', 923801 cases 0-5, output indices 1 and 5.
The archived R512 configuration is verified from combined configs/c2grid.json (rule gauss192, mesh 1024).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cfg = json.loads((C / 'configs/c2grid.json').read_text())
    s512 = [s for s in cfg['settings'] if s['name'] == 'R512'][0]
    assert s512 == dict(name='R512', Rp=512, rule='gauss192') and cfg['mesh'] == 1024
    a = C / 'runs/c2grid/archive/output/W_R512.npz'
    z = np.load(a)
    runs, cases = list(z['runs']), list(z['cases'])
    ri = runs.index('LSPG|CNR|0.005|prod')
    assert ri == 15
    F, lab = [], []
    for c in range(6):
        for k in (1, 5):
            F.append(z['W'][cases.index(f'dev6|{c}'), ri, k])
            lab.append(f'dev6|{c}|t{k}')
    F = np.array(F)
    assert np.isfinite(F).all() and F.shape == (12, 512)
    b = C / 'runs/c3grid/archive/output/W_grid.npz'
    zb = np.load(b)
    key = 'W1024|lat65536|fs|BE|0.01|K1'
    Wb = zb[key]
    G = np.array([Wb[c, k] for c in range(6) for k in (1, 5)])
    labb = [f'923801|{c}|t{k}' for c in range(6) for k in (1, 5)]
    assert np.isfinite(G).all() and G.shape == (12, 1024)
    (HERE / 'inputs').mkdir(exist_ok=True)
    np.savez(HERE / 'inputs/fixtures_p1a.npz', W=F, labels=np.array(lab))
    np.savez(HERE / 'inputs/fixtures_p1b.npz', W=G, labels=np.array(labb))
    prov = dict(p1a=dict(source=str(a), source_sha256=sha(a), run='LSPG|CNR|0.005|prod', run_index=ri, outputs=[1, 5],
                         config_check=s512, file_sha256=sha(HERE / 'inputs/fixtures_p1a.npz')),
                p1b=dict(source=str(b), source_sha256=sha(b), key=key, outputs=[1, 5], cases=list(range(6)),
                         file_sha256=sha(HERE / 'inputs/fixtures_p1b.npz')))
    (HERE / 'inputs/fixtures.json').write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))


if __name__ == '__main__':
    main()
