"""Test-once evaluation (DESIGN A2.18, A3.27): the frozen union and per-family winners at the decisive targets for C = 100,
run once on the 16 test cases against test references built by vref.py after the selection was frozen.

    python vtest.py --config configs/test-run.json --out <dir>

config: {"contrast": 100, "keys": [...], "refdir": [...], "frozen_selection_sha256": "...", "test_once": true}
Not used for selection or for the gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vcoef as VC     # noqa: E402
import vkeys as VK     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    import testgate
    allowed = testgate.verify(cfg)                       # frozen selection + opening record (A2.18) before generation
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C = float(cfg['contrast'])
    test = VC.cohort('test', unlock_test=True)
    dirs = cfg['refdir'] if isinstance(cfg['refdir'], list) else [cfg['refdir']]
    refres = {}
    for d_ in dirs:
        rj = Path(d_) / 'result.json'
        if rj.exists():
            refres.update(json.loads(rj.read_text())['cases'])
    rep = dict(config=cfg, backend=jax.default_backend(), keys={}, permitted=allowed)
    for k in cfg['keys']:
        sp = VK.parse(k)
        p = VK.program(sp)
        rows = []
        for j, c in enumerate(test):
            cid = f'test|{j}'
            r = refres.get(f'C{C:g}|{cid}')
            elig = bool(r and r['judge']['outcome'] == 'accepted' and r['eps_ref'] is not None and r['eps_ref'] <= 1e-3)
            x0, _ = VK.prepare(sp, c)
            F, st = p(x0, jnp.asarray(VC.kappa_params(c, C)))
            F = jax.block_until_ready(F)
            ok, ev = VK.validity(sp, F, st)
            e = None
            if ok and r:
                fs = [Path(d_) / f'ref_C{C:g}_test_{j}.npz' for d_ in dirs]
                z = np.load([f for f in fs if f.exists()][-1])
                e = VK.lat_err(np.asarray(F), z['lat'], float(np.linalg.norm(z['t0'].ravel())))
            rows.append(dict(case=cid, valid=ok, e=e, eligible=elig, eps_ref=r['eps_ref'] if r else None))
        el = [x['e'] for x in rows if x['eligible'] and x['valid'] and x['e'] is not None]
        rep['keys'][k] = dict(rows=rows, n_eligible=len(el), median=float(np.median(el)) if el else None,
                              worst=float(np.max(el)) if el else None, all_valid=all(x['valid'] for x in rows))
        print(k, rep['keys'][k]['n_eligible'], rep['keys'][k]['median'], rep['keys'][k]['worst'], flush=True)
    rep['complete'] = True
    (out / 'result.json').write_text(json.dumps(rep, indent=1, default=float))
    (out / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    main()
