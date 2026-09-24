"""Train extra head variants on a finished training job's saved head inputs (head_data.npz: ordered coefficients of
the training snapshots and of the bank-validation snapshots, per-group metric factors). Training-side only: the
selection statistic is the bank-validation best-found worst error, exactly as in train2.train_head. No ROM, no
validation or held-out cohort is touched.

    python head_train.py --data <dir with head_data.npz> --config <json: base train config + head_variants_extra>
                         --out <dir>
head_variants_extra: list of dicts {k, code_reg, width, depth, steps, lr, batch} (missing keys: base config).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train2 as T2  # noqa: E402
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(a.data) / 'head_data.npz')
    print(f'jax_backend={jax.default_backend()}', flush=True)
    rep = dict(config=cfg, heads={})
    for v in cfg['head_variants_extra']:
        c2 = dict(cfg, head_width=v.get('width', cfg['head_width']), head_depth=v.get('depth', cfg['head_depth']),
                  head_steps=v.get('steps', cfg['head_steps']), head_learning_rate=v.get('lr', cfg['head_learning_rate']),
                  head_batch_states=v.get('batch', cfg['head_batch_states']),
                  head_checkpoint_every=v.get('every', cfg['head_checkpoint_every']))
        t0 = time.perf_counter()
        h = T2.train_head(d['ytr'], d['gtr'], d['norm_t'], d['perp_t'], d['vy'], d['vgid'], d['vn2'], d['vperp'], d['Rs'],
                          v['k'], c2, print, code_reg=v.get('code_reg', 0.0))
        H = np.asarray(C.head(h['params'], h['codes']))
        tag = v['tag']
        T2.save(out / f'head_{tag}.pkl', dict(params=h['params'], codes=h['codes'], library_H=H, cfg=c2, info=h['info']))
        rep['heads'][tag] = dict(variant=v, info=h['info'], seconds=time.perf_counter() - t0)
        C.dump(out / 'heads.json', C.clean(rep))
    print('HEADS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
