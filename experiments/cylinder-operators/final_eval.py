"""Final evaluation of the frozen arms (PROTOCOL.md): development Re 95 again and the sealed held-out Re 105,
both scored on Q, with inference timing of every arm on the same GPU. The only script that reads the sealed dir.

Usage: final_eval.py --data <prep data> --sealed <sealed dir> --out <dir> --arm name=<train out dir> [...]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import ops_common as oc
import ops_models as om
from ops_train import make_static, time_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--sealed', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--no-heldout', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data_dir = Path(a.data); sealed = Path(a.sealed)
    geom = oc.Geometry(data_dir, dev)
    cases = [('re95', 95, data_dir)] + ([] if a.no_heldout else [('re105', 105, sealed)])
    res = {'gpu': torch.cuda.get_device_name(0) if dev != 'cpu' else 'cpu', 'arms': {}, 'reference_readout': {}}
    scorers = {}
    for name, Re, d in cases:
        sc = oc.Scorer(geom, d, name, Re); scorers[name] = sc
        rr, cd, cl = sc.reference_readout()
        res['reference_readout'][name] = rr
        np.savez(out / f'reference_forces_{name}.npz', t=sc.t, CD=cd, CL=cl, hist=sc.hist)
        if d == sealed:
            fl = {}
            for rep, mp in (('n2', geom.node_map(2)), ('n3', geom.node_map(3)), ('grid', geom.grid_map())):
                arr = np.load(d / f'{name}_{rep}.npy', mmap_mode='r')
                r, _ = sc.score((torch.as_tensor(np.asarray(arr[i]), device=dev) for i in range(len(arr))), mp)
                fl[rep] = r
            res['reference_readout'][name]['representation_floors'] = fl
    for spec in a.arm:
        label, path = spec.split('=', 1)
        ck_path = Path(path) / 'final.pt'
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        arm, meta = ck['arm'], ck['meta']
        static = make_static(arm, geom, meta['k'], meta)
        sur = om.Surrogate(arm, ck['cfg'], meta, ck['stats'], static, dev)
        sur.model.load_state_dict(ck['state'])
        rec = {'arm': arm, 'config': ck['cfg'], 'params': om.n_params(sur.model), 'rep': meta['rep'],
               'checkpoint_sha256': hashlib.sha256(ck_path.read_bytes()).hexdigest(), 'cases': {}}
        to_Q = geom.grid_map() if arm == 'grid' else geom.node_map(meta['k'])
        for name, Re, d in cases:
            arr = np.load(d / f"{name}_{meta['rep']}.npy", mmap_mode='r')
            x0 = torch.from_numpy(np.array(arr[0]))
            n = len(scorers[name].t)
            pred = sur.window(x0, Re, n)
            r, series = scorers[name].score((pred[i] for i in range(n)), to_Q)
            r['inference'] = time_window(sur, x0, Re, n)
            rec['cases'][name] = r
            np.savez(out / f'series_{label}_{name}.npz', **series)
            # a few native snapshots for figures (grid arms only keep them small)
            if arm == 'grid':
                np.savez_compressed(out / f'fields_{label}_{name}.npz', pred=pred[[0, n // 2, n - 1]].cpu().numpy(),
                                    ref=np.array(arr[[0, n // 2, n - 1]]))
            print(label, name, json.dumps({k: r.get(k) for k in ('status', 'e_u', 'e_p', 'cycle_errors', 'phase', 'targets_met')}, default=float), flush=True)
            del pred
        res['arms'][label] = rec
        (out / 'final_eval.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
        del sur
    print('final_eval_done')


if __name__ == '__main__':
    main()
