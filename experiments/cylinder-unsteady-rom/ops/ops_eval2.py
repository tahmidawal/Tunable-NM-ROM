"""Second rebuttal push, PROTOCOL.md A9b: the frozen operator checkpoints on the extra cases (no training).

Adapted copy of the operator lane's final_eval.py: same Surrogate, same Scorer, same read-outs and timing; the only
changes are the case list (--case name:Re:dir, any number of saves) and that the geometry is the operator lane's own
geom.npz (--geom).  Checkpoints are checked against the hashes given on the command line (label=dir=sha256).
Writes OPS-<tag>.json and series_<label>_<case>.npz (t, e_u, e_p, CD, CL, CD_ref, CL_ref).
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
    ap.add_argument('--geom', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--case', action='append', required=True, help='name:Re:data_dir')
    ap.add_argument('--arm', action='append', required=True, help='label=train_dir=sha256 of final.pt')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    geom = oc.Geometry(Path(a.geom).parent, dev)
    cases = []
    for s in a.case:
        name, Re, d = s.split(':', 2); cases.append((name, float(Re), Path(d)))
    res = {'gpu': torch.cuda.get_device_name(0) if dev != 'cpu' else 'cpu', 'geom_sha256': hashlib.sha256(Path(a.geom).read_bytes()).hexdigest(),
           'arms': {}, 'reference_readout': {}}
    scorers = {}
    for name, Re, d in cases:
        sc = oc.Scorer(geom, d, name, Re); scorers[name] = sc
        rr, cd, cl = sc.reference_readout()
        fl = {}
        for rep, mp in (('n2', geom.node_map(2)), ('grid', geom.grid_map())):
            arr = np.load(d / f'{name}_{rep}.npy', mmap_mode='r')
            r, _ = sc.score((torch.as_tensor(np.asarray(arr[i]), device=dev) for i in range(len(arr))), mp)
            fl[rep] = {k: r.get(k) for k in ('e_u', 'e_p', 'cycle_errors', 'phase', 'targets_met')}
        rr['representation_floors'] = fl
        res['reference_readout'][name] = rr
        print('reference', name, json.dumps({'saves': len(sc.t), 'errors': rr['errors'], 'floors': fl}, default=float), flush=True)
    for spec in a.arm:
        label, path, want = spec.split('=', 2)
        ck_path = Path(path) / 'final.pt'
        got = hashlib.sha256(ck_path.read_bytes()).hexdigest()
        if got != want:
            raise ValueError(f'checkpoint hash mismatch for {label}')
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        arm, meta = ck['arm'], ck['meta']
        static = make_static(arm, geom, meta['k'], meta)
        sur = om.Surrogate(arm, ck['cfg'], meta, ck['stats'], static, dev)
        sur.model.load_state_dict(ck['state'])
        rec = {'arm': arm, 'config': ck['cfg'], 'params': om.n_params(sur.model), 'rep': meta['rep'], 'checkpoint_sha256': got, 'cases': {}}
        to_Q = geom.grid_map() if arm == 'grid' else geom.node_map(meta['k'])
        for name, Re, d in cases:
            try:
                arr = np.load(d / f"{name}_{meta['rep']}.npy", mmap_mode='r')
                x0 = torch.from_numpy(np.array(arr[0]))
                n = len(scorers[name].t)
                pred = sur.window(x0, Re, n)
                r, series = scorers[name].score((pred[i] for i in range(n)), to_Q)
                r['inference'] = time_window(sur, x0, Re, n)
                np.savez(out / f'series_{label}_{name}.npz', **series)
                del pred
            except Exception as exc:  # retained failure, never dropped
                r = {'status': 'failed', 'error': repr(exc)}
            rec['cases'][name] = r
            print(label, name, json.dumps({k: r.get(k) for k in ('status', 'e_u', 'e_p', 'cycle_errors', 'phase', 'targets_met', 'error')}, default=float), flush=True)
            torch.cuda.empty_cache() if dev != 'cpu' else None
        res['arms'][label] = rec
        (out / f'OPS-{a.tag}.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
        del sur
    print('ops_eval_done')


if __name__ == '__main__':
    main()
