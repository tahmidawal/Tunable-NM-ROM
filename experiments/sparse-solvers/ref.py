"""Reference solutions for the Burgers stepper (amendment 3): the paper's own reference full-order solver, unchanged
vendored code, on the target mesh.  Writes refs/b<dim>d_L<L>_case<c>.npy (6 fields, full nodal mesh incl. the zero
boundary, field 0 = the initial condition) and refs/b<dim>d_L<L>.json (viscosities, cohort checksum, solver record).

  python ref.py --dim 2 --L 256 --cases 64 --out refs
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST64_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--cases', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L, K = a.L, a.cases
    sys.path.insert(0, str(HERE / 'vendor' / f'b{a.dim}d'))
    import jax
    import jax.numpy as jnp
    assert jax.default_backend() == 'gpu'
    rec = dict(dim=a.dim, L=L, cases=K, jax=jax.__version__, dt=0.005, steps=50, case_records=[])
    if a.dim == 2:
        import engines as e
        import iterative_paths as ip
        physical = e.params_draw(20260916, 64)
        sha = hashlib.sha256(np.ascontiguousarray(physical).tobytes()).hexdigest()
        rec.update(cohort='params_draw(20260916, 64) (paper test64)', cohort_sha256=sha, cohort_sha256_expected=TEST64_SHA,
                   reference="iterative_paths.make_fom(L, 0.005, 'fft'), ntol 1e-6, ltol 1e-8", ntol=1e-6, ltol=1e-8)
        rec['cohort_sha_matches_paper'] = bool(sha == TEST64_SHA)
        fn, pre = ip.make_fom(L, .005, 'fft')
        nus = [float(physical[c, 4]) for c in range(K)]
        for c in range(K):
            t = time.perf_counter()
            u0 = e.initial(L, physical[c])
            v = fn(jnp.asarray(u0), nus[c], 1e-6, 1e-8, *pre)
            f = np.asarray(v[0])
            it, rn = np.asarray(v[1]), np.asarray(v[2])
            ok = bool(np.isfinite(f).all() and rn.max() <= 1e-6 * (1 + 1e-9))
            np.save(out / f'b2d_L{L}_case{c}.npy', f.reshape(6, -1))
            rec['case_records'].append(dict(case=c, nu=nus[c], params=[float(x) for x in physical[c]], converged=ok,
                                            max_rel_residual=float(rn.max()), newton_total=int(it.sum()),
                                            seconds=time.perf_counter() - t))
            del v
    else:
        import common as C
        n = L + 1
        tab = C.table(923901, 32)
        rec.update(cohort='table(923901, 32) (paper Burgers 3D held-out cohort)', cohort_sha256=tab['sha256'],
                   reference='common.make_fom(n, 0.005, 1e-10, 1e-11)', ntol=1e-10, ltol=1e-11, n=n)
        fn = C.make_fom(n, C.DT, 1e-10, 1e-11)
        for c in range(K):
            t = time.perf_counter()
            u0 = C.initial_interior(n, tab, c)
            f, it, rn = fn(jnp.asarray(u0), float(tab['nu'][c]))
            f = np.asarray(f).reshape(6, n - 2, n - 2, n - 2)
            rn = np.asarray(rn)
            ok = bool(np.isfinite(f).all() and rn.max() < 1e-9)
            np.save(out / f'b3d_L{L}_case{c}.npy', np.pad(f, ((0, 0), (1, 1), (1, 1), (1, 1))).reshape(6, -1))
            rec['case_records'].append(dict(case=c, nu=float(tab['nu'][c]), converged=ok, max_rel_residual=float(rn.max()),
                                            newton_total=int(np.asarray(it).sum()), seconds=time.perf_counter() - t))
    rec['all_converged'] = all(r['converged'] for r in rec['case_records'])
    (out / f'b{a.dim}d_L{L}.json').write_text(json.dumps(rec, indent=1))
    print(f'ref dim={a.dim} L={L} cases={K} all_converged={rec["all_converged"]} '
          f'seconds={sum(r["seconds"] for r in rec["case_records"]):.1f}', flush=True)


if __name__ == '__main__':
    main()
