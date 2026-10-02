"""Refined references for the quadrature study (DESIGN.md section 4).

For every case of the configured cohorts, the project's Newton--BiCGStab full-order solver (engines.make_fom, the
FOM the ROM is trained on, sign-upwind advection, backward Euler, FFT-DST Helmholtz preconditioner) is run at
L_ref = 8192 with
  ST  dt = 0.005 / 16 = 0.0003125  (refined in space AND time: hires-burgers' reference, the primary 'refined reference')
  S   dt = 0.005                   (refined in space only; isolates the spatial discretisation, secondary)
at ntol 1e-11, ltol 1e-9, accepted if every step's relative nonlinear residual <= 2e-11 (hires-burgers' rule).
Saved per case: the six output fields restricted to the 257 x 257 nodes shared by every evaluation mesh (stride 32),
and for the audit cases also at 1025 x 1025 (stride 8). Nothing else is computed here.

    python refjob.py --config <cfg.json> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import qcore as Q


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu'
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    Lr = int(cfg['mesh'])
    rep = dict(config=cfg, job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
               gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), cases=[])
    begin = time.perf_counter()
    foms = {tag: Q.e.make_fom(Lr, float(r['dt']))[0] for tag, r in cfg['refs'].items()}
    for coh in cfg['cohorts']:
        ph = Q.cohort(coh)
        rep.setdefault('cohort_sha256', {})[coh] = sha(ph)
        for c in range(min(len(ph), int(cfg.get('case_limit') or len(ph)))):
            u0 = jnp.asarray(Q.e.initial(Lr, ph[c]))
            for tag, r in cfg['refs'].items():
                t0 = time.perf_counter()
                f, it, rn = foms[tag](u0, float(ph[c, 4]), r['ntol'], r['ltol'])
                f = np.asarray(f)
                rn = np.asarray(rn)
                ok = bool(np.isfinite(f).all() and np.isfinite(rn).all() and rn.max() <= r['accept_residual'])
                r257 = np.array(f[:, ::Lr // 256, ::Lr // 256], copy=True)
                extra = {}
                if c in cfg.get('audit_cases', {}).get(coh, []) and Lr >= 1024:
                    extra['f1025'] = np.array(f[:, ::Lr // 1024, ::Lr // 1024], copy=True)
                np.savez_compressed(out / f'ref_{tag}_{coh}_{c:03d}.npz', f257=r257, **extra)
                rep['cases'].append(dict(cohort=coh, case=c, ref=tag, mesh=Lr, dt=r['dt'], accepted=ok,
                                         max_relative_residual=float(np.nanmax(rn)),
                                         newton_iterations=int(np.sum(np.asarray(it))),
                                         seconds=time.perf_counter() - t0, f257_sha256=sha(r257)))
                print('REF', coh, c, tag, 'accepted', ok, f'{float(np.nanmax(rn)):.2e}',
                      round(time.perf_counter() - t0, 1), round(time.perf_counter() - begin, 1), flush=True)
                del f
            json.dump(rep, open(out / 'result.json', 'w'), indent=1)
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = True
    json.dump(rep, open(out / 'result.json', 'w'), indent=1)
    print('REF COMPLETE', flush=True)


if __name__ == '__main__':
    main()
