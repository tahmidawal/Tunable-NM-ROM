"""Local smoke for the Poisson head ablation. Tiny mesh, under a minute per process.

Gate: the generic arm machinery with the frozen neural head must reach the same
solution as the incumbent `core.rom_query` from the same start, and every other
arm family must run and return a finite field.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
# Reverse order: multiresolution-poisson must win the `pilot` name over mr-burgers2d.
for rel in ('experiments/head-ablation', 'experiments/mr-burgers2d',
            'experiments/multiresolution-poisson'):
    sys.path.insert(0, str(ROOT / rel))

import core as C            # noqa: E402
import correction_core as CC  # noqa: E402
import pilot as P           # noqa: E402
import arms as A            # noqa: E402
import sep_common as sc     # noqa: E402
import poisson_ablation as PA  # noqa: E402

RUN = ROOT / 'experiments/multiresolution-poisson/runs/correction_accuracy10'
CK = RUN / 'checkpoints/r128_joint.pkl'
BASIS = RUN / 'basis.npz'


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    params, codes, cfgck = sc.load_pkl(CK)
    basis = np.load(BASIS)
    np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
    K = int(np.asarray(codes).shape[1])
    R = int(np.asarray(params['h_lin']).shape[1])
    n, modes, budget = 32, 256, 120   # modes must exceed R for the free-bank arm
    ops = C.assemble(params, np.asarray(codes), n, modes, budget)
    bank, B, S, I, J = ops['bank'], ops['B'], ops['S'], ops['I'], ops['J']
    M = int(B.shape[0])
    out = dict(K=K, R=R, intervals=n, retained_modes=M,
               retained_bank_rank=ops['info']['retained_bank_rank'])

    draws = C.source_params(7090703, 2)
    train = C.source_params(0, 48)
    lam = jnp.asarray(C.eigenvalues(n))
    U = np.stack([np.asarray(C.dst_solve(jnp.asarray(C.full_source(n, q)), lam))[1:-1, 1:-1].ravel()
                  for q in train])
    Ut = jnp.asarray(U.T)
    Qb, Rb = A.whiten(bank)
    coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    w, V = jnp.linalg.eigh(Ut.T @ Ut)
    w, V = w[::-1], V[:, ::-1]
    Vmodes = Ut @ (V[:, :16] / jnp.sqrt(jnp.clip(w[:16], 1e-300, None))[None, :])
    coords = np.asarray(Ut.T @ Vmodes)
    Hdec = sc.head(params, jnp.asarray(codes))
    lin, Zl, Wt, ilin = A.fit_linear_map(Hdec, Rb, K, seed=7)
    quad, iq = A.fit_quadratic_map(Hdec, Rb, K, Zl, Wt, seed=7)
    lint, Zlt, Wtt, ilt = A.fit_linear_map(coef_truth, Rb, K, seed=7)
    out['fits'] = dict(linear_decoder=ilin['relative_projection_rms'],
                       linear_truth=ilt['relative_projection_rms'],
                       quadratic_heldout=iq['heldout_relative'], quadratic_ridge=iq['ridge'])

    trials = [('a_neural', lambda z: sc.head(params, z), K, B, bank, np.asarray(codes), 'gj'),
              ('b_linear_dec', A.linear_head(lin), K, B, bank, Zl, 'gj'),
              ('b_linear_truth', A.linear_head(lint), K, B, bank, Zlt, 'gj'),
              ('c_quad_dec', A.quadratic_head(quad), K, B, bank, Zl, 'gj'),
              ('e_pod16', A.identity_head(), 16, PA.reduce_bank(Vmodes, S, I, J, n), Vmodes, coords, 'gj'),
              ('d_freebank', A.identity_head(), R, B, bank, np.asarray(Hdec), 'lu')]
    source = C.full_source(n, draws[0])
    fine = P.reference(draws[0], 1024)[::1024 // n, ::1024 // n]
    rows = []
    fields = {}
    for name, head, k, Bk, bk, cand, linear in trials:
        assert M > k, (name, M, k)
        t0 = time.perf_counter()
        trust = PA.radius(cand)
        pred = jax.jit(jax.vmap(lambda z: Bk @ head(z)))(jnp.asarray(cand))
        kern = PA.make_query(head, Bk, bk, n, trust, budget, 1e-6, linear)
        field, row = PA.query_once(kern, source, ops, pred, jnp.asarray(cand), Bk, bk)
        assert np.isfinite(field).all() and field.shape == (n + 1, n + 1)
        fields[name] = field
        rows.append(dict(arm=name, k=k, linear_solve=linear, seconds=time.perf_counter() - t0,
                         physical_error=C.relative(field, fine), iterations=row['iterations'],
                         reason=row['reason'], stationarity=row['stationarity'],
                         relative_residual=row['relative_residual']))
        print('ARM', name, 'ok', round(time.perf_counter() - t0, 2), flush=True)
    out['arms'] = rows

    incumbent, irow = C.rom_query(source, ops, 0.0)
    out['generic_vs_incumbent_neural'] = dict(
        relative=C.relative(fields['a_neural'], incumbent),
        incumbent_stationarity=irow['stationarity'], incumbent_reason=irow['reason'],
        tolerance=1e-06)
    print('GENERIC vs INCUMBENT', out['generic_vs_incumbent_neural']['relative'], flush=True)
    assert out['generic_vs_incumbent_neural']['relative'] <= 1e-6

    engine = CC.prepare_correction(ops, np.asarray(codes), basis['coefficient_directions'], 32,
                                   dict(online_preset=dict(budget=budget, tau=0.0, stationarity_stop=1e-6),
                                        linear_backward_error_limit=1e-12))
    cfg_r = dict(online_preset=dict(budget=budget, tau=0.0, stationarity_stop=1e-6),
                 stationarity_tolerance=1e-6, linear_backward_error_limit=1e-12,
                 residual_reconstruction_limit=1e-10)
    field, row = CC.correction_query(source, ops, engine, cfg_r)
    out['retained_q32'] = dict(physical_error=C.relative(field, fine), solver_valid=row['solver_valid'],
                               stationarity=row['stationarity'], correction_count=32)
    print('RETAINED q32', out['retained_q32'], flush=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
    print('POISSON SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
