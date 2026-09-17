"""Local smoke for b-eqtop: 64 intervals, q = 0, sub-minute on the GB10 apart from one
archive restore. Every gate is a correctness property of THIS lane's new code; the science
(whether rho falls with m, whether more fit states help) is measured by the jobs.

  1 `eqtop.rho_of_field` reproduces q-diag's archived rho for the cclad01 `q0_m256_eq_block`
    rule on that job's own fft_tight fields (36 points) to <= 1e-9 relative -- the parent
    lane's audited baseline, from the checksum-verified cclad01 archive in this worktree.
  2 `eqcert.certify` (coefficient path, JAX, five-point stencil of the bank) equals
    `rho_of_field` (field path, NumPy) on bank states to 1e-10 relative.
  3 `fitworker.bounded_nnls` is `varpro.bounded_nnls` bitwise on a real design.
  4 QR compression: the same fitter on (R, c) returns the same support and weights (to 1e-8)
    as on (D, b), the same relative fit (to 1e-10), and the same rho.
  5 'state' scaling: ||D w - b||^2 == sum_s rho(u_s)^2 on the fit states, to 1e-12.
  6 `run_chains` end to end through real worker subprocesses, with the adaptive stop.
"""
import hashlib
import json
import pickle
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
for sub in ('experiments/b-eqtop', 'experiments/q-ridge', 'experiments/b-ladder-top',
            'experiments/cheap-corrections', 'experiments/head-ablation',
            'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import varpro as VP            # noqa: E402
import eqcert as EC            # noqa: E402
import eqtop as ET             # noqa: E402
import fitworker as FW         # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
QDIAG = (ROOT.parent / '2026-09-16-q-diag/experiments/q-diag/checks/quadrature.json')
CCLAD = ROOT / 'experiments/cheap-corrections/artifacts/cclad01'


def restore_cclad(work):
    meta = json.loads((CCLAD / 'archive.json').read_text())
    tar = work / 'collection.tar.gz'
    h = hashlib.sha256()
    with tar.open('wb') as out:
        for ch in meta['chunks']:
            b = (CCLAD / ch['path']).read_bytes()
            assert hashlib.sha256(b).hexdigest() == ch['sha256']
            h.update(b)
            out.write(b)
    assert h.hexdigest() == meta['sha256']
    with tarfile.open(tar) as tf:
        names = [n for n in tf.getnames() if n == 'output/result.json'
                 or ('fft_tight' in n and n.endswith('rep0.npz'))]
        tf.extractall(work, members=[tf.getmember(n) for n in names])
    return work / 'output', meta['sha256']


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    t_all = time.perf_counter()
    out = {}

    # ---- gate 1: the parent's audited rho, bitwise --------------------------
    with tempfile.TemporaryDirectory() as td:
        odir, sha = restore_cclad(Path(td))
        r = json.loads((odir / 'result.json').read_text())
        s = next(s for s in r['arm_setup'] if s.get('arm') == 'q0_m256_eq_block')
        nodes, w = np.asarray(s['eq_indices']), np.asarray(s['eq_weights'])
        Phi256, _, _ = e.modes(256, s['M'])
        inv = {(x['name'], x['case']): x['artifact'] for x in r['invocations']}
        Q = [x for x in json.loads(QDIAG.read_text())
             if x['job'] == 'cclad01' and x['arm'] == 'q0_m256_eq_block'
             and x['evaluated_on'] == 'fft_tight']
        worst = 0.
        for x in Q:
            f = np.load(odir / inv[('fft_tight', x['case'])])['fields']
            u = np.ascontiguousarray(f[x['time_index']][1:-1, 1:-1]).ravel()
            rho = ET.rho_of_field(u, Phi256, 256, nodes, w)
            worst = max(worst, abs(rho - x['rho']) / x['rho'])
    out['qdiag_reproduction'] = dict(points=len(Q), worst_relative_difference=worst,
                                     cclad01_archive_sha256=sha)
    assert len(Q) == 36 and worst <= 1e-9, out['qdiag_reproduction']
    print('GATE 1', out['qdiag_reproduction'], flush=True)

    # ---- the 64-interval objects ---------------------------------------------
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Z = np.asarray(ck['Z_tr'])
    K, R = Z.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    L, M = 64, 128
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Phi, lam, _ = e.modes(L, M)
    head = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))
    rng = np.random.default_rng(20260917)
    coefs = np.asarray(head(jnp.asarray(Z[rng.choice(len(Z), 40, replace=False)])))
    held = np.asarray(head(jnp.asarray(Z[rng.choice(len(Z), 64, replace=False)])))
    cand = ET.candidate_pool(L, 2048, 7)

    # ---- gate 3: the worker's fitter is varpro's, bitwise ------------------
    D, b, dinfo = ET.build_design(G, Phi, L, coefs[:16], cand, scaling='row')
    s1, w1, i1 = VP.bounded_nnls(D, b, 256, 600., block=16)
    s2, w2, i2 = FW.bounded_nnls(D, b, 256, 600., block=16)
    out['worker_bitwise'] = dict(support_equal=bool(np.array_equal(s1, s2)),
                                 weights_equal=bool(np.array_equal(w1, w2)), m=int(len(s1)))
    assert out['worker_bitwise']['support_equal'] and out['worker_bitwise']['weights_equal']
    print('GATE 3', out['worker_bitwise'], flush=True)

    # ---- gate 2: coefficient path == field path ------------------------------
    pos = cand[s1]
    rule = ET.make_rule(bank, Phi, L, pos, w1)
    cert = EC.certify(G, Phi, L, rule, held, chunk=32)
    U = np.asarray(jnp.asarray(held) @ G.T)
    rf = np.array([ET.rho_of_field(u, Phi, L, pos, w1) for u in U])
    out['paths_agree'] = dict(rho_max_coef=cert['rho_max'], rho_max_field=float(rf.max()),
                              relative=abs(cert['rho_max'] - rf.max()) / rf.max())
    assert out['paths_agree']['relative'] < 1e-10, out['paths_agree']
    print('GATE 2', out['paths_agree'], flush=True)

    # ---- gate 4: compression equivalence (rows 40*128 = 5120 > 2048 columns) ---
    D2, b2, _ = ET.build_design(G, Phi, L, coefs, cand, scaling='row')
    Rc, cc, cinfo = ET.compress(D2, b2, where='gpu')
    sa, wa, ia = VP.bounded_nnls(D2, b2, 192, 600., block=12)
    sb, wb, ib = VP.bounded_nnls(Rc, cc, 192, 600., block=12)
    fa = float(np.linalg.norm(D2[:, sa] @ wa - b2) / np.linalg.norm(b2))
    fb = float(np.sqrt(np.linalg.norm(Rc[:, sb] @ wb - cc) ** 2 + cinfo['unreachable_residual'] ** 2)
               / cinfo['b_norm'])
    same = np.array_equal(np.sort(sa), np.sort(sb))
    wd = (float(np.max(np.abs(wa[np.argsort(sa)] - wb[np.argsort(sb)])) / np.max(np.abs(wa)))
          if same else None)
    ra = EC.certify(G, Phi, L, ET.make_rule(bank, Phi, L, cand[sa], wa), held)['rho_max']
    rb = EC.certify(G, Phi, L, ET.make_rule(bank, Phi, L, cand[sb], wb), held)['rho_max']
    out['compression'] = dict(cinfo, same_support=bool(same), weight_reldiff=wd,
                              fit_full=fa, fit_compressed=fb, fit_reldiff=abs(fa - fb) / fa,
                              rho_full=ra, rho_compressed=rb, rho_reldiff=abs(ra - rb) / ra,
                              jaccard=len(set(sa) & set(sb)) / len(set(sa) | set(sb)))
    assert same and wd < 1e-8 and out['compression']['fit_reldiff'] < 1e-10 \
        and out['compression']['rho_reldiff'] < 1e-10, out['compression']
    print('GATE 4', {k: v for k, v in out['compression'].items() if k != 'target_norms'}, flush=True)

    # ---- gate 5: the 'state' scaling objective is sum rho^2 -------------------
    D3, b3, d3 = ET.build_design(G, Phi, L, coefs[:16], cand, scaling='state')
    s3, w3, _ = VP.bounded_nnls(D3, b3, 256, 600., block=16)
    lhs = float(np.linalg.norm(D3[:, s3] @ w3 - b3) ** 2)
    U3 = np.asarray(jnp.asarray(coefs[:16]) @ G.T)
    rhs = float(sum(ET.rho_of_field(u, Phi, L, cand[s3], w3) ** 2 for u in U3))
    out['state_scaling_identity'] = dict(objective=lhs, sum_rho_squared=rhs,
                                         relative=abs(lhs - rhs) / rhs)
    assert out['state_scaling_identity']['relative'] < 1e-12, out['state_scaling_identity']
    print('GATE 5', out['state_scaling_identity'], flush=True)

    # ---- gate 6: chains through real subprocesses, adaptive stop --------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        np.save(tmp / 'D.npy', D)
        np.save(tmp / 'b.npy', b)
        side = tmp / 'side.json'
        side.write_text(json.dumps(dict(unreachable_residual=0., b_norm=float(np.linalg.norm(b)))))
        got = []
        bar = 1e9    # everything certifies: the adaptive chain must stop after 1 + confirm

        def certify_fn(chain, m, supp, wts, info):
            c = EC.certify(G, Phi, L, ET.make_rule(bank, Phi, L, cand[supp], wts), held)
            return dict(key=chain['key'], m=int(len(supp)), m_target=m, certification=c,
                        certified_primary=bool(c['rho_max'] <= bar), fit=info)

        chains = [dict(key='a', q=0, M=M, arm='std', grid=[64, 128, 192, 256], adaptive=True,
                       confirm=1, design=tmp / 'D.npy', b=tmp / 'b.npy', sidecar=side,
                       tmpdir=tmp, cand=cand, Phi=Phi),
                  dict(key='b', q=0, M=M, arm='std', grid=[64, 128], adaptive=False,
                       confirm=0, design=tmp / 'D.npy', b=tmp / 'b.npy', sidecar=side,
                       tmpdir=tmp, cand=cand, Phi=Phi)]
        res = ET.run_chains(chains, certify_fn, sys.executable, workers=2, threads=1,
                            seconds=600., blocks=16, on_rule=got.append, poll=1.)
    out['chains'] = dict(result=res, rules=[(g['key'], g['m_target'], g['fit']['relative_fit'])
                                            for g in got])
    fitted = {k: [g['m_target'] for g in got if g['key'] == k] for k in ('a', 'b')}
    assert fitted['a'] == [64, 128] and fitted['b'] == [64, 128], fitted
    assert res['a']['stop_reason'] == 'certified_and_confirmed' \
        and res['b']['stop_reason'] == 'grid_exhausted', res
    assert all(g['fit']['relative_fit'] < 1 for g in got)
    print('GATE 6', out['chains'], flush=True)

    out['seconds'] = time.perf_counter() - t_all
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('EQTOP SMOKE OK', round(out['seconds'], 1), 's', flush=True)


if __name__ == '__main__':
    main()
