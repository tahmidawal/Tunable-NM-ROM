"""b-panel: the paper's central comparison in ONE allocation on ONE GPU, per mesh.

One job, one GPU, one randomised order, three timed repetitions with burn-in, for the frozen
Burgers model (bank G, head h_theta, transferred directions C):

  * the correction ladder q in {0,16,32,64,128,256}, dense quadrature, M = 4(K+q), budget 600;
  * the same ladder through the CERTIFIED empirical-quadrature rules of job qrg304 (loaded,
    not refitted), at evolution tolerances 1e-6 and 1e-3;
  * POD-LSPG at k' in {16..512} through the same weak objective, budget 600;
  * the unrestricted-bank endpoint (R = 512 free coefficients);
  * b-speed's optimised q = 0 kernel, admitted only on an in-job parity gate;
  * a grid of same-grid full-order Newton controls plus the converged `fft_tight` reference
    every same-grid error is measured against, and its dense-preconditioner twin;
  * the model-facing cohort for the trained FNO, timed by `fno_panel.py` in a second process
    of this same allocation after this process exits.

At a mesh other than 256 the certified rules are TRANSFERRED (support mapped to the same
physical points, weights refit on reachable states, certified by held-out rho) and labelled
`eqxfer`. Every subject is built and warmed up in a declared priority order; a subject that
fails with RESOURCE_EXHAUSTED is recorded as dropped and the job continues.

Everything is written incrementally to `result.json` so a truncated job is collectable. See
`DESIGN.md` for the contract, the gates, the convergence rule and the non-dominance rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import ladder as LD
import varpro as VP
import topfix as TF
import eqcert as EC
from ablation import pod_basis, radius
import fast as F
import ladders as FL

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump
TIMES = np.array([0., .05, .1, .15, .2, .25], dtype=np.float64)
IC_RESIDUAL_FLOOR = 1e-10      # DESIGN.md section 5 (iii)


def is_oom(exc):
    s = str(exc)
    return 'RESOURCE_EXHAUSTED' in s or 'out of memory' in s.lower() or 'OOM' in s


def unravel_nodes(nodes, L):
    return np.stack(np.unravel_index(np.asarray(nodes), (L - 1, L - 1)), 1) + 1


def rule_operators(bank, Phi, L, nodes, weights):
    """G5 and Pq from a (nodes, weights) rule, exactly as `eqcert.fit_rule` forms them."""
    pos = np.asarray(nodes, dtype=int)
    ij = unravel_nodes(pos, L)
    return dict(G5=bank.stencil(ij, L), Pq=jnp.asarray(np.asarray(Phi)[pos] * np.asarray(weights)[:, None]))


def transfer_support(nodes_src, L_src, L):
    """The same physical points on a finer grid: interior index ij -> (L/L_src) ij."""
    assert L % L_src == 0
    f = L // L_src
    ij = unravel_nodes(np.asarray(nodes_src, dtype=int), L_src) * f
    pos = np.ravel_multi_index(((ij[:, 0] - 1), (ij[:, 1] - 1)), (L - 1, L - 1))
    return np.asarray(pos, dtype=int), ij


def refit_weights(G, P, Phi_host, L, pos, coefficients, M):
    """Nonnegative least squares on a FIXED support, design built as `eqcert.fit_rule` builds it.

    `P` is the device test matrix already resident for the dense arm; `Phi_host` its NumPy copy.
    """
    t0 = time.perf_counter()
    Pc = np.asarray(Phi_host)[pos]
    adv = jax.jit(lambda g, c: e.spatial(g @ c, L)[0])
    rows, targets = [], []
    for c in np.asarray(coefficients):
        n = adv(G, jnp.asarray(c))
        targets.append(np.asarray(P.T @ n))
        rows.append(Pc.T * np.asarray(n)[pos])
    design = np.concatenate(rows)
    b = np.concatenate(targets)
    scale = np.linalg.norm(design, axis=1) + 1e-300
    design /= scale[:, None]
    b /= scale
    w, _ = scipy.optimize.nnls(design, b, maxiter=20 * design.shape[1])
    rel = float(np.linalg.norm(design @ w - b) / max(np.linalg.norm(b), 1e-300))
    return w, dict(relative_fit=rel, design_rows=int(design.shape[0]), support=int(len(pos)),
                   nonzero_weights=int(np.sum(w > 0)), fit_states=int(len(coefficients)),
                   M=int(M), seconds=time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True, help='experiments/b-panel/inputs')
    p.add_argument('--out', required=True)
    p.add_argument('--fno-train-index', default=None)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = Path(a.inputs)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L = int(cfg['intervals'])
    dt = cfg['dt']
    strict = cfg['strict']
    prov = json.loads((inputs / 'PROVENANCE.json').read_text())['files']

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
                  mem_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                  checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L, dt=dt,
                  spatial_bank_frozen=True, network_weights_frozen=True, final_cohort_unopened=True,
                  output_times=TIMES.tolist(), inputs_provenance=prov,
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output fields '
                                   '(gpu_seconds); the same invocation including the host upload of the '
                                   'input and the host copy of the six outputs (host_seconds); identical '
                                   'for every JAX subject; the FNO is timed in a second process of this '
                                   'same allocation by fno_panel.py'),
                  reference=[], snapshots={}, directions={}, rules=[], transfer=[], arm_setup=[],
                  reconstruction=[], invocations=[], declared_subjects=[], dropped=[], gates={},
                  fno=dict(prepared=False), verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    # ---------------------------------------------------------------- cohort --
    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['physical_sha256'] = sha_array(physical)
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    want = cfg.get('expected_physical_sha256')
    report['gates']['evaluation_cohort_bitwise_abl01'] = dict(
        expected=want, got=report['physical_sha256'], passed=(None if want is None else report['physical_sha256'] == want))
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)
    save()

    # ------------------------------------------------------------ references --
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    q_fom, _ = e.make_fom(rf, rt)
    for case, phys in enumerate(physical):
        t = time.perf_counter()
        f, it, rn = host(q_fom(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
        assert np.isfinite(f).all() and np.max(rn) < 2e-11
        f = np.array(f[:, ::rf // L, ::rf // L], copy=True)
        refs[case] = f
        name = f'ref_L{rf}_dt{rt}_case{case}.npz'
        np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
        got = sha_array(f)
        w = (cfg.get('expected_reference_sha256') or {}).get(str(case))
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=got, comparator_field_sha256=w,
                                        bitwise_matches_comparator=(None if w is None else got == w),
                                        seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q_fom
    jax.clear_caches()

    # ------------------------------------------------- the model-facing cohort --
    cohort = out / 'fno-cohort'
    cohort.mkdir(exist_ok=True)
    records = []
    for case, phys in enumerate(physical):
        target = refs[case][:, None].astype(np.float64)
        supplied = np.ascontiguousarray(target[0])
        path = cohort / f'burgers-dev-{case:05d}.npz'
        np.savez(path, input=supplied, target=target,
                 parameters=np.array([float(phys[4])], dtype=np.float64), times=TIMES)
        records.append(dict(case_id=f'burgers-dev-{case:05d}', split='development', case_index=case,
                            seed=int(cfg['eval_seed'] if case < cfg['eval_cases'] else cfg['eval_fresh_seed']),
                            path=path.name, mesh=L, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            generation_descriptors=dict(zip(('cx', 'cy', 'width', 'amplitude', 'nu'),
                                                            map(float, phys)))))
    (cohort / 'index.json').write_text(json.dumps(dict(
        schema_version=1, pde='burgers', split='development', count=len(records), mesh=L, complete=True,
        descriptors_are_model_inputs=False,
        derivation=(f'the same six opened development cases and the same {rf}-interval reference '
                    f'restricted to the {L}-interval grid that every subject in this job is graded against'),
        records=records), indent=2) + '\n')
    report['fno'] = dict(prepared=True, cohort_index=str(cohort / 'index.json'), cases=len(records))
    if a.fno_train_index and Path(a.fno_train_index).exists():
        tr = json.loads(Path(a.fno_train_index).read_text())
        desc = np.array([[r['generation_descriptors'][k] for k in ('cx', 'cy', 'width', 'amplitude', 'nu')]
                         for r in tr['records']])
        dist = np.min(np.linalg.norm(desc[None] - physical[:, None], axis=2), axis=1)
        report['gates']['fno_cohort_disjoint_from_training'] = dict(
            passed=bool(np.min(dist) > 1e-8), training_cases=int(len(desc)),
            nearest_descriptor_distance=dist.tolist())
    save()

    # ------------------------------------------------------ bank and snapshots --
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    jax.block_until_ready(Rb)
    ranks = sorted(set(cfg.get('pod_ranks', [])))
    Vmodes, coords_full = None, None
    if ranks:
        U, sinfo = LD.generate_snapshots(L, dt, train_physical, cfg['train_state_stride'],
                                         cfg['snapshot_ntol'], cfg['snapshot_ltol'])
        Ut = jnp.asarray(U.T)
        del U
        sinfo['bank_projection_relative_rms'] = float(jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
        Vmodes, eigen, energy = pod_basis(Ut, max(ranks))
        coords_full = np.asarray(Ut.T @ Vmodes)
        sinfo.update(pod_total_energy=energy, pod_eigenvalues=np.asarray(eigen).tolist(),
                     pod_tail_fraction={str(k): float(max(energy - float(np.sum(eigen[:k])), 0.) / max(energy, 1e-300))
                                        for k in ranks})
        del Ut
        jax.clear_caches()
        report['snapshots'][str(L)] = sinfo
        print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)
    report['bank'] = dict(shape=list(G.shape), whitening_diag_min=float(jnp.min(jnp.abs(jnp.diag(Rb)))))
    save()

    # ------------------------------------------------------------ directions --
    dfile = inputs / cfg['directions_file']
    dprov = prov[cfg['directions_file']]
    got = sha_file(dfile)
    dz = np.load(dfile)
    Cnp = np.ascontiguousarray(np.asarray(dz['C']))
    QS_ALL = sorted(set(cfg['dense_q']) | set(cfg['eq_q']) | set(cfg.get('extra_dense_q', [])))
    prefix = {str(q): sha_array(np.ascontiguousarray(Cnp[:, :q])) for q in QS_ALL}
    ok_prefix = all(prefix[k] == dprov['prefix_sha256'].get(k) for k in prefix)
    report['directions'] = dict(file=cfg['directions_file'], sha256=got, expected_sha256=dprov['sha256'],
                                columns=int(Cnp.shape[1]), prefix_sha256=prefix,
                                expected_prefix_sha256=dprov['prefix_sha256'], source_job=dprov['source_job'],
                                rule=dprov['rule'], transferred=True)
    report['gates']['directions_file_sha256'] = dict(passed=(got == dprov['sha256']), got=got, expected=dprov['sha256'])
    report['gates']['directions_prefix_hashes'] = dict(passed=bool(ok_prefix), detail=prefix)
    assert got == dprov['sha256'] and ok_prefix, 'transferred directions do not match PROVENANCE.json'
    Cfull = jnp.asarray(Cnp)
    del dz
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    save()

    # ----------------------------------------------- shared operator pieces --
    Phi_cache, lam_cache = {}, {}

    def modes(M):
        if M not in Phi_cache:
            ph, lm, _ = e.modes(L, M)
            Phi_cache[M], lam_cache[M] = ph, jnp.asarray(lm)
        return Phi_cache[M], lam_cache[M]

    dense_cache = {}

    def dense_data(M):
        if M not in dense_cache:
            ph, lm = modes(M)
            P = jnp.asarray(ph)
            dense_cache[M] = dict(A=P.T @ G, lam=lm, G=G, Phi=P)
            jax.block_until_ready(dense_cache[M]['A'])
        return dense_cache[M]

    colds, heads = {}, {}

    def cold_for(q):
        if q not in colds:
            C = Cfull[:, :q]
            heads[q] = TF.corrected_head(params, C, K)
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[q] = A.build_cold(bank, heads[q], Zaug, cfg['cold_axis_points'])
        return colds[q][0], colds[q][1], heads[q]

    lin = lambda dim: 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'

    # ---------------------------------------------------------- the rules ----
    rules = {}          # q -> (ops dict with G5/Pq, info)
    xfer = cfg.get('eq_transfer')
    for q in cfg['eq_q']:
        spec = cfg['rules'][str(q)]
        rfile = inputs / 'rules' / spec['file']
        rprov = prov[f"rules/{spec['file']}"]
        got = sha_file(rfile)
        assert got == rprov['sha256'], (spec['file'], got, rprov['sha256'])
        z = np.load(rfile)
        nodes, weights = np.asarray(z['nodes'], dtype=int), np.asarray(z['weights'], dtype=float)
        M = 4 * (K + q)
        assert rprov['M'] == M and rprov['q'] == q
        info = dict(q=q, M=M, file=spec['file'], sha256=got, source_job=rprov['source_job'],
                    source_m=int(len(nodes)), source_mesh=int(cfg['rules_mesh']),
                    source_rho_max=rprov['rho_max'], source_rho_p95=rprov['rho_p95'],
                    source_relative_fit=rprov['relative_fit'], rho_bar=rprov['rho_bar'],
                    certified_primary=rprov['certified_primary'], certified_secondary=rprov['certified_secondary'],
                    basis=('primary' if rprov['certified_primary'] else
                           'secondary' if rprov['certified_secondary'] else 'none'),
                    nodes_sha256=sha_array(nodes), weights_sha256=sha_array(weights),
                    nodes_sha256_matches=(sha_array(nodes) == rprov['nodes_sha256']),
                    weights_sha256_matches=(sha_array(weights) == rprov['weights_sha256']))
        assert info['nodes_sha256_matches'] and info['weights_sha256_matches'], spec['file']
        if L == int(cfg['rules_mesh']):
            ph, _ = modes(M)
            ops = rule_operators(bank, ph, L, nodes, weights)
            info.update(kind='eqcert', m=int(len(nodes)), transferred=False)
            rules[q] = (ops, info)
        else:
            assert xfer is not None, 'a rule at another mesh needs an eq_transfer block'
            rules[q] = (None, dict(info, kind='eqxfer', transferred=True, src_nodes=nodes, src_weights=weights))
        report['rules'].append({k: v for k, v in info.items() if not k.startswith('src_')})
        print('RULE q', q, info['kind'] if 'kind' in info else '', 'm', len(nodes), info['basis'], flush=True)
    save()

    # ------------------------------------------ transferred rules (job 2/3) --
    if xfer is not None and rules:
        rng = np.random.default_rng(xfer['seed'])
        fit_idx = np.asarray(xfer['fit_trajectories'], dtype=int)
        cert_idx = np.asarray(xfer['cert_trajectories'], dtype=int)
        assert not set(fit_idx.tolist()) & set(cert_idx.tolist())
        report['gates']['transfer_fit_cert_disjoint'] = dict(passed=True, fit=fit_idx.tolist(), cert=cert_idx.tolist())
        for q in cfg['eq_q']:
            ops, info = rules[q]
            M = 4 * (K + q)
            C = Cfull[:, :q]
            cold, _, head = cold_for(q)
            dd = dense_data(M)
            ph, _ = modes(M)
            t0 = time.perf_counter()
            collect = EC.make_collect_query(params, C, K, q, L, dt, trust, iters=xfer['collect_iters'],
                                            ic_budget=strict['ic_budget'], gtol=strict['gtol'],
                                            linear=lin(K + q), inner_damping=cfg['inner_damping'])
            cf = jax.jit(jax.vmap(head))
            pools = {}
            for tag, idx in (('fit', fit_idx), ('cert', cert_idx)):
                rows = []
                for i in idx:
                    phys = train_physical[i]
                    seen, _ = collect(jnp.asarray(e.initial(L, phys)), float(phys[4]), dd, cold)
                    wv = np.asarray(seen).reshape(-1, K + q)
                    rows.append(np.asarray(cf(jnp.asarray(wv))))
                pools[tag] = np.concatenate(rows)
            del collect
            jax.clear_caches()
            nfit = int(np.clip(xfer['max_fit_rows'] // M, 8, xfer['fit_states']))
            sel = np.sort(rng.choice(len(pools['fit']), min(nfit, len(pools['fit'])), replace=False))
            csel = pools['cert'][np.sort(rng.choice(len(pools['cert']), min(xfer['cert_states'], len(pools['cert'])),
                                                    replace=False))]
            pos, ij = transfer_support(info['src_nodes'], cfg['rules_mesh'], L)
            w, finfo = refit_weights(G, dd['Phi'], ph, L, pos, pools['fit'][sel], M)
            keep = w > 0
            ops = dict(G5=bank.stencil(ij[keep], L), Pq=jnp.asarray(np.asarray(ph)[pos[keep]] * w[keep][:, None]))
            cert = EC.certify(G, dd['Phi'], L, ops, csel, chunk=xfer['certify_chunk'])
            bar = float(info['rho_bar'])
            tinfo = dict(q=q, M=M, m_support=int(len(pos)), m=int(keep.sum()), refit=finfo, certification=cert,
                         fit_states_available=int(len(pools['fit'])), fit_states_used=int(len(sel)),
                         certification_states=int(len(csel)), collect_iters=xfer['collect_iters'],
                         fit_pool_sha256=sha_array(pools['fit']), certification_pool_sha256=sha_array(pools['cert']),
                         certified_primary=bool(cert['rho_max'] <= bar), certified_secondary=bool(cert['rho_p95'] <= bar),
                         rho_bar=bar, source_rho_max=info['source_rho_max'], seconds=time.perf_counter() - t0)
            tinfo['basis'] = ('primary' if tinfo['certified_primary'] else 'secondary' if tinfo['certified_secondary'] else 'none')
            info = dict({k: v for k, v in info.items() if not k.startswith('src_')}, m=tinfo['m'],
                        basis=tinfo['basis'], certified_primary=tinfo['certified_primary'],
                        certified_secondary=tinfo['certified_secondary'], rho_max=cert['rho_max'], rho_p95=cert['rho_p95'])
            rules[q] = (ops, info)
            np.savez_compressed(out / f'rule_xfer_q{q}_L{L}.npz', nodes=pos[keep], weights=w[keep])
            report['transfer'].append(tinfo)
            print('XFER q', q, 'm', tinfo['m'], 'rho_max', f"{cert['rho_max']:.4f}", 'p95', f"{cert['rho_p95']:.4f}",
                  tinfo['basis'], round(tinfo['seconds'], 1), flush=True)
            save()

    # ---------------------------------------------------------- subjects -----
    def gt(g):
        return f'g{g:g}'.replace('-', 'm').replace('.', 'p')

    specs = []
    for fs in cfg['fom_settings']:
        specs.append(dict(name=fs['name'], family='fom', setting=fs, priority=0))
    for q in cfg['dense_q']:
        specs.append(dict(name=f'q{q}_M{4 * (K + q)}_dense_{gt(strict["gtol"])}', family='rom', q=q,
                          M=4 * (K + q), quadrature='dense', gtol=strict['gtol'], priority=1))
    for g in cfg['eq_gtols']:
        pr = 2 if g == strict['gtol'] else 4
        for q in cfg['eq_q']:
            kind = rules[q][1]['kind']
            specs.append(dict(name=f'q{q}_M{4 * (K + q)}_{kind}_{gt(g)}', family='rom', q=q, M=4 * (K + q),
                              quadrature='eq', gtol=g, priority=pr, rule_kind=kind))
    for k in ranks:
        specs.append(dict(name=f'pod{k}_M{4 * k}_dense', family='pod', k=k, M=4 * k, quadrature='dense',
                          gtol=strict['gtol'], priority=3))
    for q in cfg.get('extra_dense_q', []):
        M = int(cfg['extra_dense_M'])
        specs.append(dict(name=f'q{q}_M{M}_dense_{gt(strict["gtol"])}', family='rom', q=q, M=M,
                          quadrature='dense', gtol=strict['gtol'], priority=5))
    if cfg.get('free_bank'):
        specs.append(dict(name=f'free{R}_M{cfg["free_bank_M"]}_dense', family='free', k=R, M=int(cfg['free_bank_M']),
                          quadrature='dense', gtol=strict['gtol'], priority=6))
    if cfg.get('fast_arm') and 0 in cfg['eq_q']:
        kind = rules[0][1]['kind']
        specs.append(dict(name=f'q0_M{4 * K}_{kind}_{gt(strict["gtol"])}_fast{cfg["fast_arm"]}', family='fast', q=0,
                          M=4 * K, quadrature='eq', gtol=strict['gtol'], priority=7, rule_kind=kind,
                          parity_against=f'q0_M{4 * K}_{kind}_{gt(strict["gtol"])}'))
    order = {s['name']: i for i, s in enumerate(cfg.get('priority_override', []))}
    # A name in `priority_override` is built right after the FOM controls, in the listed order,
    # whatever its family: DESIGN.md section 3.2's reduced set comes before everything else.
    specs.sort(key=lambda s: ((0.5, order[s['name']]) if s['name'] in order else (s['priority'], 0)))
    report['declared_subjects'] = [{k: v for k, v in s.items() if k != 'setting'} | (s.get('setting') or {}) for s in specs]
    save()

    built, foms = {}, {}
    Hdec = None

    def build(s):
        nonlocal Hdec
        t0 = time.perf_counter()
        fam = s['family']
        if fam == 'fom':
            key = (s['setting']['preconditioner'], s['setting']['dt'])
            if key not in foms:
                foms[key] = ip.make_fom(L, s['setting']['dt'], s['setting']['preconditioner'])
            return dict(s, kind='fom', key=key, setup=dict(arm=s['name'], family='fom', **s['setting']))
        if fam in ('rom', 'fast'):
            q, M = s['q'], s['M']
            assert M > K + q
            C = Cfull[:, :q]
            cold, cinfo, head = cold_for(q)
            if s['quadrature'] == 'dense':
                data, rinfo = dense_data(M), {}
            else:
                ops, rinfo = rules[q]
                ph, lm = modes(M)
                P = jnp.asarray(ph)
                data = dict(A=P.T @ G, lam=lm, G=G, G5=ops['G5'], Pq=ops['Pq'])
            if fam == 'rom':
                query = TF.make_query(params, C, K, q, L, dt, trust, s['quadrature'], 'base', Rb=Rb,
                                      ic_budget=strict['ic_budget'], step_budget=strict['step_budget'],
                                      gtol=s['gtol'], ic_gtol=cfg['ic_gtol'], linear=lin(K + q),
                                      inner_damping=cfg['inner_damping'], tau_y=cfg['tau_y'])
                extra = {}
            else:
                o = FL.ARMS[cfg['fast_arm']]
                tab = F.build_tables(params, data, cold, o)
                fq = F.make_query(params, K, L, dt, int(data['G5'].shape[0]), trust, o,
                                  ic_budget=strict['ic_budget'], step_budget=strict['step_budget'], gtol=s['gtol'])
                query = (lambda u0, nu, d, c, _tab=tab, _fq=fq: _fq(u0, nu, d, c, _tab))
                extra = dict(kernel_options=o, kernel_class=FL.ALL[cfg['fast_arm']]['cls'],
                             kernel_note=FL.ALL[cfg['fast_arm']]['note'], parity_against=s['parity_against'])
            setup = dict(arm=s['name'], family=fam, q=q, M=M, m=(int(data['G5'].shape[0]) if 'G5' in data else None),
                         quadrature=s['quadrature'], gtol=s['gtol'], solved_dimension=K + q, linear_solve=lin(K + q),
                         step_budget=strict['step_budget'], ic_budget=strict['ic_budget'], trust_radius=trust,
                         cold=cinfo, rule={k: v for k, v in rinfo.items() if not k.startswith('src_')},
                         directions_prefix_sha256=prefix[str(q)], **extra)
            return dict(s, kind='rom', data=data, cold=cold, query=query, head=head, dim=K + q, setup=setup)
        if fam == 'pod':
            k, M = s['k'], s['M']
            co = coords_full[:, :k]
            gb = A.GridBank(Vmodes[:, :k], L)
            head = A.identity_head()
            data, info = A.build_operators(gb, L, M, 'dense')
            cold, cinfo = A.build_cold(gb, head, co, cfg['cold_axis_points'])
            tr = radius(co)
            query = A.make_query(head, k, L, dt, tr, 'dense', linear=lin(k), ic_budget=strict['ic_budget'],
                                 step_budget=strict['step_budget'], gtol=s['gtol'])
            setup = dict(arm=s['name'], family='pod', k=k, M=M, m=None, quadrature='dense', gtol=s['gtol'],
                         solved_dimension=k, linear_solve=lin(k), step_budget=strict['step_budget'],
                         ic_budget=strict['ic_budget'], trust_radius=tr, cold=cinfo,
                         fit='classical POD of the same truth snapshots, identity head', array_bytes=info['array_bytes'])
            return dict(s, kind='rom', data=data, cold=cold, query=query, head=head, dim=k, setup=setup, pod_span=gb.V)
        if fam == 'free':
            M = s['M']
            head = A.identity_head()
            if Hdec is None:
                Hdec = np.asarray(jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))(jnp.asarray(Zsub)))
            data = dense_data(M)
            cold, cinfo = A.build_cold(bank, head, Hdec, cfg['cold_axis_points'])
            tr = radius(Hdec)
            query = A.make_query(head, R, L, dt, tr, 'dense', linear=lin(R), ic_budget=strict['ic_budget'],
                                 step_budget=strict['step_budget'], gtol=s['gtol'])
            setup = dict(arm=s['name'], family='free', k=R, M=M, m=None, quadrature='dense', gtol=s['gtol'],
                         solved_dimension=R, linear_solve=lin(R), step_budget=strict['step_budget'],
                         ic_budget=strict['ic_budget'], trust_radius=tr, cold=cinfo,
                         fit='unrestricted bank coefficients, identity head on R^R')
            return dict(s, kind='rom', data=data, cold=cold, query=query, head=head, dim=R, setup=setup)
        raise AssertionError(fam)

    for s in specs:
        t0 = time.perf_counter()
        try:
            b = build(s)
        except Exception as exc:              # noqa: BLE001 - OOM policy of DESIGN.md section 3.1
            if not is_oom(exc):
                raise
            report['dropped'].append(dict(name=s['name'], family=s['family'], phase='build',
                                          reason=str(exc)[:400], seconds=time.perf_counter() - t0))
            print('DROPPED (build, OOM)', s['name'], flush=True)
            jax.clear_caches()
            save()
            continue
        b['setup']['total_setup_seconds'] = time.perf_counter() - t0
        report['arm_setup'].append(b['setup'])
        built[s['name']] = b
        print('ARM', s['name'], round(time.perf_counter() - t0, 1), flush=True)
        save()

    # ------------------------------------------------ untimed diagnostics -----
    recon_done = {}
    for name, b in built.items():
        if b['family'] != 'rom' or b['q'] in recon_done:
            continue
        q, head = b['q'], b['head']
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zaug))
        Hn = jnp.sum((Hc @ Rb.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + q, L, cfg['recon_budget'], linear=lin(K + q))
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err = [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                bank_err.append(float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / n0)
                score = Hn - 2 * (Hc @ (G.T @ target))
                starts = jnp.asarray(Zaug)[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, G))
                man_err.append(float(rn) / n0)
            rows.append(dict(case=case, bank_projection_per_time=bank_err, bank_projection_max=float(np.max(bank_err)),
                             best_found_per_time=man_err, best_found_max=float(np.max(man_err))))
        entry = dict(family='rom', q=q, solved_dimension=K + q, cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)))
        recon_done[q] = entry
        report['reconstruction'].append(entry)
        print('RECON q', q, round(entry['worst_best_found'] * 100, 5), flush=True)
        save()
    for name, b in built.items():
        if b['family'] != 'pod':
            continue
        span, _ = jnp.linalg.qr(b['pod_span'], mode='reduced')
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            err = [float(jnp.linalg.norm(span @ (span.T @ jnp.asarray(ref[ti][1:-1, 1:-1].ravel()))
                                         - jnp.asarray(ref[ti][1:-1, 1:-1].ravel()))) / n0 for ti in range(ref.shape[0])]
            rows.append(dict(case=case, best_found_per_time=err, best_found_max=float(np.max(err))))
        report['reconstruction'].append(dict(family='pod', k=b['k'], solved_dimension=b['k'], cases=rows,
                                             worst_best_found=float(max(r['best_found_max'] for r in rows))))
        del span
        save()
    jax.clear_caches()

    # ---------------------------------------------------------- timed queries --
    inputs_u = [e.initial(L, phys) for phys in physical]

    def invoke(b, u, case):
        nu = float(physical[case, 4])
        if b['kind'] == 'fom':
            fn, pre = foms[b['key']]
            fs = b['setting']
            return fn(u, nu, fs['ntol'], fs['ltol'], *pre)
        return b['query'](u, nu, b['data'], b['cold'])

    t = time.perf_counter()
    subjects = []
    for s in specs:
        if s['name'] not in built:
            continue
        b = built[s['name']]
        t1 = time.perf_counter()
        try:
            jax.block_until_ready(invoke(b, jnp.asarray(inputs_u[0]), 0))
        except Exception as exc:              # noqa: BLE001
            if not is_oom(exc):
                raise
            report['dropped'].append(dict(name=s['name'], family=s['family'], phase='warmup',
                                          reason=str(exc)[:400], seconds=time.perf_counter() - t1))
            print('DROPPED (warmup, OOM)', s['name'], flush=True)
            del built[s['name']]
            jax.clear_caches()
            save()
            continue
        subjects.append(s['name'])
        print('WARM', s['name'], round(time.perf_counter() - t1, 1), flush=True)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    report['timed_subjects'] = subjects
    save()

    order_rng = np.random.default_rng(cfg['order_seed'])
    artifacts, fields_kept = {}, {}
    for rep in range(cfg['reps']):
        for case in range(len(physical)):
            for i in order_rng.permutation(len(subjects)):
                name = subjects[int(i)]
                b = built[name]
                e.burn(cfg['burn_seconds'])
                ht = time.perf_counter()
                u = jax.device_put(np.array(inputs_u[case], copy=True))
                jax.block_until_ready(u)
                gt0 = time.perf_counter()
                value = invoke(b, u, case)
                jax.block_until_ready(value)
                gs = time.perf_counter() - gt0
                f = np.asarray(value[0])
                hs = time.perf_counter() - ht
                v = host(value)
                assert np.isfinite(f).all(), name
                h = sha_array(f)
                key = (name, case, h)
                if key not in artifacts:
                    fn = f'L{L}_{name}_case{case}_rep{rep}.npz'
                    extra = dict(internal_latents=v[7]) if b['kind'] == 'rom' else {}
                    np.savez_compressed(out / fn, fields=f, **extra)
                    artifacts[key] = fn
                if (name, case) not in fields_kept:
                    fields_kept[(name, case)] = (f, v)
                row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case], rep=rep, kind=b['kind'],
                           family=b['family'], name=name, gpu_seconds=gs, host_seconds=hs, output_bytes=int(f.nbytes),
                           field_sha256=h, t0_field_sha256=sha_array(f[0]), artifact=artifacts[key],
                           error=e.errors(f, refs[case], L), iterations=v[1].tolist(), residuals=v[2].tolist(),
                           finite=True)
                if b['kind'] == 'rom':
                    reasons = v[3].tolist()
                    gn = np.asarray(v[8], dtype=float)
                    gj = np.asarray(v[12], dtype=float) if b['family'] == 'rom' else gn
                    icgj = float(v[13]) if b['family'] == 'rom' else float(v[9])
                    uin = float(v[11])
                    ic_rel = float(v[10]) / max(uin, 1e-300)
                    gtol = b['gtol']
                    step_ok = all(r in (1, 2, 4) for r in reasons)
                    step_grad_ok = all((g <= gtol * (1 + 1e-7)) or (r == 1) for g, r in zip(gj, reasons))
                    ic_ok = (int(v[6]) in (1, 2, 4)) and ((icgj <= gtol * (1 + 1e-7)) or (ic_rel <= IC_RESIDUAL_FLOOR))
                    worst = max(float(np.max(gj)), icgj)
                    row.update(q=b.get('q'), k=b.get('k'), solved_dimension=b['dim'], M=b['M'],
                               m=b['setup'].get('m'), quadrature=b['quadrature'], gtol=gtol, dt=dt,
                               rule_kind=b.get('rule_kind'), linear_solve=b['setup']['linear_solve'],
                               stop_reasons=reasons, ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=gn.tolist(), step_joint_stationarity=gj.tolist(),
                               ic_stationarity=float(v[9]), ic_joint_stationarity=icgj, ic_residual=float(v[10]),
                               ic_input_norm=uin, ic_relative_residual=ic_rel,
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
                               residual_exits=int(sum(1 for r in reasons if r == 1)),
                               tiny_step_exits=int(sum(1 for r in reasons if r == 2)),
                               gradient_exits=int(sum(1 for r in reasons if r == 4)),
                               max_iterations=int(np.max(v[1])), median_iterations=float(np.median(v[1])),
                               worst_joint_stationarity=worst,
                               completed=bool(step_ok and int(v[6]) in (1, 2, 4)),
                               converged_strict=bool(worst <= gtol * (1 + 1e-7) and step_ok and int(v[6]) in (1, 2, 4)),
                               converged=bool(step_ok and step_grad_ok and ic_ok))
                else:
                    fs = b['setting']
                    row.update(dt=fs['dt'], ntol=fs['ntol'], ltol=fs['ltol'], preconditioner=fs['preconditioner'],
                               newton_iterations_total=int(np.sum(v[1])),
                               nonlinear_converged=bool(np.max(v[2]) <= fs['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------------------------- in-job gates
    def rel(x, y):
        return float(np.linalg.norm(np.asarray(x) - np.asarray(y)) / max(np.linalg.norm(np.asarray(y)), 1e-300))

    tight = cfg.get('same_grid_reference', 'fft_tight')
    if tight in built and cfg.get('direct_control') in built:
        d = max(rel(fields_kept[(cfg['direct_control'], c)][0], fields_kept[(tight, c)][0]) for c in range(len(physical)))
        report['gates']['direct_reproduces_fft_tight'] = dict(passed=bool(d <= 1e-9), worst_relative=d,
                                                              direct=cfg['direct_control'], tight=tight)
    for s in specs:
        if s['family'] == 'fast' and s['name'] in built and s['parity_against'] in built:
            worst_f, same_ints = 0., True
            per = []
            for c in range(len(physical)):
                fa, va = fields_kept[(s['name'], c)]
                fb, vb = fields_kept[(s['parity_against'], c)]
                d = rel(fa, fb)
                ints = bool(np.array_equal(va[1], vb[1]) and np.array_equal(va[3], vb[3]) and int(va[5]) == int(vb[5]))
                per.append(dict(case=c, relative=d, integers_identical=ints))
                worst_f, same_ints = max(worst_f, d), same_ints and ints
            report['gates']['fast_parity'] = dict(passed=bool(worst_f <= cfg['fast_parity_bar'] and same_ints),
                                                  worst_relative=worst_f, integers_identical=same_ints,
                                                  bar=cfg['fast_parity_bar'], against=s['parity_against'], cases=per)
    hashes = {}
    for x in report['invocations']:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    report['gates']['repetition_output_identical'] = dict(passed=all(len(v) == 1 for v in hashes.values()))
    report['gates']['fft_tight_converged_everywhere'] = dict(
        passed=all(x['nonlinear_converged'] for x in report['invocations'] if x['name'] == tight))

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
