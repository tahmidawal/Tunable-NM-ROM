"""quadratic-manifold: the b-panel harness with a QUADRATIC-MANIFOLD trial map added.

b-panel's `panel.py` (exp/2026-09-17-b-panel @ 25434a27) with one new subject family, `qman`,
and nothing else changed. One job, one GPU, one randomised order, timed repetitions with
burn-in, for the frozen Burgers model (bank G, head h_theta, transferred directions C):

  * the NM-ROM correction ladder, dense and certified-EQ quadrature, and b-speed's optimised
    kernel -- the paper's fast and accurate settings;
  * POD-LSPG at the same ranks the quadratic manifold is run at, through the same objective;
  * NEW -- the `qman` family: u = u_ref + V_r a + W vech(a a^T) with W from one regularised
    linear solve on the same truth snapshots (Geelen-Wright-Willcox 2022; Barnett-Farhat 2022),
    and the same map with the W block dropped, so the quadratic term is isolated against an
    identical linear part. The trial map is the ONLY thing that changes: the columns
    [u_ref | V_r | W] become an `arms.GridBank`, the coefficient map is `qman.head`, and the
    residual, test projection, initializer, LM driver, budgets and output contract are the
    shared ones. See `qman.py` and DESIGN.md section 3;
  * a grid of same-grid full-order Newton controls plus the converged `fft_tight` reference
    every same-grid error is measured against, and its dense-preconditioner twin.

Every subject is built and warmed up in a declared priority order; a subject that fails with
RESOURCE_EXHAUSTED is recorded as dropped and the job continues. Everything is written
incrementally to `result.json` so a truncated job is collectable. See `DESIGN.md` for the
contract, the gates, the convergence rule and the non-dominance rule.
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
import qman as QM

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


def gt(g):
    return f'g{g:g}'.replace('-', 'm').replace('.', 'p')


def declare_subjects(cfg, K, R, sets):
    """Every subject the job builds, in build priority order (DESIGN.md sections 3.1-3.2).

    Pure configuration: no file is read and no JAX work is done, so main() calls it before the
    references, the snapshots and the rule transfers, and a malformed config fails at once.
    `priority_override` is a list of subject NAMES (strings); every name must be a declared
    subject. bpn201 (job 3783817) died twenty minutes in, after all six rule transfers, because
    this list was indexed as a list of dicts and no smoke config carried the key (DESIGN A6).
    """
    strict = cfg['strict']
    ranks = sorted(set(cfg.get('pod_ranks', [])))
    specs = []
    for fs in cfg['fom_settings']:
        specs.append(dict(name=fs['name'], family='fom', setting=fs, priority=0))
    for q in cfg['dense_q']:
        specs.append(dict(name=f'q{q}_M{4 * (K + q)}_dense_{gt(strict["gtol"])}', family='rom', q=q,
                          M=4 * (K + q), quadrature='dense', gtol=strict['gtol'], priority=1))
    for g in cfg['eq_gtols']:
        pr = 2 if g == strict['gtol'] else 4
        for si, rs in enumerate(sets):
            for q in cfg['eq_q']:
                kind = rs['resolved']
                specs.append(dict(name=f'q{q}_M{4 * (K + q)}_{kind}_{gt(g)}', family='rom', q=q,
                                  M=4 * (K + q), quadrature='eq', gtol=g, priority=pr + (0 if si == 0 else 0.5),
                                  rule_kind=kind, rule_set=kind))
    for k in ranks:
        specs.append(dict(name=f'pod{k}_M{4 * k}_dense', family='pod', k=k, M=4 * k, quadrature='dense',
                          gtol=strict['gtol'], priority=3))
    # quadratic-manifold lane: u = u_ref + V_r a + W vech(a a^T), and the same map with W
    # dropped, at the SAME solved dimension r and the same M = 4r as POD-LSPG (DESIGN section 3).
    for r_ in sorted(set(cfg.get('qman_ranks', []))):
        for variant in cfg.get('qman_variants', ['quad', 'lin']):
            assert variant in ('quad', 'lin'), variant
            specs.append(dict(name=f'qman{r_}_{variant}_M{4 * r_}', family='qman', k=r_, M=4 * r_,
                              variant=variant, quadrature='dense', gtol=strict['gtol'], priority=3.5))
    for q in cfg.get('extra_dense_q', []):
        M = int(cfg['extra_dense_M'])
        specs.append(dict(name=f'q{q}_M{M}_dense_{gt(strict["gtol"])}', family='rom', q=q, M=M,
                          quadrature='dense', gtol=strict['gtol'], priority=5))
    if cfg.get('free_bank'):
        specs.append(dict(name=f'free{R}_M{cfg["free_bank_M"]}_dense', family='free', k=R, M=int(cfg['free_bank_M']),
                          quadrature='dense', gtol=strict['gtol'], priority=6))
    if cfg.get('fast_arm') and 0 in cfg['eq_q']:
        kind = sets[0]['resolved']
        specs.append(dict(name=f'q0_M{4 * K}_{kind}_{gt(strict["gtol"])}_fast{cfg["fast_arm"]}', family='fast', q=0,
                          M=4 * K, quadrature='eq', gtol=strict['gtol'], priority=7, rule_kind=kind,
                          parity_against=f'q0_M{4 * K}_{kind}_{gt(strict["gtol"])}'))
    override = list(cfg.get('priority_override', []))
    assert all(isinstance(n, str) for n in override), f'priority_override must list subject names: {override}'
    declared = [s['name'] for s in specs]
    assert len(declared) == len(set(declared)), 'duplicate subject names'
    missing = [n for n in override if n not in declared]
    assert not missing, f'priority_override names no declared subject: {missing}; declared: {declared}'
    order = {name: i for i, name in enumerate(override)}
    # A name in `priority_override` is built right after the FOM controls, in the listed order,
    # whatever its family: DESIGN.md section 3.2's reduced set comes before everything else.
    specs.sort(key=lambda s: ((0.5, order[s['name']]) if s['name'] in order else (s['priority'], 0)))
    return specs


def resolve_rule_sets(cfg, L):
    """The named rule sets a job carries (DESIGN.md A5.1) with their resolved arm names.

    The primary set is named by whether it is used at its own mesh (`eqcert`) or transferred to
    another (`eqxfer`); an extra set carries its declared name, suffixed `xfer` when transferred.
    """
    sets = [dict(name=None, mesh=int(cfg['rules_mesh']), rules=cfg['rules'], subdir='rules')]
    for es in cfg.get('extra_rule_sets', []):
        sets.append(dict(name=es['name'], mesh=int(es['mesh']), rules=es['rules'],
                         subdir=es.get('subdir', 'rules')))
    names = [x['name'] for x in sets]
    assert len(names) == len(set(names)), f'duplicate rule-set names: {names}'
    for rs in sets:
        at_mesh = (L == rs['mesh'])
        sname = rs['name'] or ('eqcert' if at_mesh else 'eqxfer')
        if rs['name'] and not at_mesh:
            sname = f"{rs['name']}xfer"
        rs['resolved'] = sname
    return sets


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
    sets = resolve_rule_sets(cfg, L)
    specs = declare_subjects(cfg, K, R, sets)      # validates priority_override before any work
    print('DECLARED', len(specs), 'subjects:', ' '.join(s['name'] for s in specs), flush=True)
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
                  quadratic_manifold=[],
                  reconstruction=[], invocations=[], declared_subjects=[], dropped=[], gates={},
                  fno=dict(prepared=False), verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    # ---------------------------------------------------------------- cohort --
    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    roles = ['opened development'] * cfg['eval_cases'] + ['fresh development'] * cfg['eval_fresh_cases']
    report['physical_sha256'] = sha_array(physical)      # of the FULL draw: params_draw is column-wise
    if cfg.get('case_subset'):
        # smoke only: keep listed cases of the full draw (drawing fewer would change the cases)
        idx = [int(i) for i in cfg['case_subset']]
        physical, roles = physical[idx], [roles[i] for i in idx]
        report['case_subset'] = idx
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = roles
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
    qranks = sorted(set(cfg.get('qman_ranks', [])))
    qman_maps = {}
    # The quadratic manifolds are built from the same snapshot matrix as the POD arms, so the
    # snapshot block must run; a config that asks for one without the other is rejected here.
    assert not qranks or ranks, 'qman_ranks needs pod_ranks: both are built from the same snapshots'
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
        # The quadratic manifolds are fitted HERE, while the snapshot matrix is still resident:
        # one ridge-regularised linear solve per rung, no training run. The ridge is selected on
        # a seeded held-out split of these snapshots and never on the evaluation cohort.
        for r_ in qranks:
            m_ = QM.fit(Ut, r_, cfg['qman_gammas'], cfg['qman_seed'], cfg['qman_holdout'])
            qman_maps[r_] = m_
            report['quadratic_manifold'].append(dict(m_['info'], snapshot_sha256=sinfo['snapshot_sha256'],
                                                     coefficients_sha256=sha_array(m_['coefficients'])))
            print('QMAN r', r_, 'P', m_['info']['quadratic_terms'], 'ridge', m_['info']['ridge'],
                  'heldout', round(m_['info']['heldout_relative'], 6),
                  'lin/quad', round(m_['info']['snapshot_relative_linear_only'], 6),
                  round(m_['info']['snapshot_relative_with_quadratic'], 6),
                  round(m_['info']['seconds'], 1), flush=True)
            save()
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
    recorded = {k: v for k, v in prefix.items() if k in dprov['prefix_sha256']}
    # every rung the source recorded must match; a rung it never recorded (smoke q = 4) is reported, not gated
    ok_prefix = bool(recorded) and all(dprov['prefix_sha256'][k] == v for k, v in recorded.items())
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
    dense_queries = {}

    def dense_query(q):
        # one compiled dense query per q, shared by the transfer population and the timed arm
        if q not in dense_queries:
            dense_queries[q] = TF.make_query(params, Cfull[:, :q], K, q, L, dt, trust, 'dense', 'base', Rb=Rb,
                                             ic_budget=strict['ic_budget'], step_budget=strict['step_budget'],
                                             gtol=strict['gtol'], ic_gtol=cfg['ic_gtol'], linear=lin(K + q),
                                             inner_damping=cfg['inner_damping'], tau_y=cfg['tau_y'])
        return dense_queries[q]

    # ---------------------------------------------------------- the rules ----
    # A job may carry SEVERAL named rule sets at matched q, M and tolerance, so that a new set
    # is measured against its predecessor INSIDE one allocation instead of across jobs
    # (DESIGN.md A5.1). The primary set is named by whether it is used at its own mesh
    # (`eqcert`) or transferred to another (`eqxfer`); extra sets carry their declared name.
    rules = {}          # (set_name, q) -> (ops dict with G5/Pq, info)
    xfer = cfg.get('eq_transfer')
    for rs in sets:
        at_mesh = (L == rs['mesh'])
        sname = rs['resolved']
        for q in cfg['eq_q']:
            spec = rs['rules'][str(q)]
            rfile = inputs / rs['subdir'] / spec['file']
            rprov = prov[f"{rs['subdir']}/{spec['file']}"]
            got = sha_file(rfile)
            assert got == rprov['sha256'], (spec['file'], got, rprov['sha256'])
            z = np.load(rfile)
            nodes, weights = np.asarray(z['nodes'], dtype=int), np.asarray(z['weights'], dtype=float)
            M = 4 * (K + q)
            assert rprov['M'] == M and rprov['q'] == q
            info = dict(q=q, M=M, rule_set=sname, file=spec['file'], sha256=got,
                        source_job=rprov['source_job'], source_m=int(len(nodes)), source_mesh=rs['mesh'],
                        source_rho_max=rprov['rho_max'], source_rho_p95=rprov['rho_p95'],
                        source_relative_fit=rprov['relative_fit'], rho_bar=rprov['rho_bar'],
                        source_fit_states=rprov.get('fit_states'),
                        certified_primary=rprov['certified_primary'],
                        certified_secondary=rprov['certified_secondary'],
                        basis=('primary' if rprov['certified_primary'] else
                               'secondary' if rprov['certified_secondary'] else 'none'),
                        nodes_sha256=sha_array(nodes), weights_sha256=sha_array(weights),
                        nodes_sha256_matches=(sha_array(nodes) == rprov['nodes_sha256']),
                        weights_sha256_matches=(sha_array(weights) == rprov['weights_sha256']))
            assert info['nodes_sha256_matches'] and info['weights_sha256_matches'], spec['file']
            # DESIGN A8: the construction status (confirmed / marginal / certified in one draw /
            # single qrg304 draw) travels with the rule into every row and caption.
            info.update(source_lane=rprov.get('source_lane', 'q-ridge'), source_attempt=rprov.get('source_attempt'),
                        construction_status=rprov.get('construction_status'), export_basis=rprov.get('export_basis'),
                        construction_note=rprov.get('construction_note') or rprov.get('status_note'),
                        same_file_as_qrg304_rule=rprov.get('same_file_as_qrg304_rule'))
            if at_mesh:
                ph, _ = modes(M)
                ops = rule_operators(bank, ph, L, nodes, weights)
                info.update(kind=sname, m=int(len(nodes)), transferred=False)
                rules[(sname, q)] = (ops, info)
            else:
                assert xfer is not None, 'a rule at another mesh needs an eq_transfer block'
                rules[(sname, q)] = (None, dict(info, kind=sname, transferred=True,
                                                src_nodes=nodes, src_weights=weights))
            report['rules'].append({k: v for k, v in info.items() if not k.startswith('src_')})
            print('RULE', sname, 'q', q, 'm', len(nodes), info['basis'], flush=True)
    save()

    # ------------------------------------------ transferred rules (job 2/3) --
    if xfer is not None and rules:
        rng = np.random.default_rng(xfer['seed'])
        fit_idx = np.asarray(xfer['fit_trajectories'], dtype=int)
        cert_idx = np.asarray(xfer['cert_trajectories'], dtype=int)
        assert not set(fit_idx.tolist()) & set(cert_idx.tolist())
        report['gates']['transfer_fit_cert_disjoint'] = dict(passed=True, fit=fit_idx.tolist(), cert=cert_idx.tolist())
        for (sname, q) in [k for k, v in rules.items() if v[1].get('transferred')]:
            ops, info = rules[(sname, q)]
            M = 4 * (K + q)
            C = Cfull[:, :q]
            cold, _, head = cold_for(q)
            dd = dense_data(M)
            ph, _ = modes(M)
            t0 = time.perf_counter()
            # The reachable population is the PRODUCTION dense query's own converged per-step
            # states (its internal latents), DESIGN.md A3: qrg304's fixed-iterate collector never
            # accepts a step when the first time step needs more iterations than it unrolls
            # (29 at 64 intervals from the same start), and then returns one frozen state.
            qfn = dense_query(q)
            cf = jax.jit(jax.vmap(head))
            pools = {}
            for tag, idx in (('fit', fit_idx), ('cert', cert_idx)):
                rows = []
                for i in idx:
                    phys = train_physical[i]
                    v = qfn(jnp.asarray(e.initial(L, phys)), float(phys[4]), dd, cold)
                    wv = np.asarray(v[7]).reshape(-1, K + q)
                    assert np.isfinite(wv).all() and np.linalg.norm(wv[-1] - wv[0]) > 0, ('frozen rollout', q, int(i))
                    rows.append(np.asarray(cf(jnp.asarray(wv))))
                pools[tag] = np.concatenate(rows)
            # DESIGN A7: every configured fit state at every rung. The former convention
            # clip(max_fit_rows / M, 8, fit_states) gave bpn201 / bpn202 14 and 8 states at q = 128
            # and 256 -- the starvation b-eqtop identified -- and their top rungs came back
            # uncertified. 64 states x M = 1088 rows on a 2048-point support is 87 s of host NNLS
            # (checks/nnls-size.json), so no cap is needed.
            assert 'max_fit_rows' not in xfer, 'max_fit_rows is retired (DESIGN A7): the fit-state count is fit_states'
            nfit = int(xfer['fit_states'])
            sel = np.sort(rng.choice(len(pools['fit']), min(nfit, len(pools['fit'])), replace=False))
            csel = pools['cert'][np.sort(rng.choice(len(pools['cert']), min(xfer['cert_states'], len(pools['cert'])),
                                                    replace=False))]
            pos, ij = transfer_support(info['src_nodes'], cfg['rules_mesh'], L)
            w, finfo = refit_weights(G, dd['Phi'], ph, L, pos, pools['fit'][sel], M)
            keep = w > 0
            ops = dict(G5=bank.stencil(ij[keep], L), Pq=jnp.asarray(np.asarray(ph)[pos[keep]] * w[keep][:, None]))
            cert = EC.certify(G, dd['Phi'], L, ops, csel, chunk=xfer['certify_chunk'])
            bar = float(info['rho_bar'])
            tinfo = dict(q=q, M=M, rule_set=sname, m_support=int(len(pos)), m=int(keep.sum()),
                         refit=finfo, certification=cert,
                         fit_states_available=int(len(pools['fit'])), fit_states_used=int(len(sel)),
                         fit_state_rule='fit_states, uncapped (DESIGN A7)',
                         certification_states=int(len(csel)), population='production dense query, converged per-step states (A3)',
                         fit_pool_sha256=sha_array(pools['fit']), certification_pool_sha256=sha_array(pools['cert']),
                         certified_primary=bool(cert['rho_max'] <= bar), certified_secondary=bool(cert['rho_p95'] <= bar),
                         rho_bar=bar, source_rho_max=info['source_rho_max'], seconds=time.perf_counter() - t0)
            tinfo['basis'] = ('primary' if tinfo['certified_primary'] else 'secondary' if tinfo['certified_secondary'] else 'none')
            info = dict({k: v for k, v in info.items() if not k.startswith('src_')}, m=tinfo['m'],
                        basis=tinfo['basis'], certified_primary=tinfo['certified_primary'],
                        certified_secondary=tinfo['certified_secondary'], rho_max=cert['rho_max'], rho_p95=cert['rho_p95'])
            rules[(sname, q)] = (ops, info)
            np.savez_compressed(out / f'rule_xfer_{sname}_q{q}_L{L}.npz', nodes=pos[keep], weights=w[keep])
            report['transfer'].append(tinfo)
            print('XFER', sname, 'q', q, 'm', tinfo['m'], 'rho_max', f"{cert['rho_max']:.4f}",
                  'p95', f"{cert['rho_p95']:.4f}", tinfo['basis'], round(tinfo['seconds'], 1), flush=True)
            save()

    # ------------------------------------------------ untimed diagnostics -----
    # DESIGN A7: run BEFORE the subjects are built, while the GPU holds only the bank, the test
    # matrices and the snapshot basis. bpn202 (job 3787247, 1024^2, H200) died in this diagnostic
    # at q = 256 with 29 subjects resident: the eight-start vmapped Jacobian batch is 18 GB and an
    # [8, n, 512] intermediate 34 GB. The programs and their inputs are unchanged, so the values
    # are; the diagnostic is also OOM-tolerant (recorded as dropped, the job continues). Chunking
    # the starts was tried first and dropped: it agrees with the vmapped batch only to round-off
    # (1e-13 residuals, checks/recon-chunk-probe.json), whereas moving the diagnostic keeps the
    # pre-registered program byte-identical.
    recon_done = {}
    for s in specs:
        if s['family'] != 'rom' or s['q'] in recon_done:
            continue
        q = s['q']
        _, _, head = cold_for(q)
        t0 = time.perf_counter()
        try:
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
                         worst_best_found=float(max(r['best_found_max'] for r in rows)),
                         starts=int(cfg['recon_starts']), budget=int(cfg['recon_budget']), seconds=time.perf_counter() - t0)
        except Exception as exc:              # noqa: BLE001 - DESIGN A7: an untimed diagnostic must not kill the job
            if not is_oom(exc):
                raise
            report['dropped'].append(dict(name=f'reconstruction_q{q}', family='diagnostic', phase='reconstruction',
                                          reason=str(exc)[:400], seconds=time.perf_counter() - t0))
            print('DROPPED (reconstruction, OOM) q', q, flush=True)
            recon_done[q] = None
            jax.clear_caches()
            save()
            continue
        recon_done[q] = entry
        report['reconstruction'].append(entry)
        print('RECON q', q, round(entry['worst_best_found'] * 100, 5), round(entry['seconds'], 1), flush=True)
        save()
    for k in ranks:
        span, _ = jnp.linalg.qr(Vmodes[:, :k], mode='reduced')
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            err = [float(jnp.linalg.norm(span @ (span.T @ jnp.asarray(ref[ti][1:-1, 1:-1].ravel()))
                                         - jnp.asarray(ref[ti][1:-1, 1:-1].ravel()))) / n0 for ti in range(ref.shape[0])]
            rows.append(dict(case=case, best_found_per_time=err, best_found_max=float(np.max(err))))
        report['reconstruction'].append(dict(family='pod', k=k, solved_dimension=k, cases=rows,
                                             worst_best_found=float(max(r['best_found_max'] for r in rows))))
        del span
        save()
    jax.clear_caches()

    # ---------------------------------------------------------- subjects -----
    # Declared (and validated) at the top of main(), before any expensive work; the list is
    # unchanged here and is recorded so result.json carries it in the pre-existing position.
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
                ops, rinfo = rules[(s.get('rule_set') or sets[0]['resolved'], q)]
                # DESIGN A8: A = Phi^T G and lambda are the dense arm's own resident arrays for
                # this M (the same device inputs, the same program). Every EQ arm used to hold
                # its own copy of Phi: 24 copies at 256^2 (4.9 GB), 12 at 1024^2 (40 GB).
                dd = dense_data(M)
                data = dict(A=dd['A'], lam=dd['lam'], G=G, G5=ops['G5'], Pq=ops['Pq'])
            if fam == 'rom':
                if s['quadrature'] == 'dense' and s['gtol'] == strict['gtol'] and M == 4 * (K + q):
                    query = dense_query(q)
                else:
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
            return dict(s, kind='rom', data=data, cold=cold, query=query, head=head, dim=k, setup=setup)
        if fam == 'qman':
            # The ONLY thing that changes is the trial map: B = [u_ref | V_r | W] with the
            # coefficient map eta(a) = [1, a, vech(a a^T)] (or [1, a] for the `lin` control on the
            # identical u_ref and V_r). Residual, test projection, initializer, LM driver, budgets,
            # tolerance and output contract are `arms`' shared ones, exactly as for POD-LSPG.
            r_, M, quad = s['k'], s['M'], (s['variant'] == 'quad')
            mp = qman_maps[r_]
            Bq = QM.bank_columns(mp, quad)
            gb = A.GridBank(Bq, L)
            head = QM.head(r_, quad)
            co = mp['coefficients']
            data, info = A.build_operators(gb, L, M, 'dense')
            # The fixed-Gauss initializer must stay over-determined in the bank's OWN columns:
            # the quadratic block takes D from r to 1 + r + r(r+1)/2, which at r = 64 is 2145
            # against the shared rule's 48^2 = 2304 points. `qman_cold_axis_points` raises the
            # rule for this family only; declared in DESIGN section 3.2 as a necessary deviation.
            axis = int(cfg.get('qman_cold_axis_points', cfg['cold_axis_points']))
            assert axis ** 2 > 2 * int(Bq.shape[1]), (axis, Bq.shape)
            cold, cinfo = A.build_cold(gb, head, co, axis)
            tr = radius(co)
            query = A.make_query(head, r_, L, dt, tr, 'dense', linear=lin(r_), ic_budget=strict['ic_budget'],
                                 step_budget=strict['step_budget'], gtol=s['gtol'])
            setup = dict(arm=s['name'], family='qman', k=r_, M=M, m=None, quadrature='dense', gtol=s['gtol'],
                         solved_dimension=r_, linear_solve=lin(r_), step_budget=strict['step_budget'],
                         ic_budget=strict['ic_budget'], trust_radius=tr, cold=cinfo, variant=s['variant'],
                         quadratic=bool(quad), bank_columns=int(Bq.shape[1]),
                         quadratic_terms=(QM.terms(r_) if quad else 0), manifold_fit=mp['info'],
                         fit=('quadratic manifold u_ref + V_r a + W vech(a a^T), W by ridge least squares on the '
                              'same truth snapshots' if quad else
                              'affine linear control u_ref + V_r a on the identical u_ref and V_r'),
                         array_bytes=info['array_bytes'])
            return dict(s, kind='rom', data=data, cold=cold, query=query, head=head, dim=r_, setup=setup)
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
                    row.update(q=b.get('q'), k=b.get('k'), variant=b.get('variant'), solved_dimension=b['dim'], M=b['M'],
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
    # DESIGN A8: two rule SETS carrying the same rule file at a rung must produce bitwise
    # identical arms at matched tolerance (b-eqtop's q = 0, 16, 32 rules are qrg304's files).
    file_sha = {(x['rule_set'], x['q']): x['sha256'] for x in report['rules']}
    twins = []
    for (sa, q), ha in file_sha.items():
        for (sb, q2), hb in file_sha.items():
            if q2 != q or not sa < sb or ha != hb:
                continue
            for g in cfg['eq_gtols']:
                na, nb = f'q{q}_M{4 * (K + q)}_{sa}_{gt(g)}', f'q{q}_M{4 * (K + q)}_{sb}_{gt(g)}'
                if na not in built or nb not in built:
                    continue
                same = all(hashes[(na, c)] == hashes[(nb, c)] for c in range(len(physical)))
                ints = all(np.array_equal(fields_kept[(na, c)][1][1], fields_kept[(nb, c)][1][1])
                           and np.array_equal(fields_kept[(na, c)][1][3], fields_kept[(nb, c)][1][3])
                           for c in range(len(physical)))
                twins.append(dict(q=q, gtol=g, arms=[na, nb], file_sha256=ha, same_fields=bool(same), same_iterations=bool(ints)))
    report['gates']['matched_rule_files_bitwise'] = dict(
        passed=(all(t['same_fields'] and t['same_iterations'] for t in twins) if twins else None), pairs=twins,
        note='None when no two sets share a rule file; a pair that differs means the multi-set path perturbs a result')
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
