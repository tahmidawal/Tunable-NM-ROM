"""Local smoke for the quadratic-manifold lane: the real driver at 64 intervals, then the audit.

    jaxrun python smoke_panel.py <out.json>

b-panel's `smoke_panel.py` (exp/2026-09-17-b-panel @ 25434a27) cut down to this lane: the
128-interval rule-TRANSFER half and the duplicate-rule-set gate are dropped because this lane
runs one mesh with one rule set and never transfers a rule. What is kept is the part that
protects a number, plus the new family's own gates:

  1 `rule_operators` rebuilt from the consolidated fixture's own (nodes, weights) equals the
    fixture's G5 and Pq to 1e-13, and the driver's q = 0 certified-EQ arm at 64 intervals then
    reproduces the consolidated saved Burgers case (`expected.npz`, the panel's case 0) to
    <= 1e-12 relative -- the audited baseline number of the parent lanes;
  2 every family this lane runs (dense rom, EQ rom, POD-LSPG, the fast kernel, the FOM grid and
    the QUADRATIC MANIFOLD at two ranks, W on and off) builds, warms up, times and audits;
  3 the quadratic block actually changes the answer: at each rank the `quad` and `lin` arms of
    the identical (u_ref, V_r) must differ in their output fields, and the `quad` arm must carry
    1 + r + r(r+1)/2 bank columns against the `lin` arm's 1 + r;
  4 `qman.demo()` -- the bank/head algebra and the analytic Jacobian -- passes;
  5 the cluster config's `priority_override` names only declared subjects.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANE = 'experiments/quadratic-manifold'
SUBS = (LANE, f'{LANE}/speed', 'experiments/q-ridge', 'experiments/b-ladder-top',
        'experiments/cheap-corrections', 'experiments/head-ablation', 'experiments/mr-burgers2d',
        'experiments/separable-decoder')
for sub in SUBS:
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import panel as PN             # noqa: E402
import qman as QM              # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/'
              'accuracy09/archive/out/result.json')
PY = sys.executable


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    t_all = time.perf_counter()
    out_json = Path(sys.argv[1])
    scratch = Path(os.environ.get('SMOKE_SCRATCH', str(HERE / 'runs/smoke')))
    if scratch.exists() and os.environ.get('SMOKE_REUSE') != '1':
        shutil.rmtree(scratch)
    inputs = scratch / 'inputs'
    (inputs / 'rules').mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / 'inputs/directions_qtd02.npz', inputs / 'directions_qtd02.npz')
    real_prov = json.loads((HERE / 'inputs/PROVENANCE.json').read_text())
    out = {}

    QM.demo()
    out['qman_demo'] = 'passed'

    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = int(Zold.shape[1]), int(np.asarray(ck['params']['h_lin']).shape[1])
    raw = json.loads(RAW.read_text())
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))
    L = 64
    bank = A.CoordBank(params, K, R)
    arch = {k: np.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}

    # ---- gate 1a: the rule rebuild path against the fixture operators ---------
    nodes0 = np.asarray(setup['eq_indices'], dtype=int)
    w0 = np.asarray(setup['eq_weights'], dtype=float)
    Phi64, _, _ = e.modes(L, 64)
    ops = PN.rule_operators(bank, Phi64, L, nodes0, w0)
    g5, pq = rel(ops['G5'], arch['G5']), rel(ops['Pq'], arch['Pq'])
    out['rule_rebuild_vs_fixture'] = dict(G5_relative=g5, Pq_relative=pq, m=int(len(nodes0)))
    assert g5 <= 1e-13 and pq <= 1e-13, (g5, pq)
    print('GATE1a rule rebuild', g5, pq, flush=True)

    prov = dict(note='SMOKE inputs', files={})
    prov['files']['directions_qtd02.npz'] = real_prov['files']['directions_qtd02.npz']
    fn = f'rule_q0_smoke_m{len(nodes0)}.npz'
    np.savez_compressed(inputs / 'rules' / fn, nodes=nodes0, weights=w0)
    prov['files'][f'rules/{fn}'] = dict(
        sha256=LD.sha_file(inputs / 'rules' / fn), source_job='smoke', q=0, M=64, m=int(len(nodes0)),
        relative_fit=float(setup['eq_relative_fit']), rho_max=None, rho_p95=None, rho_bar=0.116,
        certified_primary=False, certified_secondary=False,
        nodes_sha256=LD.sha_array(nodes0), weights_sha256=LD.sha_array(w0))
    (inputs / 'PROVENANCE.json').write_text(json.dumps(prov, indent=2) + '\n')

    # ---- gates 1c, 2, 3: the driver at 64 intervals ---------------------------
    env = dict(os.environ, JAX_ENABLE_X64='true', JAX_DEFAULT_MATMUL_PRECISION='highest',
               PYTHONPATH=':'.join(str(ROOT / s) for s in SUBS))
    od = scratch / 'smoke64/output'
    t0 = time.perf_counter()
    reuse = os.environ.get('SMOKE_REUSE') == '1' and (od / 'COMPLETE').exists()
    if not reuse:
        subprocess.run([PY, str(HERE / 'panel.py'), '--config', str(HERE / 'config-smoke64.json'),
                        '--checkpoint', str(CK), '--inputs', str(inputs), '--out', str(od)], check=True, env=env)
    res = json.loads((od / 'result.json').read_text())
    assert np.allclose(np.asarray(res['physical_cases'][0]), np.asarray(raw['physical_cases'][0]),
                       rtol=0, atol=1e-15), 'case 0 is not the fixture case'
    assert res['complete'] and not res['dropped'], res['dropped']
    run = dict(seconds=time.perf_counter() - t0, reused_existing_output=bool(reuse),
               subjects=res['timed_subjects'], gates={k: v.get('passed') for k, v in res['gates'].items()})
    print('DRIVER smoke64', round(run['seconds'], 1), len(res['timed_subjects']), 'subjects', flush=True)

    cfg0 = json.loads((HERE / 'config-smoke64.json').read_text())
    au = scratch / 'smoke64/audit.json'
    subprocess.run([PY, str(HERE / 'audit_panel.py'), str(od / 'result.json'), '--fields', str(od),
                    '--out', str(au), '--comparators', str(HERE / 'checks/comparators.json')], check=True)
    aj = json.loads(au.read_text())
    run['audit_failed'] = aj['failed']
    run['arms'] = {x['arm']: dict(family=x['family'], evolved=x['worst_evolved_percent'],
                                  gpu_ms=x['median_gpu_ms'], converged=x['converged_design5'],
                                  admissible=x['admissible'], bank_columns=x.get('bank_columns'),
                                  best_found=x.get('best_found_percent'))
                   for x in aj['arms']}
    assert not aj['failed'], aj['failed']

    saved = np.load(FIX / 'expected.npz')
    x = next(i for i in res['invocations'] if i['name'] == 'q0_M64_eqcert_g1em06' and i['case'] == 0)
    f = np.load(od / x['artifact'])
    r0 = rel(f['fields'], saved['fields'])
    out['baseline_saved_case'] = dict(arm=x['name'], relative_l2_to_saved_case=r0,
                                      latent_relative_l2=rel(f['internal_latents'], saved['internal_latents']),
                                      budget_exits=x['budget_exits'], converged=x['converged'])
    assert r0 <= 1e-12, r0
    print('GATE1c baseline', r0, flush=True)
    assert res['gates']['fast_parity']['passed'], res['gates']['fast_parity']
    assert res['gates']['quadratic_block_changes_the_answer']['passed'], \
        res['gates']['quadratic_block_changes_the_answer']
    # DESIGN A1: the representation floor must exist for every qman arm, or a large evolved error
    # cannot be attributed to the manifold rather than to the solve
    fam = {(e['k'], e['variant']) for e in res['reconstruction'] if e['family'] == 'qman'}
    assert fam == {(r_, v) for r_ in cfg0['qman_ranks'] for v in cfg0['qman_variants']}, fam
    assert res['gates']['direct_reproduces_fft_tight']['passed'], res['gates']['direct_reproduces_fft_tight']

    # ---- gate 3: the quadratic block is real and changes the answer -----------
    cfg = cfg0
    art = {(i['name'], i['case']): i['artifact'] for i in res['invocations']}
    cases = sorted({i['case'] for i in res['invocations']})
    qm = []
    for r_ in cfg['qman_ranks']:
        nq, nl = f'qman{r_}_quad_M{4 * r_}', f'qman{r_}_lin_M{4 * r_}'
        sq = next(s for s in res['arm_setup'] if s['arm'] == nq)
        sl = next(s for s in res['arm_setup'] if s['arm'] == nl)
        assert sq['bank_columns'] == 1 + r_ + QM.terms(r_) and sl['bank_columns'] == 1 + r_, (sq, sl)
        assert sq['quadratic_terms'] == QM.terms(r_) and sl['quadratic_terms'] == 0
        assert sq['manifold_fit']['ridge'] in sq['manifold_fit']['ridge_grid']
        # DESIGN A1: the ridge split is by TRAJECTORY, and W is finite and non-trivial
        fi = sq['manifold_fit']
        assert fi['split'].startswith('by trajectory') and fi['heldout_trajectories'] >= 1
        assert fi['trajectories'] * fi['states_per_trajectory'] == fi['snapshots']
        assert fi['weight_frobenius_norm'] > 0 and all(t['finite'] for t in fi['ridge_trace'])
        d = min(rel(np.load(od / art[(nq, c)])['fields'], np.load(od / art[(nl, c)])['fields']) for c in cases)
        assert d > 1e-6, (r_, d)
        qm.append(dict(rank=r_, bank_columns=sq['bank_columns'], ridge=sq['manifold_fit']['ridge'],
                       heldout_relative=sq['manifold_fit']['heldout_relative'],
                       snapshot_linear_only=sq['manifold_fit']['snapshot_relative_linear_only'],
                       snapshot_with_quadratic=sq['manifold_fit']['snapshot_relative_with_quadratic'],
                       min_quad_vs_lin_field_relative=d,
                       weight_frobenius_norm=sq['manifold_fit']['weight_frobenius_norm'],
                       heldout_trajectories=sq['manifold_fit']['heldout_trajectories'],
                       cold_axis_points=sq['cold_axis_points'],
                       converged_quad=run['arms'][nq]['converged'], converged_lin=run['arms'][nl]['converged'],
                       best_found_quad=run['arms'][nq]['best_found'], best_found_lin=run['arms'][nl]['best_found'],
                       evolved_quad=run['arms'][nq]['evolved'], evolved_lin=run['arms'][nl]['evolved'],
                       gpu_ms_quad=run['arms'][nq]['gpu_ms'], gpu_ms_lin=run['arms'][nl]['gpu_ms']))
    out['quadratic_manifold'] = qm
    print('GATE3 quadratic manifold', json.dumps(qm), flush=True)

    # ---- gate 5: the cluster config declares every override name -------------
    out['cluster_config_declarations'] = {}
    for cname in ('config-256-qman.json',):
        c = json.loads((HERE / cname).read_text())
        sp = PN.declare_subjects(c, K, R, PN.resolve_rule_sets(c, int(c['intervals'])))
        names = [y['name'] for y in sp]
        ov, nf = list(c.get('priority_override', [])), len(c['fom_settings'])
        assert names[nf:nf + len(ov)] == ov, (cname, names, ov)
        out['cluster_config_declarations'][cname] = dict(subjects=len(sp), override=ov)
        print('GATE5', cname, len(sp), 'subjects', flush=True)

    out['runs'] = dict(smoke64=run)
    rp = scratch / 'report'
    subprocess.run([PY, str(HERE / 'reports/make_table.py'), str(au), '--out-dir', str(rp),
                    '--stem', 'smoke-panel'], check=True)
    assert (rp / 'smoke-panel-table.md').exists() and (rp / 'summary.json').exists()
    sm = json.loads((rp / 'summary.json').read_text())
    assert [e['rank'] for e in sm['ladder']] == list(cfg['qman_ranks'])
    assert all(e['quadratic_terms'] == QM.terms(e['rank']) for e in sm['ladder'])
    out['report'] = dict(table_bytes=(rp / 'smoke-panel-table.md').stat().st_size,
                         summary_rows=len(sm['rows']), ladder=sm['ladder'], answers=sm['answers'])
    out['total_seconds'] = time.perf_counter() - t_all
    out['deviation'] = 'one driver run with ~15 compiled subjects; exceeds the sub-minute rule (DESIGN A2)'
    out_json.write_text(json.dumps(out, indent=2) + '\n')
    print('SMOKE OK', round(out['total_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
