"""Local smoke for the panel: the real driver, twice, then the audit and the report generator.

    jaxrun python smoke_panel.py <out.json>

Gates, in order:
  1 `rule_operators` rebuilt from the consolidated fixture's own (nodes, weights) equals the
    fixture's G5 and Pq to 1e-13, and the driver's q = 0 certified-EQ arm at 64 intervals then
    reproduces the consolidated saved Burgers case (`expected.npz`, the panel's case 0) to
    <= 1e-12 relative in the output fields -- the audited baseline number of the parent lanes;
  2 every family (dense rom, EQ rom at two tolerances, POD, free bank, fast kernel, FOM grid)
    builds, warms up, times and audits at 64 intervals; the fast arm passes parity;
  3 the transfer path (support 64 -> 128, weights refit on reachable states, held-out rho)
    runs end to end at 128 intervals and the audit and report generator accept the output.
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
for sub in ('experiments/b-panel', 'experiments/b-panel/speed', 'experiments/q-ridge', 'experiments/b-ladder-top',
            'experiments/cheap-corrections', 'experiments/head-ablation', 'experiments/mr-burgers2d',
            'experiments/separable-decoder'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import topfix as TF            # noqa: E402
import panel as PN             # noqa: E402

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
    g5 = rel(ops['G5'], arch['G5'])
    pq = rel(ops['Pq'], arch['Pq'])
    out['rule_rebuild_vs_fixture'] = dict(G5_relative=g5, Pq_relative=pq, m=int(len(nodes0)))
    assert g5 <= 1e-13 and pq <= 1e-13, (g5, pq)
    print('GATE1a rule rebuild', g5, pq, flush=True)

    # ---- the q = 4 smoke rule: retained fitter on static codes at 64 intervals -
    C = np.load(inputs / 'directions_qtd02.npz')['C']
    head4 = TF.corrected_head(params, jnp.asarray(C[:, :4]), K)
    Zsub = Zold[::max(1, len(Zold) // 8192)]
    Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), 4))), axis=1)
    d4, i4 = TF.build_operators(bank, L, 80, 'eq', head=head4, Wcodes=Zaug, m=320, fitter='retained',
                                eq_seed=20259, candidate_cap=8192, fit_states=16)
    ops4 = PN.rule_operators(bank, e.modes(L, 80)[0], L, np.asarray(i4['eq_indices']), np.asarray(i4['eq_weights']))
    out['rule_rebuild_bitwise_q4'] = dict(G5=bool(np.array_equal(np.asarray(ops4['G5']), np.asarray(d4['G5']))),
                                          Pq=bool(np.array_equal(np.asarray(ops4['Pq']), np.asarray(d4['Pq']))),
                                          m=int(i4['m']))
    assert out['rule_rebuild_bitwise_q4']['G5'] and out['rule_rebuild_bitwise_q4']['Pq']
    print('GATE1b q4 rule rebuilt bitwise', flush=True)

    prov = dict(note='SMOKE inputs', files={})
    prov['files']['directions_qtd02.npz'] = real_prov['files']['directions_qtd02.npz']
    for q, nodes, w, M in ((0, nodes0, w0, 64), (4, np.asarray(i4['eq_indices']), np.asarray(i4['eq_weights']), 80)):
        fn = f'rule_q{q}_smoke_m{len(nodes)}.npz'
        np.savez_compressed(inputs / 'rules' / fn, nodes=nodes, weights=w)
        prov['files'][f'rules/{fn}'] = dict(
            sha256=LD.sha_file(inputs / 'rules' / fn), source_job='smoke', q=q, M=M, m=int(len(nodes)),
            relative_fit=float(setup['eq_relative_fit'] if q == 0 else i4['eq_relative_fit']), rho_max=None, rho_p95=None,
            rho_bar=0.116, certified_primary=False, certified_secondary=False,
            nodes_sha256=LD.sha_array(nodes), weights_sha256=LD.sha_array(w))
    (inputs / 'PROVENANCE.json').write_text(json.dumps(prov, indent=2) + '\n')

    # ---- gate 1c + 2: the driver at 64 intervals ---------------------------------
    env = dict(os.environ, JAX_ENABLE_X64='true', JAX_DEFAULT_MATMUL_PRECISION='highest',
               PYTHONPATH=':'.join(str(ROOT / s) for s in (
                   'experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/head-ablation',
                   'experiments/cheap-corrections', 'experiments/b-ladder-top', 'experiments/q-ridge',
                   'experiments/b-panel', 'experiments/b-panel/speed')))
    runs = {}
    for tag, cfgname in (('smoke64', 'config-smoke64.json'), ('smoke128', 'config-smoke128.json')):
        od = scratch / tag / 'output'
        t0 = time.perf_counter()
        reuse = os.environ.get('SMOKE_REUSE') == '1' and (od / 'COMPLETE').exists()
        if not reuse:
            subprocess.run([PY, str(HERE / 'panel.py'), '--config', str(HERE / cfgname), '--checkpoint', str(CK),
                        '--inputs', str(inputs), '--out', str(od)], check=True, env=env)
        res = json.loads((od / 'result.json').read_text())
        assert np.allclose(np.asarray(res['physical_cases'][0]), np.asarray(raw['physical_cases'][0]), rtol=0, atol=1e-15), 'case 0 is not the fixture case'
        assert res['complete'] and not res['dropped'], (tag, res['dropped'])
        runs[tag] = dict(seconds=time.perf_counter() - t0, reused_existing_output=bool(reuse), subjects=res['timed_subjects'],
                         gates={k: v.get('passed') for k, v in res['gates'].items()})
        print('DRIVER', tag, round(runs[tag]['seconds'], 1), len(res['timed_subjects']), 'subjects', flush=True)
        au = scratch / tag / 'audit.json'
        subprocess.run([PY, str(HERE / 'audit_panel.py'), str(od / 'result.json'), '--fields', str(od),
                        '--out', str(au), '--comparators', str(HERE / 'checks/comparators.json')], check=True)
        aj = json.loads(au.read_text())
        runs[tag]['audit_failed'] = aj['failed']
        runs[tag]['arms'] = {x['arm']: dict(evolved=x['worst_evolved_percent'], all=x['worst_all_times_percent'],
                                            gpu_ms=x['median_gpu_ms'], converged=x['converged'], strict=x['converged_strict'],
                                            admissible=x['admissible']) for x in aj['arms']}
        runs[tag]['nondominated_admissible_evolved'] = aj['nondominated']['gpu_evolved']['admissible']
        assert not aj['failed'], (tag, aj['failed'])
        if tag == 'smoke64':
            # DESIGN A5.1 plumbing: a second rule SET over the identical files must produce
            # bitwise identical arms. If it does not, the multi-set path perturbs a result.
            dup = {}
            for inv in res['invocations']:
                if '_eqdup_' in inv['name']:
                    twin = inv['name'].replace('_eqdup_', '_eqcert_')
                    other = next(y for y in res['invocations']
                                 if y['name'] == twin and y['case'] == inv['case'] and y['rep'] == inv['rep'])
                    dup[(inv['name'], inv['case'])] = dict(
                        same_field_hash=(inv['field_sha256'] == other['field_sha256']),
                        same_iterations=(inv['iterations'] == other['iterations']),
                        twin=twin)
            assert dup, 'the duplicate rule set produced no arms'
            assert all(v['same_field_hash'] and v['same_iterations'] for v in dup.values()), dup
            out['duplicate_rule_set_bitwise'] = dict(
                arms=len({k[0] for k in dup}), invocations=len(dup), all_bitwise=True)
            print('GATE4 duplicate rule set bitwise over', len(dup), 'invocations', flush=True)
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
            assert res['gates']['direct_reproduces_fft_tight']['passed'], res['gates']['direct_reproduces_fft_tight']
        else:
            out['transfer'] = [{k: v for k, v in t.items() if k in ('q', 'M', 'm', 'certification', 'basis', 'seconds')}
                               for t in res['transfer']]
    out['runs'] = runs
    rp = scratch / 'report'
    subprocess.run([PY, str(HERE / 'reports/generate_panel.py'), '--audit', str(scratch / 'smoke64/audit.json'),
                    str(scratch / 'smoke128/audit.json'), '--out-dir', str(rp), '--stem', 'smoke-panel'], check=True)
    assert (rp / 'smoke-panel.md').exists() and (rp / 'summary.json').exists()
    out['report'] = dict(md_bytes=(rp / 'smoke-panel.md').stat().st_size,
                         summary_rows=len(json.loads((rp / 'summary.json').read_text())['rows']))
    out['total_seconds'] = time.perf_counter() - t_all
    out['deviation'] = 'two driver runs; exceeds the sub-minute rule, recorded in DESIGN.md A2'
    out_json.write_text(json.dumps(out, indent=2) + '\n')
    print('SMOKE OK', round(out['total_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
