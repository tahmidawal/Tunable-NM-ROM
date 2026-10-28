"""Fixed dev6 cost gate for affine16 initialization of a shared-head64 solver.

Only this development cohort is generated. No reference or sealed data is read.
All timing repetitions execute and retain evidence for complete fused queries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/jcp-heads'))
import p1common as P
from p1common import Q, QS, T2
import hcommon as HC
import hcore as H
import harm
import stepcore as C
from affine import make_candidate, make_active_residual, INFO, DECISION_COLUMNS
import jax
import jax.numpy as jnp

ARMS = ('direct', 'affine')
SEED = 20261028
L, K, M, NSTEPS, REPS = 1024, 64, 256, 50, 7
DECISIONS = ('step_it', 'step_reason', 'it_sum', 'it_max', 'rej', 'exits',
             'nfail', 'ic_it', 'ic_reason', 'ic_idx')


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def timed(fn):
    t = time.perf_counter(); v = fn(); jax.block_until_ready(v)
    return time.perf_counter() - t, v


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def dump_npz(path, v, **extra):
    np.savez_compressed(path, W=v['W'], fields=v['fields'],
                        **{k: np.asarray(x) for k, x in v['stats'].items()}, **extra)


def relative_gap(x, y):
    return float(jnp.linalg.norm(x-y)/(1+jnp.linalg.norm(y)))


def make_guard_check(hf, nl):
    """Rebuild histories and the original predictor independently of the callback."""
    def check(states, codes, nu, data, sch):
        def one(k):
            zn, zp, zp2 = states[k], states[jnp.maximum(k-1, 0)], states[jnp.maximum(k-2, 0)]
            hist = (data['A']@hf(zn), data['A']@hf(zp), jnp.zeros(M))
            rc = T2.lspg_residual(nl, sch['co'], sch['dt'], nu, data, hist)
            warm, proposed = codes[k, :K], codes[k, K:]
            r0, r1 = rc(hf(warm)), rc(hf(proposed))
            ze = zn+(zn-zp)
            zqd = jnp.where(k >= 2, 3.*zn-3.*zp+zp2, ze)
            candidates = jnp.stack((zn, ze, zqd))
            norms = jax.vmap(lambda z: jnp.linalg.norm(rc(hf(z))))(candidates)
            predictor = candidates[jnp.argmin(jnp.where(jnp.isfinite(norms), norms, jnp.inf))]
            return jnp.array([jnp.vdot(r0, r0), jnp.vdot(r1, r1),
                jnp.max(jnp.abs(proposed[16:]-warm[16:])), jnp.max(jnp.abs(warm-predictor))])
        return jax.lax.map(one, jnp.arange(NSTEPS))
    return jax.jit(check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    P.check_env({}); started = time.perf_counter()
    model_dir = Path(a.model_dir); model_path = model_dir/'MODEL.json'
    manifest = json.loads(model_path.read_text())
    assert manifest['schema'] == 1 and manifest['complete'] is True
    assert manifest['training']['steps'] == 200000 and manifest['training']['prefixes'] == [16, 32, 64]
    assert manifest['training']['seed'] == SEED
    assert manifest['source_ckpt_sha256'] == HC.sha_file(T2.CKPT)
    for width in (16, 32, 64):
        desc = manifest['heads'][str(width)]
        assert desc['file'] == f'h{width}.npz' and desc['k'] == width
        assert HC.sha_file(model_dir/desc['file']) == desc['sha256']
    rep = dict(complete=False, scope='dev6 affine16-coordinate initialization cost screen; no multigrid',
        commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(), gpu=P.gpu_info(), jax_version=jax.__version__,
        precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), x64=bool(jax.config.jax_enable_x64),
        model=manifest, model_manifest_sha256=HC.sha_file(model_path),
        settings=dict(L=L, nodes_per_axis=L+1, fine_K=K, active_K=16, fixed_tail=48, bank_R=512,
            M=M, mode_policy='frozen1024', rule='gauss128', quadrature_points=128**2,
            scheme='BE', dt=.005, steps=NSTEPS, gtol=1e-3, tolf=1e-9,
            aux_budget=600, aux_gtol=1e-3, aux_initial_damping=1e-6,
            fine_budget=600, initial_budget=400, initial_gtol=1e-6,
            repetitions=REPS, order_seed=SEED, guard_relative_decrease=1e-8),
        records=[], timings=[], parity=[], diagnostics=[], setup={}, checks={}, seed_info_columns=INFO)
    def save():
        tmp = out/'result.json.tmp'; tmp.write_text(json.dumps(clean(rep), indent=2, allow_nan=False)+'\n')
        tmp.replace(out/'result.json')
    save(); t = time.perf_counter()
    bank = P.Bank(L, [0, 128, 256, 384, 512])
    heads = {k: HC.Head(model_dir/f'h{k}.npz', bank.mdl) for k in (16, 32, 64)}
    for width, head in heads.items():
        assert head.K == width and len(head.Zsub) == 8192
        assert np.isfinite(head.trust) and head.trust > 0 and 'hB' not in head.params
        assert head.cfg['jcp_heads_steps'] == 200000
        assert head.trust == manifest['heads'][str(width)]['trust']
    f, c = heads[64], heads[16]
    # Structural identity establishes the common map for every latent vector;
    # the numerical field/derivative checks below test the implemented folding.
    for width in (16, 32):
        head = heads[width]
        assert len(head.params['h']) == len(f.params['h'])
        np.testing.assert_array_equal(head.params['h'][0][0], f.params['h'][0][0][:width])
        np.testing.assert_array_equal(head.params['h'][0][1], f.params['h'][0][1])
        for layer, fine_layer in zip(head.params['h'][1:], f.params['h'][1:]):
            for param, fine_param in zip(layer, fine_layer):
                np.testing.assert_array_equal(param, fine_param)
        np.testing.assert_array_equal(head.params['h_lin'], f.params['h_lin'][:width])
        np.testing.assert_array_equal(head.Zsub, f.Zsub[:, :width])
    rep['checks']['shared_prefix_parameters'] = dict(passed=True)
    fine = HC.HeadSetting(bank, 'shared64', f, 'gauss128', M=M, nrec=51, nstep=NSTEPS)
    fine.check('main', bank); d = fine.data['main']
    assert d['A'].shape == (M, 512) and d['Gq'].shape == (128**2, 512)
    hf = H.head_fn(f.pf)
    nl, nlJ = Q.tested_value('point', L), Q.tested_jac('point', L)
    seed = make_candidate(hf, nl, nlJ, c.trust)
    qrec = {'direct': fine.query(record=True),
        'affine': C.make_head_query(f.pf, K, L, f.trust, nrec=51, nstep=NSTEPS, seed=seed)}
    qtim = {'direct': fine.query(record=False),
        'affine': C.make_head_query(f.pf, K, L, f.trust, nrec=0, nstep=NSTEPS, seed=seed)}
    qnone = C.make_head_query(f.pf, K, L, f.trust, nrec=51, nstep=NSTEPS)
    qfail = C.make_head_query(f.pf, K, L, f.trust, nrec=51, nstep=NSTEPS,
        seed=make_candidate(hf, nl, nlJ, c.trust, force_failure=True))
    jax.block_until_ready(d)
    rep['setup'] = dict(seconds=time.perf_counter()-t, heads={str(k):h.sha256 for k,h in heads.items()},
        trust={str(k):h.trust for k,h in heads.items()}, memory=P.mem())
    rep['table_meta'] = fine.meta
    rep['table_hashes'] = {k: sha(d[k]) for k in ('A', 'lam', 'Gq', 'Gs', 'Psi')}
    rep['bank_sha256'] = HC.sha_file(T2.CKPT); rep['rotation_sha256'] = HC.sha_file(Q.ROTATION)
    cases = Q.cohort('dev6'); assert cases.shape == (6, 5)
    U0 = [jnp.asarray(Q.e.initial(L, ph)) for ph in cases]
    rep['cases'] = cases; rep['seeds'] = dict(dev4=7090702, dev2=911702, order=SEED)
    rep['initial_hashes'] = [sha(u) for u in U0]
    sch = T2.sched('BE', .005, 1e-3, 1e-9)
    def call(q, ci):
        return q['fused'](U0[ci], float(cases[ci, 4]), d, fine.cold, sch)

    # Check transfer in coefficients, represented fields and spatial derivatives.
    X = np.asarray([[.07,.13], [.23,.39], [.51,.42], [.77,.68], [.91,.84]])
    B, Bx, By = bank.mdl.values_grads(X, 512)
    rep['checks']['exact_prefix'] = []
    for width in (16, 32):
        hh = H.head_fn(heads[width].pf)
        z = jnp.asarray(heads[width].Zsub[[0, 100, 1000]])
        small = jax.vmap(hh)(z); big = jax.vmap(hf)(jnp.pad(z, ((0, 0), (0, K-width))))
        gaps = {'coefficients': relative_gap(small, big)}
        for name, basis in [('field', B), ('dx', Bx), ('dy', By)]:
            gaps[name] = relative_gap(small@basis.T, big@basis.T)
        assert max(gaps.values()) < 1e-11
        rep['checks']['exact_prefix'].append(dict(width=width, gaps=gaps, passed=True))
    z0 = jnp.pad(jnp.asarray(c.Zsub[0]), (0, 48))
    Jexp = jax.jacfwd(hf)(z0)[:, 16:]
    norms = np.asarray(jnp.linalg.norm(Jexp, axis=0))
    assert np.all(np.isfinite(norms)) and np.all(norms > 1e-12)
    rep['checks']['expansion'] = dict(column_norms=norms, passed=True)
    active_res, active_ev = make_active_residual(hf, nl, nlJ)
    ad_checks = []
    for z in (jnp.asarray(f.Zsub[0]), jnp.asarray(f.Zsub[101])):
        cz = hf(z); hist = (d['A']@cz, d['A']@cz, jnp.zeros(M))
        args = (hist, float(cases[0, 4]), d, sch['co'], sch['dt'])
        rv, J = active_ev(z[:16], (args, z[16:]))
        Ja = jax.jacfwd(active_res)(z[:16], (args, z[16:]))
        gap = relative_gap(J, Ja)
        assert J.shape == (M, 16) and gap < 1e-11
        np.testing.assert_allclose(rv, qrec['direct']['res'](z, args), rtol=1e-11, atol=1e-12)
        ad_checks.append(dict(relative_gap=gap, columns=int(J.shape[1]), passed=True))
    rep['checks']['active_jacobian'] = ad_checks
    rep['compile_warm_seconds'] = {}
    for arm in ARMS:
        for label, q in [('recorded', qrec[arm]), ('timed', qtim[arm])]:
            dur, _ = timed(lambda: call(q, 0)); rep['compile_warm_seconds'][arm+'_'+label] = dur
    print('SETUP_PASS', round(time.perf_counter()-started, 3), flush=True); save()
    cert = H.make_head_certifier(hf, nl); guard_check = make_guard_check(hf, nl)
    records, witnesses = {}, {}
    for ci in range(6):
        for arm in ARMS:
            dur, v = timed(lambda: call(qrec[arm], ci)); records[(arm, ci)] = v
            st = v['stats']; states = st['states']
            _, (g, rn) = timed(lambda: cert(states, float(cases[ci, 4]), d, sch))
            _, scale, _ = qrec['direct']['init'](U0[ci], fine.cold)
            fresh = bool(np.all(np.isfinite(g)) and np.all((np.asarray(g)<=1e-3)|(np.asarray(rn)<=float(scale)*1e-9)))
            agree = P.gcert_ok(np.asarray(g), np.asarray(st['step_gn']))
            np.testing.assert_allclose(st['step_tolratio'], np.asarray(rn)/(float(scale)*1e-9+1e-300), rtol=1e-7, atol=1e-12)
            valid = bool(harm.run_valid(v) and fresh and agree and int(st['nfail']) == 0)
            info = np.asarray(st.get('seed_info', np.zeros((NSTEPS, 16))))
            entry = dict(case=ci, arm=arm, valid=valid, fresh_stationarity=fresh, gradient_agreement=agree,
                gamma_max=float(np.max(g)), recorded_seconds=dur, initial_scale=float(scale),
                fine_corrections=int(st['it_sum']), fine_J=NSTEPS+int(st['it_sum']),
                fine_rejections=int(st['rej']), fine_guard_residuals=int(info[:, 10].sum()),
                auxiliary_sums=dict(zip(INFO, info.sum(axis=0).tolist())),
                auxiliary_failures=int(np.sum(~np.isin(info[:, 2], [1, 4]))) if arm == 'affine' else 0,
                coarse_J=int(info[:, 9].sum()))
            if arm == 'affine':
                diag = np.asarray(guard_check(states, st['seed_codes'], float(cases[ci, 4]), d, sch))
                accepted = info[:, 0].astype(bool)
                assert np.all(diag[:, 2] == 0.) and np.max(diag[:, 3]) <= 1e-12
                np.testing.assert_allclose(diag[:, :2], info[:, 7:9], rtol=1e-7, atol=1e-12, equal_nan=True)
                assert np.all(np.isfinite(diag[accepted, :2]))
                assert np.all(diag[accepted, 1] < diag[accepted, 0])
                assert np.all(diag[accepted, 1] <= (1-1e-8)*diag[accepted, 0])
                assert np.all(np.isin(info[accepted, 2], [1, 4]))
                assert np.all(info[:, 1] <= 600)
                np.savez_compressed(out/f'diagnostics_affine_case{ci}.npz', objectives=diag[:, :2],
                    tail_displacements=diag[:, 2], predictor_gaps=diag[:, 3], accepted=accepted)
                rep['diagnostics'].append(dict(case=ci, passed=True, accepted=int(accepted.sum()),
                    tail_max_abs=float(np.max(diag[:, 2])), predictor_max_abs=float(np.max(diag[:, 3]))))
            rep['records'].append(entry)
            dump_npz(out/f'{arm}_case{ci}.npz', v, gamma_fresh=g, residual_norm_fresh=rn)
            _, vt = timed(lambda: call(qtim[arm], ci))
            np.testing.assert_allclose(vt['W'], v['W'], rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(vt['fields'], v['fields'], rtol=1e-12, atol=1e-13)
            assert P.gcert_ok(np.asarray(g), np.asarray(vt['stats']['step_gn']))
            np.testing.assert_allclose(vt['stats']['step_tolratio'], np.asarray(rn)/(float(scale)*1e-9+1e-300), rtol=1e-7, atol=1e-12)
            entry['timed_production_stationarity'] = bool(int(vt['stats']['nfail']) == 0 and
                np.all((np.asarray(vt['stats']['step_gn'])<=1e-3)|(np.asarray(vt['stats']['step_tolratio'])<=1)))
            entry['valid'] = entry['valid'] and entry['timed_production_stationarity']
            for k in DECISIONS:
                assert np.array_equal(vt['stats'][k], st[k])
            if arm == 'affine':
                assert np.array_equal(np.asarray(vt['stats']['seed_info'])[:, DECISION_COLUMNS], info[:, DECISION_COLUMNS])
                np.testing.assert_allclose(vt['stats']['seed_info'], info, rtol=1e-12, atol=1e-13, equal_nan=True)
            entry['timed_certificate_matches'] = True
            dump_npz(out/f'timed_identity_{arm}_case{ci}.npz', vt)
            witnesses[(arm, ci)] = {k: np.asarray(vt[k]) for k in ('W', 'fields')}
            witnesses[(arm, ci)]['stats'] = {k: np.asarray(x) for k, x in vt['stats'].items()}
        vd = records[('direct', ci)]
        for label, q in [('disabled', qnone)]+([('forced_failure', qfail)] if ci == 0 else []):
            _, vv = timed(lambda: call(q, ci))
            err = float(jnp.max(jnp.abs(vv['stats']['states']-vd['stats']['states'])))
            decisions = all(np.array_equal(vv['stats'][k], vd['stats'][k]) for k in DECISIONS)
            fields_equal = bool(np.allclose(vv['fields'], vd['fields'], rtol=1e-12, atol=1e-13))
            ok = err <= 1e-12 and decisions and fields_equal
            rep['parity'].append(dict(case=ci, check=label, state_max_abs=err,
                decisions_equal=decisions, fields_equal=fields_equal, passed=ok))
            assert ok
            if label == 'forced_failure':
                assert np.all(np.asarray(vv['stats']['seed_info'])[:, 2] == 3)
                assert not np.any(np.asarray(vv['stats']['seed_info'])[:, 0])
        print('RECORDED', ci, [(x['arm'], x['fine_corrections'], x['valid']) for x in rep['records'][-2:]], flush=True); save()
    rng = np.random.default_rng(SEED)
    QS.burn(.1)
    for block in range(REPS):
        for ci_ in rng.permutation(6):
            ci = int(ci_)
            for ai in rng.permutation(len(ARMS)):
                arm = ARMS[ai]; QS.burn(.1)
                elapsed, v = timed(lambda: call(qtim[arm], ci))
                want = witnesses[(arm, ci)]
                identity = all(np.array_equal(np.asarray(v[k]), want[k]) for k in ('W', 'fields')) and all(
                    np.array_equal(v['stats'][k], x, equal_nan=True) for k, x in want['stats'].items())
                path = f'timed_identity_{arm}_case{ci}.npz'
                if not identity:
                    path = f'timed_variant_{arm}_case{ci}_block{block}.npz'; dump_npz(out/path, v)
                rep['timings'].append(dict(block=block, case=ci, arm=arm, seconds=elapsed, identity=identity,
                    W_sha256=sha(v['W']), fields_sha256=sha(v['fields']), array_witness=path,
                    fine_corrections=int(v['stats']['it_sum']),
                    online_valid=bool(harm.run_valid(v) and int(v['stats']['nfail']) == 0)))
        print('TIMING_BLOCK', block, flush=True); save()
    arr = {(arm, ci): np.array([t['seconds'] for t in rep['timings'] if t['arm'] == arm and t['case'] == ci])
        for arm in ARMS for ci in range(6)}
    ratios = [float(np.median(arr[('affine', ci)]/arr[('direct', ci)])) for ci in range(6)]
    valid = all(x['valid'] for x in rep['records'])
    identity = all(x['identity'] and x['online_valid'] for x in rep['timings'])
    controls = all(x['passed'] for x in rep['parity']) and all(x['passed'] for x in rep['diagnostics'])
    rep['gate'] = dict(per_case_paired_median_ratios=ratios, median_ratio=float(np.median(ratios)),
        maximum_case_ratio=max(ratios), all_numerically_valid=valid, all_timing_identities=identity,
        all_controls=controls, passed=bool(valid and identity and controls and np.median(ratios)<=.85 and max(ratios)<=1.10))
    rep['continue_accuracy'] = rep['gate']['passed']
    rep['strong_accuracy_checks'] = 'not run; cost pass requires a separately reviewed tight/tighter and quadrature stage'
    rep['complete'] = True; rep['elapsed_program_seconds'] = time.perf_counter()-started; rep['memory'] = P.mem(); save()
    print('FINAL', json.dumps(clean(rep['gate'])), flush=True)


if __name__ == '__main__': main()
