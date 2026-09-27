"""burgers-compare-hires: the Burgers 2D comparison panel at one target mesh, in ONE allocation (DESIGN.md).

Adapted from experiments/burgers-repanel/repanel.py @ 219feb6e (itself burgers-eqcert/eqcert.py @ 176b2a9a ->
hires-burgers/hires.py @ 0ab60014). The NM-ROM arms, the rules, the Phi-free operators, the FOM grid and the
output contract are that driver's, unchanged; the audited pre-optimisation (`base`/`l4`) arms, the parity chain,
the certificate phases and the refined-reference phase are removed (this lane re-times frozen, already-audited
settings), and three things are added:

  * Phase S -- POD-LSPG and the quadratic manifold, fitted in-job from training snapshots at this mesh with the
    streamed fits of `sfit.py` (== ablation.pod_basis / qman.fit, trajectory-split ridge), built and run by
    `arms` exactly as the 256^2 panel (qmn102) ran them, plus the robust-rule NM-ROM arm. Slow arms run ONE AT A
    TIME in their own timed block, never inside the randomised fast panel (repanel's order-effect finding).
  * Phase B -- a bracket re-timing of every Phase F subject after Phase S, for the drift gate.
  * every arm's six full fields are saved for every case (`full_<arm>_case<c>.npy`) so the NumPy audit
    recomputes every error from full fields; the operator cohort (dev6 truth in the operator contract) is written
    for the PyTorch phase that follows in the same allocation.

Phases, each written incrementally to result.json:
  0 cohort + same-grid reference fft_tight           1 bank, Phi-free operators (parity-gated at `parity_mesh`)
  2 Phase F quick runs + randomised timed panel       3 Phase S (slow arms, one block each)
  4 Phase B bracket                                   5 operator cohort
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import ladder as LD
import topfix as TF
import hops as H
import hfast as HF
import xfast as XF
import qman as QM
from ablation import radius
import sfit
import gridarm

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump
TIMES = np.array([0., .05, .1, .15, .2, .25])


def gt(g):
    return f'g{g:g}'.replace('-', 'm').replace('.', 'p')


def is_oom(exc):
    s = str(exc)
    return 'RESOURCE_EXHAUSTED' in s or 'out of memory' in s.lower() or 'Out of memory' in s


def rel_per_time(f, truth, n0):
    return [float(np.linalg.norm(a - b)) / n0 for a, b in zip(f, truth)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fields_dir = out / 'fields'
    fields_dir.mkdir(exist_ok=True)
    inputs = Path(a.inputs)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:                               # noqa: BLE001
        smi = []

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L, dt = int(cfg['intervals']), cfg['dt']
    st = cfg['strict']
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), x64=True,
               matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L, dt=dt,
               weights_frozen=True, final_cohort_unopened=True, output_times=TIMES.tolist(),
               timing_contract=('gpu_seconds: supplied dense initial field resident on the GPU to six dense GPU '
                                'output fields, block_until_ready on both sides; host_seconds: the same invocation '
                                'plus the host upload of the input and the host copy of the six outputs; identical '
                                'for every JAX subject'),
               rule_status=cfg.get('rule_status', {}), rules=[], quick=[], arm_setup=[], invocations=[],
               snapshots={}, pod={}, quadratic_manifold=[], dropped=[], gates={}, phases={}, complete=False)

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x

    save = lambda: dump(out / 'result.json', clean(rep))
    save()

    # ------------------------------------------------------------------ cohort --
    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    rep['cohort_name'] = 'dev6: params_draw(7090702,4) + params_draw(911702,2), opened development cases'
    rep['physical_sha256'] = sha_array(physical)
    want = cfg.get('expected_physical_sha256')
    rep['gates']['evaluation_cohort_matches_b_panel'] = dict(expected=want, got=rep['physical_sha256'],
                                                             passed=(None if want is None else want == rep['physical_sha256']))
    assert want is None or want == rep['physical_sha256'] or cfg.get('local_smoke_waives_cohort_hash'), 'cohort differs from b-panel'
    if cfg.get('case_subset') is not None:
        physical = physical[[int(i) for i in cfg['case_subset']]]
    ncase = len(physical)
    rep['physical_cases'] = physical.tolist()
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    rep['train_physical_sha256'] = sha_array(train_physical)
    inputs_u = [e.initial(L, ph) for ph in physical]
    n0 = [float(np.linalg.norm(u)) for u in inputs_u]

    bank = A.CoordBank(params, K, R)
    t0 = time.perf_counter()
    G = H.build_bank(bank, L, nblocks=cfg.get('bank_blocks'))
    rep['phases']['bank'] = dict(shape=[H.bank_rows(G), R], blocks=len(G), seconds=time.perf_counter() - t0,
                                 bytes=int(sum(g.nbytes for g in G)))
    print('BANK', H.bank_rows(G), 'rows in', len(G), 'blocks', el(), flush=True)

    # ---------------------------------------------------- full-order solvers ----
    foms = {}

    def fom_for(fs):
        key = (L, fs['dt'], fs.get('impl', 'audited'))
        if key not in foms:
            if key[2] == 'audited':
                foms[key] = ip.make_fom(L, fs['dt'], 'fft')
            else:
                foms[key] = (e.make_fom(L, fs['dt'], target=L)[0], None)
        return foms[key]

    def run_fom(fs, u, nu):
        fn, pre = fom_for(fs)
        if pre is None:
            return fn(u, nu, fs['ntol'], fs['ltol'])
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    tight = next(fs for fs in cfg['fom_settings'] if fs['name'] == cfg['same_grid_reference'])
    truth = {}
    for c in range(ncase):
        t0 = time.perf_counter()
        v = run_fom(tight, jnp.asarray(inputs_u[c]), float(physical[c, 4]))
        jax.block_until_ready(v)
        f = np.asarray(v[0])
        rn = np.asarray(v[2])
        assert np.isfinite(f).all() and np.isfinite(rn).all() and rn.max() <= tight['ntol'] * (1 + 1e-9), ('truth not converged', c)
        truth[c] = f
        rep['phases'].setdefault('truth', []).append(dict(case=c, seconds=time.perf_counter() - t0,
                                                          max_relative_residual=float(rn.max()),
                                                          converged=bool(rn.max() <= tight['ntol'] * (1 + 1e-9)),
                                                          newton_total=int(np.sum(np.asarray(v[1]))),
                                                          field_sha256=sha_array(f)))
        print('TRUTH', c, el(), flush=True)
    save()

    # ===== operator cohort (Phase O input), written at once: it depends only on the truth solve (audit F3)
    cohort = out / 'opcohort'
    cohort.mkdir(exist_ok=True)
    records = []
    for c in range(ncase):
        target = truth[c][:, None].astype(np.float64)
        path = cohort / f'burgers-dev-{c:05d}.npz'
        np.savez(path, input=np.ascontiguousarray(target[0]), target=target,
                 parameters=np.array([float(physical[c, 4])]), times=TIMES)
        records.append(dict(case_id=f'burgers-dev-{c:05d}', split='development', case_index=c,
                            seed=int(cfg['eval_seed'] if c < cfg['eval_cases'] else cfg['eval_fresh_seed']) * 100 + c,
                            path=path.name, mesh=L, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            generation_descriptors=dict(zip(('cx', 'cy', 'width', 'amplitude', 'nu'), map(float, physical[c])))))
    (cohort / 'index.json').write_text(json.dumps(dict(
        schema_version=1, pde='burgers', split='development', count=len(records), mesh=L, complete=True,
        derivation=f'dev6 with this job\'s own same-grid fft_tight solve at {L} intervals, in the operator contract',
        records=records), indent=1) + '\n')
    def score(f, c):
        sg = rel_per_time(f, truth[c], n0[c])
        return dict(same_grid_per_time=sg, same_grid_all=float(max(sg)), same_grid_evolved=float(max(sg[1:])),
                    t0_compression=float(sg[0]))

    # -------------------------------------------------------- directions, colds --
    dfile = inputs / cfg['directions_file']
    Cnp = np.ascontiguousarray(np.load(dfile)['C'])
    rep['directions'] = dict(file=cfg['directions_file'], sha256=sha_file(dfile), expected=cfg.get('directions_sha256'))
    assert cfg.get('directions_sha256') in (None, rep['directions']['sha256'])
    Cfull = jnp.asarray(Cnp)
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    Zsub = np.asarray(Zold[::max(1, len(Zold) // cfg['decoder_code_subsample'])])
    colds, heads = {}, {}

    def cold_for(q):
        if q not in colds:
            heads[q] = TF.corrected_head(params, Cfull[:, :q], K)
            colds[q] = A.build_cold(bank, heads[q], np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1),
                                    cfg['cold_axis_points'])[0]
        return colds[q], heads[q]

    # ------------------------------------- operators without Phi, parity-gated ----
    ops_M = {}

    def operators(M):
        if M not in ops_M:
            kx, ky, lam = H.modes_lean(L, M)
            sx, sy = H.sine_tables(L, kx, ky)
            sxd, syd = jnp.asarray(sx), jnp.asarray(sy)
            ops_M[M] = dict(kx=kx, ky=ky, lam=jnp.asarray(lam), sx=sxd, sy=syd, A=H.project_bank(G, sxd, syd, L))
            jax.block_until_ready(ops_M[M]['A'])
        return ops_M[M]

    Lp = int(cfg.get('parity_mesh', L))
    Gp = bank.on_grid(Lp) if Lp != L else jnp.concatenate(G, 0)
    pg = []
    for Mg in sorted({r['M'] for r in cfg['rungs']}):
        Phig, lamg, _ = e.modes(Lp, Mg)
        kxg, kyg, lamg2 = H.modes_lean(Lp, Mg)
        sxg, syg = H.sine_tables(Lp, kxg, kyg)
        Ag = H.project_bank((Gp,), jnp.asarray(sxg), jnp.asarray(syg), Lp)
        Aref = jnp.asarray(Phig).T @ Gp
        pick = np.arange(0, (Lp - 1) ** 2, 97)
        g0 = dict(mesh=Lp, M=Mg, lam_identical=bool(np.array_equal(lamg, lamg2)),
                  A_relative=float(jnp.linalg.norm(Ag - Aref) / jnp.linalg.norm(Aref)),
                  rows_max_abs=float(np.max(np.abs(H.phi_rows(Lp, kxg, kyg, H.unravel_nodes(pick, Lp)) - Phig[pick]))))
        g0['passed'] = bool(g0['lam_identical'] and g0['A_relative'] <= 1e-12 and g0['rows_max_abs'] <= 1e-14)
        pg.append(g0)
        del Phig, Aref, Ag
    del Gp
    rep['gates']['phi_free_operator_parity'] = dict(passed=all(x['passed'] for x in pg), per_M=pg)
    assert rep['gates']['phi_free_operator_parity']['passed'], pg
    save()

    # ----------------------------------------------------------- rules, arms ----
    rules, built, subjects = {}, {}, []
    lin = lambda d: 'gj' if d <= cfg['gauss_jordan_max'] else 'lu'

    def build_rule(rung, rs):
        q, M, name = rung['q'], rung['M'], rs['name']
        if (q, M, name) in rules:
            return
        o = operators(M)
        t0 = time.perf_counter()
        part = rs['parts'][0]
        assert len(rs['parts']) == 1
        if 'lattice' in part:
            s_ = int(part['lattice'])
            ij, w = H.lattice_rule(L, s_)
            src = dict(kind='lattice', s=s_, m=int(len(ij)))
        else:
            z = np.load(inputs / part['file'])
            assert sha_file(inputs / part['file']) == part['sha256'], part['file']
            ij = H.transfer_nodes(z['nodes'].astype(int), part['mesh'], L)
            w = np.asarray(z['weights'], float) * (L / part['mesh']) ** 2
            src = dict(kind='transfer', file=part['file'], sha256=part['sha256'], source_mesh=part['mesh'],
                       source_m=int(len(ij)), source_status=part.get('status'))
        _, uniq = np.unique(H.ravel_nodes(ij, L), return_index=True)
        ij, w = ij[np.sort(uniq)], w[np.sort(uniq)]
        ops, ij, w = H.rule_ops(bank, L, o['kx'], o['ky'], ij, w)
        np.savez_compressed(out / f'rule_L{L}_q{q}_M{M}_{name}.npz', ij=ij, weights=w)
        info = dict(q=q, M=M, rule=name, m=int(len(ij)), source=src, build_seconds=time.perf_counter() - t0)
        rules[(q, M, name)] = (ops, info)
        rep['rules'].append(info)

    def add(name, **kw):
        built[name] = dict(name=name, **kw)
        rep['arm_setup'].append({k: v for k, v in kw.items() if k in (
            'family', 'phase', 'q', 'M', 'm', 'k', 'rule', 'gtol', 'kernel', 'solver', 'setting', 'quadrature',
            'variant', 'exact_steps', 'unknowns', 'fit', 'manifold_fit', 'bank_columns', 'cold_axis_points',
            'trust_radius', 'array_bytes', 'rule_status')} | dict(arm=name))
        return name

    def register(rung, rs, phase):
        """The optimised NM-ROM arms of one (rung, rule) -- repanel.register without the base/l4 twins."""
        q, M = rung['q'], rung['M']
        o = operators(M)
        cold, head = cold_for(q)
        C = Cfull[:, :q]
        ops, info = rules[(q, M, rs['name'])]
        data = dict(A=o['A'], lam=o['lam'], G=G, G5=ops['G5'], Pq=ops['Pq'], sx=o['sx'], sy=o['sy'])
        tab = HF.build_tables(params, C, K, data, cold)
        tag = f"q{q}_M{M}_{rs['name']}"
        d = K + q
        chunks = next(k for k in range(1, d + 1) if d % k == 0 and d // k <= cfg['dense_tangent_group'])
        names = []
        for var in rs['variants']:
            for g in var['gtols']:
                solver = var.get('solver', 'lu')
                j = int(var.get('exact_steps', 0))
                sfx = ''.join(f'_{k_}' for k_ in ([solver] if solver != 'lu' else []) +
                              [k_ for k_ in ('clip', 'lamcarry', 'pred2') if var.get(k_)])
                if j:
                    sfx += f'_x{j}'
                fq, parts = XF.make_query(params, C, K, q, L, dt, trust, 'eq', exact_steps=j,
                                          ic_budget=st['ic_budget'], step_budget=st['step_budget'], gtol=g,
                                          ic_gtol=var.get('ic_gtol', cfg['ic_gtol']), ridge=cfg['inner_damping'],
                                          solver=solver, tangent_chunks=chunks, parts=True, clip=bool(var.get('clip')),
                                          lam_carry=bool(var.get('lamcarry')),
                                          predictor='quad' if var.get('pred2') else 'lin')
                names.append(add(f'{tag}_{gt(g)}_fast{sfx}', family='nmrom', kind='rom', phase=phase, q=q, M=M,
                                 m=info['m'], rule=rs['name'], gtol=g, unknowns=K + q,
                                 kernel=(f'xfast EQ, first {j} step(s) exact' if j else 'hfast EQ'),
                                 solver=solver, quadrature='eq', data=data, cold=cold, variant=var, exact_steps=j,
                                 rule_status=rs.get('status'),
                                 query=(lambda u, nu, d_, c, _t=tab, _f=fq: _f(u, nu, d_, c, _t))))
        return names

    def invoke(b, u, c):
        nu = float(physical[c, 4])
        if b['kind'] == 'fom':
            return run_fom(b['setting'], u, nu)
        return b['query'](u, nu, b['data'], b['cold'])

    def rom_row(b, v):
        reasons = np.asarray(v[3]).tolist()
        it = np.asarray(v[1])
        gj = np.asarray(v[12] if len(v) > 12 else v[8], float)
        row = dict(stop_reasons=reasons, iterations=it.tolist(), total_iterations=int(it.sum()),
                   median_iterations=float(np.median(it)), max_iterations=int(it.max()), ic_iterations=int(v[5]),
                   ic_reason=int(v[6]), budget_exits=int(sum(r == 0 for r in reasons)),
                   rejected_exits=int(sum(r == 3 for r in reasons)), residual_exits=int(sum(r == 1 for r in reasons)),
                   tiny_step_exits=int(sum(r == 2 for r in reasons)), gradient_exits=int(sum(r == 4 for r in reasons)),
                   worst_joint_stationarity=float(np.max(gj)) if np.isfinite(gj).all() else None,
                   stalled_exits=int(sum(r in (0, 2, 3) for r in reasons)))
        return row

    def fom_row(b, v):
        fs = b['setting']
        rn = np.asarray(v[2])
        okstep = np.isfinite(rn) & (rn <= fs['ntol'] * (1 + 1e-9))
        return dict(dt=fs['dt'], ntol=fs['ntol'], ltol=fs['ltol'], impl=fs.get('impl', 'audited'),
                    newton_iterations_total=int(np.sum(np.asarray(v[1]))), steps=int(len(rn)),
                    stalled_steps=int(np.sum(~okstep)), nonlinear_converged=bool(okstep.all()),
                    max_relative_residual=float(np.nanmax(rn)))

    quick_sha = {}
    quick_seconds = {}

    def run_quick(name):
        """One untimed query per case; the saved full fields are what the audit scores. False on OOM (dropped)."""
        b = built[name]
        t0 = time.perf_counter()
        rows, ct = [], []
        try:
            for c in range(ncase):
                tc = time.perf_counter()
                v = invoke(b, jnp.asarray(inputs_u[c]), c)
                jax.block_until_ready(v)
                f = np.asarray(v[0])
                assert np.isfinite(f).all(), name
                vh = (None,) + tuple(host(v[1:]))
                row = dict(name=name, case=c, family=b['family'], phase=b['phase'], field_sha256=sha_array(f),
                           **score(f, c))
                row.update(rom_row(b, vh) if b['kind'] == 'rom' else fom_row(b, vh))
                rows.append(row)
                quick_sha[(name, c)] = row['field_sha256']
                np.save(fields_dir / f'full_{name}_case{c}.npy', f)
                ct.append(time.perf_counter() - tc)
                del v, f, vh
        except Exception as exc:                     # noqa: BLE001
            if not is_oom(exc):
                raise
            rep['dropped'].append(dict(name=name, phase='quick', reason=str(exc)[:400]))
            print('DROPPED (OOM)', name, str(exc)[:200], flush=True)
            built.pop(name, None)
            jax.clear_caches()
            gc.collect()
            save()
            return False
        rep['quick'].extend(rows)
        quick_seconds[name] = float(np.median(ct[1:] if len(ct) > 1 else ct)) * ncase   # compile excluded
        subjects.append(name)
        print('QUICK', name, 'evolved%', round(100 * max(x['same_grid_evolved'] for x in rows), 4),
              'stalled', sum(x.get('stalled_exits', x.get('stalled_steps', 0)) for x in rows),
              round(time.perf_counter() - t0, 1), el(), flush=True)
        save()
        return True

    seq = [0]
    last = dict(name=None, gpu_seconds=None)

    def timed_call(name, c, r_, phase):
        b = built[name]
        e.burn(cfg['burn_seconds'])
        ht = time.perf_counter()
        u = jax.device_put(np.array(inputs_u[c], copy=True))
        jax.block_until_ready(u)
        g0_ = time.perf_counter()
        v = invoke(b, u, c)
        jax.block_until_ready(v)
        gs = time.perf_counter() - g0_
        f = np.asarray(v[0])
        hs = time.perf_counter() - ht
        same = sha_array(f) == quick_sha[(name, c)]
        row = dict(name=name, case=c, rep=r_, phase=phase, family=b['family'], seq=seq[0], gpu_seconds=gs,
                   host_seconds=hs, previous_name=last['name'], previous_gpu_seconds=last['gpu_seconds'],
                   identical_to_quick=same, **score(f, c))
        vh = (None,) + tuple(host(v[1:]))
        row.update(rom_row(b, vh) if b['kind'] == 'rom' else fom_row(b, vh))
        rep['invocations'].append(row)
        seq[0] += 1
        last.update(name=name, gpu_seconds=gs)

    # ============================================================ Phase F ====
    for fs in cfg['fom_settings']:
        add(fs['name'], family='fom', kind='fom', phase='F', setting=fs, unknowns=(L - 1) ** 2)
        run_quick(fs['name'])
    fast_rom = []
    for rung in cfg['rungs']:
        for rs in rung['rules']:
            if rs.get('phase', 'F') != 'F':
                continue
            build_rule(rung, rs)
            for n_ in register(rung, rs, 'F'):
                if run_quick(n_):
                    fast_rom.append(n_)
    assert fast_rom, ('no NM-ROM arm survived Phase F quick runs', rep['dropped'])
    phaseF = [n for n in subjects if built[n]['phase'] == 'F']
    rep['phases']['F_subjects'] = phaseF
    for n in phaseF:                                   # re-warm
        jax.block_until_ready(invoke(built[n], jnp.asarray(inputs_u[0]), 0))
    order_rng = np.random.default_rng(cfg['order_seed'])
    tF = time.perf_counter()
    for r_ in range(cfg['reps']):
        for c in range(ncase):
            for i in order_rng.permutation(len(phaseF)):
                timed_call(phaseF[int(i)], c, r_, 'F')
        print('TIMED F rep', r_, el(), flush=True)
        save()
    rep['phases']['F_seconds'] = time.perf_counter() - tF

    # ============================================================ Phase S ====
    tS = time.perf_counter()

    def time_block(name):
        """A slow arm's own timed block: burn before every query, 5 reps x 6 cases, nothing else interleaved."""
        last.update(name=None, gpu_seconds=None)
        for r_ in range(cfg['reps']):
            for c in range(ncase):
                timed_call(name, c, r_, 'S')
        save()

    def affordable(name):
        """A slow arm is timed only if its 5 x 6 block fits before `phase_s_deadline_seconds` of job time (audit F4);
        otherwise it is dropped with the projected cost recorded -- never silently shortened."""
        dl = cfg.get('phase_s_deadline_seconds')
        per = quick_seconds.get(name)
        if dl is None or per is None:
            return True
        proj = per * cfg['reps'] * 1.2 + cfg['reps'] * ncase * cfg['burn_seconds']
        if time.perf_counter() - begin + proj > dl:
            rep['dropped'].append(dict(name=name, phase='timed', reason='deadline: projected block %.0f s would pass '
                                       'phase_s_deadline_seconds=%s' % (proj, dl)))
            print('DROPPED (deadline)', name, round(proj), flush=True)
            save()
            return False
        return True

    def release(name):
        b = built.get(name)
        if b is not None:
            for k_ in ('data', 'cold', 'query', 'bankobj'):
                b.pop(k_, None)
        jax.clear_caches()
        gc.collect()

    # (a) the robust-rule NM-ROM arm(s)
    for rung in cfg['rungs']:
        for rs in rung['rules']:
            if rs.get('phase') != 'S':
                continue
            build_rule(rung, rs)
            for n_ in register(rung, rs, 'S'):
                if run_quick(n_) and affordable(n_):
                    time_block(n_)
                    print('BLOCK', n_, el(), flush=True)
                release(n_)

    # (a2) the bank-span model (coordinator 2026-09-23, DESIGN A1): u = G T[:, :R'] a, LM on the R' unknowns a, the
    # advection through an EQ rule. Implemented as the ORIGINAL bank G with the linear head h(a) = T[:, :R'] a, so
    # the rule's stencils, the test projection and arms' LM/initializer are unchanged. Every arm's rule is
    # re-certified on held-out states under the truncation (rho on the states its own query visits on a
    # population disjoint from dev6 and from training), and marked when rho_max exceeds the 0.116 bar.
    bs = cfg.get('bank_span')
    if bs:
        rz = np.load(inputs.parent.parent / bs['rotation_file']) if not Path(bs['rotation_file']).is_absolute() else np.load(bs['rotation_file'])
        rot_sha = sha_file(inputs.parent.parent / bs['rotation_file'])
        assert rot_sha == bs['rotation_sha256'], rot_sha
        Tn, Ln = np.asarray(rz['T']), np.asarray(rz['L'])
        assert Tn.shape == (R, R) and float(np.linalg.norm(Ln @ Tn - np.eye(R))) < 1e-8
        rep['bank_span'] = dict(rotation_file=bs['rotation_file'], rotation_sha256=rot_sha, arms=[])
        hz = np.asarray(jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))(jnp.asarray(Zsub)))
        popc = e.params_draw(*bs['population_draw'])[bs['population_rows'][0]:bs['population_rows'][1]]
        for other, nm in ((physical, 'dev6'), (train_physical, 'train')):
            assert not any(np.any(np.all(np.isclose(o, popc), axis=1)) for o in other), f'population/{nm} overlap'
        for Rp in bs['ranks']:
            Tr = jnp.asarray(Tn[:, :Rp])
            head = (lambda a_, _T=Tr: _T @ a_)
            cand = hz @ Ln[:Rp].T                              # training codes' coefficients, rotated and truncated
            M = int(bs['tests_per_unknown'] * Rp)
            for rname in bs['rules']:
                rung = dict(q=f'bs{Rp}', M=M)
                rs = dict(name=rname, parts=[dict(lattice=int(rname.replace('lat', '')))])
                build_rule(rung, rs)
                ops, info = rules[(rung['q'], M, rname)]
                o = operators(M)
                cold = A.build_cold(bank, head, cand, cfg['cold_axis_points'])[0]
                tr = radius(cand)
                data = dict(A=o['A'], lam=o['lam'], G5=ops['G5'], Pq=ops['Pq'], Gb=G)
                query = gridarm.make_query_eq(head, Rp, L, dt, tr, ic_budget=st['ic_budget'],
                                              step_budget=st['step_budget'], gtol=bs['gtol'], linear=lin(Rp))
                name = f"bank{Rp}_M{M}_{rname}_{gt(bs['gtol'])}"
                add(name, family='bankspan', kind='rom', phase='S', k=Rp, M=M, m=info['m'], rule=rname,
                    gtol=bs['gtol'], unknowns=Rp, trust_radius=tr, quadrature='eq', data=data, cold=cold,
                    query=query, kernel='arms EQ LM (jacfwd), linear head T[:, :R\']; output via row-blocked bank',
                    rule_status='re-certified in this job on held-out states (see bank_span)')
                if run_quick(name) and affordable(name):
                    # held-out rho under the truncation, on the states this arm's own query visits
                    t0 = time.perf_counter()
                    co = []
                    for ph in popc:
                        v = query(jnp.asarray(e.initial(L, ph)), float(ph[4]), data, cold)
                        co.append(np.asarray(jax.vmap(head)(v[7])))
                        del v
                    co = np.concatenate(co)
                    tg = H.dense_targets(G, co, o['sx'], o['sy'], L, chunk=cfg['target_chunk'])
                    r = H.rho(ops['Pq'], H.sampled_advection(ops['G5'], co, L), tg)
                    per = int(round(.25 / dt)) + 1
                    kidx = np.tile(np.arange(per), len(popc))
                    ent = dict(arm=name, R_prime=Rp, M=M, rule=rname, m=info['m'], states=int(len(r)),
                               population=dict(draw=bs['population_draw'], rows=bs['population_rows']),
                               rho_max=float(r.max()), rho_max_k_ge_1=float(r[kidx >= 1].max()),
                               rho_p99=float(np.quantile(r, .99)), rho_median=float(np.median(r)),
                               bar=cfg['rho_bar'], exceeds_bar=bool(r.max() > cfg['rho_bar']),
                               seconds=time.perf_counter() - t0)
                    rep['bank_span']['arms'].append(ent)
                    np.save(fields_dir / f'rho_{name}.npy', r)
                    built[name]['rule_status'] = ('rho_max %.4f on %d held-out states: %s the %.3f bar' % (
                        ent['rho_max'], ent['states'], 'EXCEEDS' if ent['exceeds_bar'] else 'within', cfg['rho_bar']))
                    rep['arm_setup'][-1]['rule_status'] = built[name]['rule_status']
                    print('BANKSPAN RHO', name, round(ent['rho_max'], 4), 'exceeds' if ent['exceeds_bar'] else 'ok', el(), flush=True)
                    save()
                    time_block(name)
                    print('BLOCK', name, el(), flush=True)
                release(name)

    # (b) snapshots at this mesh, then the POD and quadratic-manifold fits (streamed; trajectory-split ridge)
    pod_ranks = sorted(set(cfg.get('pod_ranks', [])))
    qranks = sorted(set(cfg.get('qman_ranks', [])))
    if pod_ranks or qranks:
        Ut, sinfo = sfit.snapshots(L, dt, train_physical, cfg['train_state_stride'], cfg['snapshot_ntol'],
                                   cfg['snapshot_ltol'])
        rep['snapshots'] = sinfo
        print('SNAPSHOTS', Ut.shape, round(sinfo['seconds'], 1), el(), flush=True)
        save()
        tb = float(cfg.get('fit_block_bytes', 2e9))
        Vmodes, coords = None, None
        if pod_ranks:
            t0 = time.perf_counter()
            Vmodes, eig, energy, coords = sfit.pod(Ut, max(pod_ranks), tb)
            rep['pod'] = dict(kmax=max(pod_ranks), eigenvalues=np.asarray(eig).tolist(), total_energy=energy,
                              tail_fraction={str(k): float(max(energy - float(np.sum(eig[:k])), 0.) / max(energy, 1e-300))
                                             for k in pod_ranks},
                              seconds=time.perf_counter() - t0, basis='uncentred method of snapshots (ablation.pod_basis), streamed',
                              modes_sha256=sha_array(Vmodes))
            print('POD', max(pod_ranks), round(time.perf_counter() - t0, 1), el(), flush=True)
            save()
        # The quadratic manifolds are fitted LAZILY, one rank at a time, each run and released before the next is
        # fitted: at 2048^2 the snapshot matrix (111 GB) plus all three banks at once (96 GB) would not fit in host RAM.
        state = dict(Ut=Ut, cg=None)
        del Ut

        def qman_fit(r_):
            if state['cg'] is None:
                state['cg'] = sfit.CentredGram(state['Ut'], tb)
            m_ = sfit.qman_fit(state['cg'], r_, cfg['qman_gammas'], cfg['qman_seed'], cfg['qman_holdout'],
                               sinfo['states_per_trajectory'])
            assert m_['info']['split'].startswith('by trajectory') and m_['info']['ridge'] in m_['info']['ridge_grid']
            json.dumps(m_['info'], allow_nan=False)
            rep['quadratic_manifold'].append(dict(m_['info'], bank_sha256=sha_array(m_['bank'])))
            print('QMAN r', r_, 'ridge', m_['info']['ridge'], '|W|', f"{m_['info']['weight_frobenius_norm']:.4g}",
                  'heldout', round(m_['info']['heldout_relative'], 6), 'lin/quad',
                  round(m_['info']['snapshot_relative_linear_only'], 6),
                  round(m_['info']['snapshot_relative_with_quadratic'], 6), round(m_['info']['seconds'], 1), el(), flush=True)
            save()
            return m_

        rep['phases']['fits_seconds'] = time.perf_counter() - tS

        def build_grid_arm(name, family, Bhost, head, dim, co, M, axis_points, setup_extra, twin=False):
            """twin=False: the Phi-free row-blocked arm (gridarm) that is timed. twin=True: the ORIGINAL arms.make_query
            dense arm on the same bank, run untimed at small rank as the grid_arm_parity gate's reference."""
            t0 = time.perf_counter()
            if twin:
                gb = A.GridBank(jnp.asarray(Bhost), L)
                data, info = A.build_operators(gb, L, M, 'dense')
                query = A.make_query(head, dim, L, dt, radius(co), 'dense', linear=lin(dim), ic_budget=st['ic_budget'],
                                     step_budget=st['step_budget'], gtol=st['gtol'])
            else:
                gb = gridarm.BlockGridBank(Bhost, L)
                data, info = gridarm.build_operators(gb, M)
                query = gridarm.make_query(head, dim, L, dt, radius(co), ic_budget=st['ic_budget'],
                                           step_budget=st['step_budget'], gtol=st['gtol'], linear=lin(dim),
                                           chunk=cfg['grid_tangent_chunk'])
            cold, cinfo = A.build_cold(gb, head, co, axis_points)
            add(name, family=family, kind='rom', phase=('P' if twin else 'S'), k=dim, M=M, m=None, quadrature='dense',
                gtol=st['gtol'], unknowns=dim, trust_radius=radius(co), cold_axis_points=axis_points,
                array_bytes=info['array_bytes'], data=data, cold=cold, query=query, bankobj=gb,
                kernel=('arms.make_query dense (explicit Phi) -- parity twin, untimed' if twin else
                        'gridarm: arms dense residual, Phi-free, row-blocked, chunked-jvp Jacobian'), **setup_extra)
            rep['arm_setup'][-1]['setup_seconds'] = time.perf_counter() - t0

        def grid_parity(name, twin):
            per = []
            for c in range(ncase):
                fa = np.load(fields_dir / f'full_{name}_case{c}.npy')
                fb = np.load(fields_dir / f'full_{twin}_case{c}.npy')
                qa = next(x for x in rep['quick'] if x['name'] == name and x['case'] == c)
                qb = next(x for x in rep['quick'] if x['name'] == twin and x['case'] == c)
                per.append(dict(case=c, relative=float(np.linalg.norm(fa - fb) / np.linalg.norm(fb)),
                                iterations_identical=qa['iterations'] == qb['iterations'],
                                reasons_identical=qa['stop_reasons'] == qb['stop_reasons'],
                                ic_identical=(qa['ic_iterations'], qa['ic_reason']) == (qb['ic_iterations'], qb['ic_reason'])))
            worst = max(x['relative'] for x in per)
            ints = all(x['iterations_identical'] and x['reasons_identical'] and x['ic_identical'] for x in per)
            rep['gates'].setdefault('grid_arm_parity', dict(pairs=[], bar=1e-9))
            rep['gates']['grid_arm_parity']['pairs'].append(dict(arm=name, twin=twin, worst_relative=worst,
                                                                 integers_identical=ints, passed=bool(worst <= 1e-9 and ints),
                                                                 cases=per))
            rep['gates']['grid_arm_parity']['passed'] = all(x['passed'] for x in rep['gates']['grid_arm_parity']['pairs'])
            print('GRID PARITY', name, f'{worst:.3e}', 'integers', ints, flush=True)
            save()

        specs = [('pod', k) for k in pod_ranks] + [('qman', r_) for r_ in qranks]
        for fam, k in specs:
            if fam == 'pod':
                name = f'pod{k}_M{4 * k}_dense'
                args_ = (name, 'pod', np.ascontiguousarray(Vmodes[:, :k]), A.identity_head(), k,
                         coords[:, :k], 4 * k, cfg['cold_axis_points'],
                         dict(fit='classical POD of truth snapshots at this mesh, identity head'))
            else:
                if Vmodes is not None:                       # every POD arm has run: release the POD basis
                    Vmodes, coords = None, None
                    gc.collect()
                m_ = qman_fit(k)
                if k == max(qranks):                         # last fit: release the snapshots before the biggest bank runs
                    state['Ut'] = state['cg'] = None
                    gc.collect()
                ncol = m_['bank'].shape[1]
                axis = int(cfg['cold_axis_points'])
                if axis ** 2 <= 2 * ncol:
                    axis = int(cfg['qman_cold_axis_points'])
                assert axis ** 2 > 2 * ncol, (axis, ncol)
                name = f'qman{k}_quad_M{4 * k}'
                args_ = (name, 'qman', m_['bank'], QM.head(k, True), k, m_['coefficients'], 4 * k, axis,
                         dict(bank_columns=int(ncol), manifold_fit=m_['info'],
                              fit='quadratic manifold u_ref + V_r a + W vech(a a^T), ridge by trajectory-split holdout'))
            if k in cfg.get('grid_parity_ranks', []):
                tw = args_[0] + '_armsTwin'
                try:
                    build_grid_arm(tw, *args_[1:], twin=True)
                    if run_quick(tw):
                        release(tw)
                    twin_ok = tw in subjects
                except Exception as exc:              # noqa: BLE001
                    if not is_oom(exc):
                        raise
                    rep['dropped'].append(dict(name=tw, phase='build', reason=str(exc)[:400]))
                    built.pop(tw, None)
                    twin_ok = False
                release(tw)
            else:
                twin_ok = False
            try:
                build_grid_arm(*args_)
            except Exception as exc:                  # noqa: BLE001
                if not is_oom(exc):
                    raise
                rep['dropped'].append(dict(name=args_[0], phase='build', reason=str(exc)[:400]))
                print('DROPPED (build, OOM)', args_[0], flush=True)
                built.pop(args_[0], None)
                jax.clear_caches()
                gc.collect()
                save()
                continue
            if run_quick(name):
                if twin_ok:
                    grid_parity(name, tw)
                try:
                    time_block(name)
                except Exception as exc:              # noqa: BLE001
                    if not is_oom(exc):
                        raise
                    rep['dropped'].append(dict(name=name, phase='timed', reason=str(exc)[:400]))
                print('BLOCK', name, el(), flush=True)
            release(name)
            if fam == 'qman':
                m_['bank'] = None
                args_ = None
            gc.collect()
        state.clear()
        Vmodes = coords = None
        gc.collect()
    rep['phases']['S_seconds'] = time.perf_counter() - tS

    # ============================================================ Phase B ====
    tB = time.perf_counter()
    last.update(name=None, gpu_seconds=None)
    for n in phaseF:
        jax.block_until_ready(invoke(built[n], jnp.asarray(inputs_u[0]), 0))
    brng = np.random.default_rng(cfg['order_seed'] + 1)
    for r_ in range(cfg['bracket_reps']):
        for c in range(ncase):
            for i in brng.permutation(len(phaseF)):
                timed_call(phaseF[int(i)], c, r_, 'B')
    rep['phases']['B_seconds'] = time.perf_counter() - tB
    save()

    inv = rep['invocations']
    cover = {}
    for x in inv:
        if x['phase'] in ('F', 'S'):
            cover[(x['name'], x['case'])] = cover.get((x['name'], x['case']), 0) + 1
    rep['gates']['repetition_output_identical'] = dict(
        passed=bool(inv) and all(x['identical_to_quick'] for x in inv),
        basis='full-field SHA256 of every timed repetition against the untimed quick run of the same arm and case')
    rep['gates']['five_retained_repetitions_everywhere'] = dict(
        passed=bool(cover) and min(cover.values()) >= cfg['required_reps'] and
        all(cover.get((n, c), 0) >= cfg['required_reps'] for n in subjects if built[n]['phase'] in ('F', 'S')
            for c in range(ncase)),
        minimum=min(cover.values()) if cover else 0)
    rep['gates']['fft_tight_converged_everywhere'] = dict(
        passed=all(x['nonlinear_converged'] for x in inv if x['name'] == tight['name']))

    np.save(fields_dir / 'truth_sha256.npy', np.array([rep['phases']['truth'][c]['field_sha256'] for c in range(ncase)]))

    rep['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert rep['checkpoint_sha256'] == rep['checkpoint_sha256_after']
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('CMP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
