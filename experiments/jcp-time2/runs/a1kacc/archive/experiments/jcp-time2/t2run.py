"""jcp-time2 driver: one allocation per (mesh, settings). DESIGN.md (+ amendments A1-A3) is the specification.

Per setting (acc: Gauss 96^2, fast: Fibonacci 1597; R', M as the 2D lane):
  0  setup: cohorts (hash-checked, disjoint from training), references (manifest + sha256, ST, S; TX = (16 ST - S)/15),
     frozen model, rotated bank on the mesh, operators, rule blocks, form data (GAL: QR of A; rank/cond of A recorded)
  1  per case: every run of the grid (rule x form x scheme x dt x tolerance level), each an end-to-end query
     (initial fit, LMM stepping, decode of six fields). Kept per run: output coefficients W (6, R'), w0, statistics,
     and on the device the fields on the 257^2 shared nodes. Metrics per run: e_ST, e_S, e_TX (257^2), anchor
     discrepancy (257^2 and full mesh), production-vs-tight and tight-vs-tighter distances, self-differences
     d(h) = max_t ||u_h - u_{h/2}|| / ||u0|| along the dyadic chain. G1a: vendor qcore BE query vs generic LSPG-BE.
  2  timing: paired A-B-A (A = generic LSPG-BE at dt0) on the timing cases; every timed output's 257^2 sha256 must
     equal the accuracy-phase output of the same (run, case); the compiled-cache size of every query is recorded
     before and after.
Outputs result.json, W_<setting>.npz (coefficients of every run). Order claims, selection and the report are made
offline (analyze_t2.py / make_report.py) from these; audit_t2.py reconstructs metrics independently.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import t2core as T2
import jax
import jax.numpy as jnp

Q = T2.load_qcore()
import qstudy as QS  # noqa: E402  (Mesh, burn, clean; vendored, unchanged)

DT0 = .005
EXPECTED = {'test64': 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def tol_of(form, level):
    """(gtol, tolf) per form and tolerance level (DESIGN A1.5, A2.11)."""
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]     # A5.1
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


def run_list(cfg):
    """The grid of DESIGN section 3 / A1.3 / A1.5 / A1.11 for one setting: tuples (rule_role, form, scheme, dt, level)."""
    dts = [DT0 * f for f in cfg['dt_factors']]                       # 1/8 .. 10
    tighter = [DT0 * f for f in cfg['tighter_factors']]              # 1/8 .. 2
    timed = [d for d in dts if d >= DT0 / 2 - 1e-15]
    runs = []
    for form in cfg['forms']:
        for sc in cfg['schemes']:
            runs += [('main', form, sc, d, 'prod') for d in dts]
            runs += [('main', form, sc, d, 'tight') for d in dts]
            runs += [('main', form, sc, d, 'tighter') for d in tighter]
        for sc in cfg.get('control_schemes', []):
            runs += [('main', form, sc, d, 'tight') for d in dts] + [('main', form, sc, d, 'tighter') for d in tighter]
        if form == 'GAL':
            for sc in cfg.get('anchor_schemes', []):
                runs += [('main', form, sc, DT0 / 16, 'tight'), ('main', form, sc, DT0 / 16, 'tighter')]
        if cfg.get('hq'):
            runs += [('hq', form, sc, d, 'prod') for sc in cfg['schemes'] for d in timed]
    if cfg.get('old'):
        runs += [('old', 'LSPG', sc, DT0 * f, 'prod') for sc in cfg['old_schemes'] for f in cfg['old_dt_factors']]
    return runs


def key(r):
    role, form, sc, dt, lev = r
    return f'{role}|{form}|{sc}|{dt:.8g}|{lev}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    L = int(cfg['mesh'])
    s256 = max(1, L // 256)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT), cohorts={}, references={},
               setup={}, rows=[], g1a=[], vendor_rows=[], timing={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)

    # ------------------------------------------------------------------ cohorts ----
    train = Q.e.params_draw(0, 128)
    allc = {k: Q.cohort(k) for k in ('dev6', 'val32', 'test64')}
    for k1 in allc:
        assert not any(np.allclose(t, s) for t in train for s in allc[k1]), f'{k1} overlaps training'
    cases = []
    for coh in cfg['cohorts']:
        ph = allc[coh]
        h = sha(ph)
        if coh in EXPECTED:
            assert h == EXPECTED[coh], (coh, h)
        rep['cohorts'][coh] = dict(n=len(ph), physical_sha256=h, physical=ph.tolist())
        sub = cfg.get('case_subset', {}).get(coh)
        for c in (range(len(ph)) if sub is None else sub):
            cases.append((coh, int(c), ph[c]))
    print('CASES', len(cases), el(), flush=True)

    # --------------------------------------------------------------- references ----
    rdir = Path(cfg['refs'])
    man = json.loads((rdir / 'result.json').read_text())
    assert man.get('complete') and man.get('all_accepted'), 'reference job incomplete or rejected'
    want = cfg['ref_contract']
    assert int(man['config']['mesh']) == want['mesh']
    for tag, w_ in want['refs'].items():
        got = man['config']['refs'][tag]
        assert all(abs(float(got[k_]) - float(w_[k_])) <= 1e-15 * max(1., abs(float(w_[k_]))) for k_ in w_), (tag, got)
    idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    assert len(idx) == len(man['cases'])
    refs = {}
    for coh, c, ph in cases:
        if not cfg.get('local_smoke_waives_cohort_hash'):    # local GB10 numpy differs from the cluster by ~1 ulp
            assert man['cohort_sha256'][coh] == rep['cohorts'][coh]['physical_sha256'], coh
        for tag in ('ST', 'S'):
            ent = idx[(coh, c, tag)]
            assert ent['accepted'] and ent['mesh'] == want['mesh'] and abs(ent['dt'] - want['refs'][tag]['dt']) <= 1e-15
            r = np.load(rdir / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert r.shape == (6, 257, 257) and np.isfinite(r).all() and sha(r) == ent['f257_sha256'], (coh, c, tag)
            refs[(coh, c, tag)] = r
            rep['references'][f'{tag}|{coh}|{c}'] = ent['f257_sha256']
    rep['references_job'] = dict(dir=str(rdir), job_id=man.get('job_id'), commit=man.get('commit'))
    print('REFS', len(refs), el(), flush=True)

    # ------------------------------------------------------------- model, mesh ----
    mdl = Q.Model(ckpt=T2.CKPT)
    mesh = QS.Mesh(mdl, L, cfg['settings'], nrb=cfg.get('bank_blocks'))
    rep['setup']['bank'] = dict(edges=mesh.edges, seconds=mesh.bank_seconds, bytes=mesh.bank_bytes)
    Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
    hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
    Hsub = np.asarray(hv(jnp.asarray(Zsub)))
    print('BANK+OPS', el(), flush=True)
    W_all = {}

    for s in cfg['settings']:
        st_ = Q.SETTINGS[s]
        Rp, M = st_['Rp'], st_['M']
        assert st_['kind'] == 'lin'
        trust, _ = mdl.trust_linear(Rp)
        cold = Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T)
        o = mesh.ops[M]
        base = mesh.base(s)
        rules = cfg['rules'][s]                 # {'main': {'kind','rule'}, 'hq': ..., 'old': ...}
        blocks = {}
        for role, spec in rules.items():
            if spec['kind'] == 'mesh':
                ij, w = Q.mesh_rule(spec['rule'], L)
                blocks[role] = (Q.mesh_data(mdl, Rp, ij, w, L, o['kx'], o['ky']), len(ij))
            else:
                X, w = Q.offmesh_rule(spec['rule'])
                blocks[role] = (Q.offmesh_data(mdl, Rp, X, w, L, o['kx'], o['ky'], spec['kind']), len(X))
        rep['setup'][s] = dict(R_prime=Rp, M=M, trust=trust, rules={r_: dict(rules[r_], m=blocks[r_][1]) for r_ in rules},
                               A=T2.a_conditioning(base['A']))
        data, queries = {}, {}
        for role in rules:
            for form in (('LSPG',) if role == 'old' else ('LSPG', 'GAL')):
                data[(role, form)] = T2.form_data(base, blocks[role][0], form)
                queries[(role, form)] = T2.make_query(Q, rules[role]['kind'], form, Rp, L, trust)
        vq, _ = Q.make_linear_query(rules['main']['kind'], Rp, L, DT0, trust, step_budget=600, gtol=1e-3)
        vdata = dict(base, **blocks['main'][0])
        runs = run_list(dict(cfg['grid'], hq='hq' in rules, old='old' in rules))
        rep['setup'][s]['runs'] = [key(r) for r in runs]
        rep['setup'][s]['output_times'] = [.05 * j for j in range(6)]     # rows 0..5 of every W
        anchor_key = key(('main', 'GAL', cfg['grid']['anchor'], DT0 / 16, 'tight'))
        print('SETUP', s, len(runs), 'runs/case', el(), flush=True)
        Wset = np.full((len(cases), len(runs), 6, Rp), np.nan)
        w0set = np.full((len(cases), Rp), np.nan)
        acc_sha = {}

        def call(r, u0, nu):
            role, form, sc, dt, lev = r
            gtol, tolf = tol_of(form, lev)
            return queries[(role, form)](u0, nu, data[(role, form)], cold, T2.sched(sc, dt, gtol, tolf))

        # ------------------------------------------------------------- phase 1 ----
        order = sorted(range(len(runs)), key=lambda i: key(runs[i]) != anchor_key)     # anchor first
        for ci, (coh, c, ph) in enumerate(cases):
            u0 = jnp.asarray(Q.e.initial(L, ph))
            nu = float(ph[4])
            u0n = np.asarray(u0)
            n0, n0r = float(np.linalg.norm(u0n)), float(np.linalg.norm(u0n[::s256, ::s256]))
            rST, rS = (jnp.asarray(refs[(coh, c, t_)]) for t_ in ('ST', 'S'))
            rTX = (16. * rST - rS) / 15.
            f257 = {}
            anc_full = gen_full = None
            for i in order:
                r = runs[i]
                t1 = time.perf_counter()
                v = call(r, u0, nu)
                jax.block_until_ready(v)
                secs = time.perf_counter() - t1
                f = v['fields']
                fr = f[:, ::s256, ::s256]
                k_ = key(r)
                f257[k_] = fr
                W = np.asarray(v['W'])
                Wset[ci, i] = W
                w0set[ci] = np.asarray(v['w0'])
                stt = {k2: np.asarray(x) for k2, x in v['stats'].items()}
                row = dict(setting=s, run=k_, role=r[0], form=r[1], scheme=r[2], dt=r[3], level=r[4], cohort=coh, case=c,
                           finite=bool(jnp.all(jnp.isfinite(f))), seconds_first=secs, stats=stt,
                           verified=bool(int(stt['nfail']) == 0 and np.all(np.isfinite(W))))
                for tag, R_ in (('ST', rST), ('S', rS), ('TX', rTX)):
                    pe = [float(x) for x in jnp.linalg.norm((fr - R_).reshape(6, -1), axis=1) / n0r]
                    row[f'e_{tag}_per_time'] = pe
                    row[f'e_{tag}'] = max(pe[1:])
                if k_ == anchor_key:
                    anc_full = f
                if r == ('main', 'LSPG', 'BE', DT0, 'prod'):
                    gen_full = f
                if anc_full is not None:
                    row['anchor_full'] = max(float(x) for x in (jnp.linalg.norm((f - anc_full).reshape(6, -1), axis=1) / n0)[1:])
                if r[4] == 'prod' and ci < cfg['timing']['cases'] and coh == cfg['timing']['cohort']:
                    acc_sha[(k_, ci)] = sha(np.asarray(fr))
                rep['rows'].append(row)
                del v, f
            # metrics needing several runs (257^2 nodes)
            dist = lambda k1, k2, nrm=n0r: max(float(x) for x in (jnp.linalg.norm((f257[k1] - f257[k2]).reshape(6, -1), axis=1) / nrm)[1:])
            pair = {}
            for i, r in enumerate(runs):
                k_ = key(r)
                role, form, sc, dt, lev = r
                m = {}
                if anchor_key in f257:
                    m['anchor_257'] = dist(k_, anchor_key)
                half = key((role, form, sc, dt / 2, lev))
                if half in f257:
                    m['d_half'] = dist(k_, half)
                if lev == 'prod':
                    kt = key((role, form, sc, dt, 'tight'))
                    if kt in f257:
                        m['prod_vs_tight'] = dist(k_, kt)
                if role == 'hq':      # quadrature sensitivity: paired field distance to the main-rule run (code audit 2, item 15)
                    km = key(('main', form, sc, dt, lev))
                    if km in f257:
                        m['vs_main'] = dist(k_, km)
                if lev == 'tight':
                    kt = key((role, form, sc, dt, 'tighter'))
                    if kt in f257:
                        m['s_h'] = dist(k_, kt)
                pair[k_] = m
            base_i = len(rep['rows']) - len(runs)
            for j, i in enumerate(order):
                rep['rows'][base_i + j].update(pair[key(runs[i])])
            # G1a: vendor BE vs generic LSPG-BE at dt0, production tolerance
            vv = vq(u0, nu, vdata, cold)
            jax.block_until_ready(vv)
            ig = runs.index(('main', 'LSPG', 'BE', DT0, 'prod'))
            Wv = np.asarray(vv['internal'])[::int(round(.05 / DT0))]
            frv = vv['fields'][:, ::s256, ::s256]
            if ci < cfg['timing']['cases'] and coh == cfg['timing']['cohort']:
                acc_sha[('vendor|LSPG|BE|0.005|prod', ci)] = sha(np.asarray(frv))
            gen_it = int(rep['rows'][base_i + order.index(ig)]['stats']['it_sum'])
            rep['g1a'].append(dict(setting=s, cohort=coh, case=c,
                                   w0_rel=float(np.linalg.norm(w0set[ci] - Wv[0]) / np.linalg.norm(Wv[0])),
                                   W_rel=float(np.max(np.linalg.norm(Wset[ci, ig] - Wv, axis=1)) / np.linalg.norm(Wv[0])),
                                   field_rel=max(float(x) for x in jnp.linalg.norm((vv['fields'] - gen_full).reshape(6, -1), axis=1) / n0),
                                   it_vendor=int(np.sum(np.asarray(vv['it']))), it_generic=gen_it))
            vrow = dict(setting=s, cohort=coh, case=c)
            for tag, R_ in (('ST', rST), ('S', rS)):
                pe = [float(x) for x in jnp.linalg.norm((frv - R_).reshape(6, -1), axis=1) / n0r]
                vrow[f'e_{tag}'] = max(pe[1:])
            rep['vendor_rows'].append(vrow)
            del vv, f257, anc_full, gen_full, rST, rS, rTX
            print('CASE', s, ci, coh, c, el(), flush=True)
            if ci % 4 == 3:
                save()
        np.savez_compressed(out / f'W_{s}.npz', W=Wset, w0=w0set, runs=np.array([key(r) for r in runs]),
                            cases=np.array([f'{coh}|{c}' for coh, c, _ in cases]))
        save()

        # ------------------------------------------------------------- phase 2: timing ----
        tc = cfg['timing']
        tcases = [(ci, x) for ci, x in enumerate(cases) if x[0] == tc['cohort']][:tc['cases']]
        A_run = ('main', 'LSPG', 'BE', DT0, 'prod')
        cands = [r for r in runs if r[4] == 'prod' and r[0] in ('main', 'old') and r[3] >= DT0 / 2 - 1e-15 and r != A_run]
        cands = [r for r in cands if r[0] == 'main' or (r[2] == 'BE' and abs(r[3] - DT0) < 1e-15)]
        rng = np.random.default_rng(int(tc.get('seed', 20261008)))
        inputs = {ci: (jnp.asarray(Q.e.initial(L, ph)), float(ph[4])) for ci, (coh, c, ph) in tcases}
        cache0 = {f"{k_[0]}|{k_[1]}": getattr(queries[k_], "_cache_size", lambda: -1)() for k_ in queries}
        cache0['vendor'] = getattr(vq, '_cache_size', lambda: -1)()
        vend = ('vendor', 'LSPG', 'BE', DT0, 'prod')

        def timed(r, ci):
            u0, nu = inputs[ci]
            QS.burn(tc['burn'])
            t1 = time.perf_counter()
            if r == vend:
                o_ = vq(u0, nu, vdata, cold)
            else:
                o_ = call(r, u0, nu)
            jax.block_until_ready(o_)
            secs = time.perf_counter() - t1
            hh = sha(np.asarray(o_['fields'][:, ::s256, ::s256]))
            return secs, hh

        for r in [A_run, vend] + cands:              # warm every subject on every timing case
            for ci, _ in tcases:
                timed(r, ci)
        inv = []
        for rep_i in range(tc['reps']):
            orderT = [(j, ci) for j in range(len(cands) + 1) for ci, _ in tcases]
            rng.shuffle(orderT)
            for j, ci in orderT:
                B = vend if j == len(cands) else cands[j]
                ta1, ha1 = timed(A_run, ci)
                tb, hb = timed(B, ci)
                ta2, ha2 = timed(A_run, ci)
                ent = dict(setting=s, B=key(B), case_index=ci, rep=rep_i, tA1=ta1, tB=tb, tA2=ta2,
                           A_sha=[ha1, ha2], A_expected_sha=acc_sha.get((key(A_run), ci)),
                           ratio=tb / (.5 * (ta1 + ta2)), drift=ta2 / ta1,
                           A_matches_accuracy=bool(ha1 == acc_sha.get((key(A_run), ci)) and ha2 == acc_sha.get((key(A_run), ci))))
                ent['B_expected_sha'] = acc_sha.get((key(B), ci))
                ent['B_sha'] = hb
                ent['B_matches_accuracy'] = bool(hb == acc_sha.get((key(B), ci)))
                inv.append(ent)
        cache1 = {f"{k_[0]}|{k_[1]}": getattr(queries[k_], "_cache_size", lambda: -1)() for k_ in queries}
        cache1['vendor'] = getattr(vq, '_cache_size', lambda: -1)()
        rep['timing'][s] = dict(candidates=[key(r) for r in cands] + [key(vend)], cases=[f'{cases[ci][0]}|{cases[ci][1]}' for ci, _ in tcases], reps=tc['reps'],
                                burn=tc['burn'], A=key(A_run), invocations=inv, cache_before=cache0, cache_after=cache1,
                                gpu_uuid=smi)
        print('TIMING', s, len(inv), el(), flush=True)
        save()
        del data, queries, blocks, vq, vdata, Wset
        gc.collect()

    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('T2RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
