"""quadrature-study driver: one allocation per (mesh, cohort set). DESIGN.md is the specification.

Phases
  0  setup: frozen model, rotated bank on the mesh (nested column blocks 128 | 384 | 512), operators for M in
     {1536, 512, 64}, cold initialisers, every arm's rule blocks, references (refined ST and S) for the cohort
  1  per case (case-major): the same-grid truth (Newton--BiCGStab lean_tight), then per setting every rollout arm:
     same-grid error, refined-reference errors (ST, S) on the shared 257^2 nodes, distance from the dense rollout and
     from the continuum (gref) rollout, LM iterations / exits / rejections, query seconds. The deployed lat64 arm's
     internal states k = 1..50 are kept as the rho population.
  2  rho ladders per setting on that population: every rule's tested advection against the continuum target (gref)
     and the mesh target (dense), worst / p95 / median / argmax; continuum-target convergence gate (gref vs gref_check)
  3  timing: randomised order, burn-in before every invocation, block_until_ready; ROM arms, the decode alone, and the
     full-order settings, all in this allocation
Outputs result.json (+ npz side files) in --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import qcore as Q

EDGES = [0, 128, 384, 512]
REASONS = {0: 'budget', 1: 'tol', 2: 'tiny_step', 3: 'damping_limit', 4: 'stationary'}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


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
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    L, dt = int(cfg['mesh']), .005
    s256 = max(1, L // 256)     # stride to the 257^2 nodes shared by every mesh (L >= 256)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, jax_version=jax.__version__, mesh=L,
               checkpoint_sha256=sha_file(Q.CKPT), rotation_sha256=sha_file(Q.ROTATION), eq_rule_sha256=sha_file(Q.EQ_RULE),
               cohorts={}, references={}, setup={}, rows=[], fom_rows=[], rho={}, gates={}, timing={}, complete=False)
    save = lambda: json.dump(clean(rep), open(out / 'result.json', 'w'), indent=1)

    # ------------------------------------------------------------- cohorts ----
    cases = []
    train = Q.e.params_draw(0, 128)
    for coh in cfg['cohorts']:
        ph = Q.cohort(coh)
        assert not any(np.allclose(t, s) for t in train for s in ph), f'{coh} overlaps training'
        rep['cohorts'][coh] = dict(n=len(ph), physical_sha256=sha(ph), physical=ph.tolist())
        sub = cfg.get('case_subset', {}).get(coh)
        for c in (range(len(ph)) if sub is None else sub):
            cases.append((coh, int(c), ph[c]))
    print('CASES', len(cases), el(), flush=True)

    # ---------------------------------------------------------- references ----
    def load_ref(tag, coh, c):
        d = cfg.get('refs', {}).get(tag)
        if not d:
            return None
        f = Path(d) / f'ref_{tag}_{coh}_{c:03d}.npz'
        if not f.exists():
            return None
        r = np.load(f)['f257']
        rep['references'].setdefault(tag, {})[f'{coh}{c}'] = dict(file=str(f), f257_sha256=sha(r))
        return r

    # ---------------------------------------------------------- model, bank ----
    mdl = Q.Model()
    t0 = time.perf_counter()
    Rb = max(Q.SETTINGS[s]['Rp'] for s in cfg['settings'])
    edges = [e_ for e_ in EDGES if e_ <= Rb]
    Grot = Q.build_rotated_bank(mdl, L, edges, nrb=cfg.get('bank_blocks'))
    rep['setup']['bank'] = dict(columns=Rb, edges=edges, seconds=time.perf_counter() - t0,
                                bytes=int(sum(c.nbytes for rb in Grot for c in rb)))
    print('BANK', el(), flush=True)
    ops = {}
    for s in cfg['settings']:
        M = Q.SETTINGS[s]['M']
        if M not in ops:
            ops[M] = Q.operators(Grot, edges, L, M)
    print('OPS', el(), flush=True)
    Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
    hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
    Hsub = np.asarray(hv(jnp.asarray(Zsub)))

    S = {}
    for s in cfg['settings']:
        st = Q.SETTINGS[s]
        Rp, M = st['Rp'], st['M']
        o = ops[M]
        base = dict(A=o['Arot'][:, :Rp], lam=o['lam'], G=Q.BK.prefix(Grot, edges, Rp))
        if st['kind'] == 'lin':
            trust, _ = mdl.trust_linear(Rp)
            cold = Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T)
            pf = None
            cmap = lambda W: W
        else:
            trust = mdl.trust_head()
            pf = mdl.head_params(Rp)
            cold = Q.build_cold_head(mdl, pf, Zsub)
            hf = jax.jit(jax.vmap(lambda z, pf=pf: Q.sc.head(pf, z)))
            cmap = lambda W, hf=hf: np.asarray(hf(jnp.asarray(W)))
        S[s] = dict(Rp=Rp, M=M, kind=st['kind'], base=base, trust=trust, cold=cold, pf=pf, cmap=cmap, arms={})
        rep['setup'][s] = dict(R_prime=Rp, M=M, kind=st['kind'], trust=trust)

    rule_cache = {}

    def rule_blocks(s, kind, rule):
        key = (s, kind, rule)
        if key in rule_cache:
            return rule_cache[key]
        st, o = S[s], ops[S[s]['M']]
        if kind == 'mesh':
            ij, w = Q.mesh_rule(rule, L)
            d = Q.mesh_data(mdl, st['Rp'], ij, w, L, o['kx'], o['ky'])
            m = len(ij)
        elif kind in ('point', 'flux'):
            X, w = Q.offmesh_rule(rule)
            d = Q.offmesh_data(mdl, st['Rp'], X, w, L, o['kx'], o['ky'], kind)
            m = len(X)
        elif kind == 'dense':
            d, m = dict(sx=o['sx'], sy=o['sy']), (L - 1) ** 2
        else:
            raise ValueError(kind)
        rule_cache[key] = (d, m)
        return rule_cache[key]

    def build(s, spec):
        st = S[s]
        name = spec['name']
        if name in st['arms']:
            return st['arms'][name]
        t1 = time.perf_counter()
        d, m = rule_blocks(s, spec['kind'], spec.get('rule'))
        data = dict(st['base'], **d)
        gtol = spec.get('gtol', cfg['gtol'])
        chunks = cfg.get('tangent_chunks', {}).get(s, 1) if spec['kind'] == 'dense' else 1
        if st['kind'] == 'lin':
            fq, dec = Q.make_linear_query(spec['kind'], st['Rp'], L, dt, st['trust'], step_budget=cfg['step_budget'],
                                          gtol=gtol, tangent_chunks=chunks)
            call = lambda u, nu, fq=fq, data=data: fq(u, nu, data, st['cold'])
        else:
            fq, dec = Q.make_head_query(spec['kind'], st['pf'], mdl.K, L, dt, st['trust'],
                                        step_budget=cfg['step_budget'], gtol=gtol, tangent_chunks=chunks)
            tab = Q.head_tables(st['pf'], mdl.K, data, st['cold'])
            call = lambda u, nu, fq=fq, data=data, tab=tab: fq(u, nu, data, st['cold'], tab)
        if 'decode' not in st:
            # the six-field decode alone (identical for every arm of the setting): W (6, R') or (6, K) codes
            if st['kind'] == 'lin':
                st['decode'] = (lambda W, dec=dec, data=data: dec(W, data), st['Rp'])
            else:
                st['decode'] = (lambda W, dec=dec, data=data, tab=tab: dec(W, data, tab), mdl.K)
        st['arms'][name] = dict(spec=spec, call=call, dec=dec, data=data, m=m, gtol=gtol)
        rep['setup'].setdefault('arms', []).append(dict(setting=s, arm=name, kind=spec['kind'], rule=spec.get('rule'),
                                                        m=m, gtol=gtol, build_seconds=time.perf_counter() - t1))
        return st['arms'][name]

    for s in cfg['settings']:
        for spec in cfg['arms'][s] + [x for x in cfg.get('parity_arms', []) if x['setting'] == s]:
            build(s, spec)
    print('ARMS BUILT', el(), flush=True)
    save()

    # --------------------------------------------------------------- FOM ----
    fom = Q.e.make_fom(L, dt)[0]
    tr = cfg['fom']['truth']

    def evolved_max(x):
        return float(np.max(np.asarray(x)[1:]))

    def errs_vs(f, g, n0):
        return [float(v) for v in jnp.linalg.norm((f - g).reshape(6, -1), axis=1) / n0]

    # ------------------------------------------------------- phase 1: cases ----
    pop = {s: dict(states=[], labels=[]) for s in cfg['settings']}
    audit = cfg.get('audit', {})
    for ci, (coh, c, ph) in enumerate(cases):
        u0 = jnp.asarray(Q.e.initial(L, ph))
        nu = float(ph[4])
        n0 = float(jnp.linalg.norm(u0))
        u0r = np.asarray(u0)[::s256, ::s256]
        n0r = float(np.linalg.norm(u0r))
        refs = {tag: load_ref(tag, coh, c) for tag in ('ST', 'S')}
        t1 = time.perf_counter()
        vt = fom(u0, nu, tr['ntol'], tr['ltol'])
        truth = jax.block_until_ready(vt[0])
        rn = np.asarray(vt[2])
        assert np.isfinite(np.asarray(truth)).all() and rn.max() <= tr['ntol'] * (1 + 1e-9), ('truth', coh, c)
        trr = np.asarray(truth)[:, ::s256, ::s256]
        frow = dict(cohort=coh, case=c, name=tr['name'], seconds=time.perf_counter() - t1, max_relative_residual=float(rn.max()),
                    newton_iterations=int(np.sum(np.asarray(vt[1]))), same_grid_restricted_sha256=sha(trr))
        for tag, r in refs.items():
            if r is not None:
                pe = [float(np.linalg.norm(x - y)) / n0r for x, y in zip(trr, r)]
                frow[f'ref_{tag}_per_time'] = pe
                frow[f'ref_{tag}_evolved'] = evolved_max(pe)
        rep['fom_rows'].append(frow)
        if coh == audit.get('cohort') and c in audit.get('cases', []):
            np.savez_compressed(out / f'audit_truth_{coh}{c}.npz', f257=trr,
                                **({'full': np.asarray(truth)} if L <= audit.get('full_max_mesh', 256) else {}))
        for s in cfg['settings']:
            st = S[s]
            dense_ok = c in cfg['dense_cases'].get(coh, []) if isinstance(cfg['dense_cases'].get(coh), list) else (
                cfg['dense_cases'].get(coh) == 'all')
            order = ([x for x in st['arms'] if x == 'gref'] + [x for x in st['arms'] if x == 'dense'] +
                     [x for x in st['arms'] if x not in ('gref', 'dense')])
            keepf = {}
            for name in order:
                arm = st['arms'][name]
                if arm['spec'].get('parity_only') and coh != arm['spec'].get('cohort'):
                    continue
                if name == 'dense' and not dense_ok:
                    continue
                t2 = time.perf_counter()
                v = arm['call'](u0, nu)
                jax.block_until_ready(v['fields'])
                secs = time.perf_counter() - t2
                f = v['fields']
                fin = bool(jnp.all(jnp.isfinite(f)))
                fr = np.asarray(f)[:, ::s256, ::s256]
                it, reason, rej = (np.asarray(v[k_]) for k_ in ('it', 'reason', 'rej'))
                row = dict(setting=s, arm=name, kind=arm['spec']['kind'], rule=arm['spec'].get('rule'), m=arm['m'],
                           gtol=arm['gtol'], cohort=coh, case=c, finite=fin, seconds_first=secs,
                           iterations_total=int(it.sum()), iterations_max=int(it.max()),
                           exits={REASONS[k_]: int(np.sum(reason == k_)) for k_ in REASONS},
                           rejected_total=int(rej.sum()), restricted_sha256=sha(fr))
                sg = errs_vs(f, truth, n0)
                row['same_grid_per_time'] = sg
                row['same_grid_evolved'] = evolved_max(sg)
                sgr = [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, trr)]
                row['same_grid_restricted_evolved'] = evolved_max(sgr)
                for tag, r in refs.items():
                    if r is not None:
                        pe = [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, r)]
                        row[f'ref_{tag}_per_time'] = pe
                        row[f'ref_{tag}_evolved'] = evolved_max(pe)
                for other in ('dense', 'gref'):
                    if other in keepf and other != name:
                        row[f'vs_{other}_per_time'] = errs_vs(f, keepf[other], n0)
                        row[f'vs_{other}_evolved'] = evolved_max(row[f'vs_{other}_per_time'])
                if name in ('dense', 'gref'):
                    keepf[name] = f
                rep['rows'].append(row)
                if name == cfg['population_arm']:
                    W = np.asarray(v['internal'])[1:]
                    pop[s]['states'].append(st['cmap'](W))
                    pop[s]['labels'] += [(coh, c, k + 1) for k in range(len(W))]
                if coh == audit.get('cohort') and c in audit.get('cases', []):
                    np.savez_compressed(out / f'audit_{s}_{name}_{coh}{c}.npz', f257=fr, internal=np.asarray(v['internal']),
                                        **({'full': np.asarray(f)} if L <= audit.get('full_max_mesh', 256)
                                           and name in audit.get('full_arms', []) else {}))
                del v, f
            keepf.clear()
        print('CASE', ci, coh, c, el(), flush=True)
        if ci % 4 == 3:
            save()
    rep['phases_seconds'] = dict(cases=el())
    save()

    # -------------------------------------------------------- phase 2: rho ----
    for s in cfg['settings']:
        st = S[s]
        C = np.concatenate(pop[s]['states'])
        labels = pop[s]['labels']
        np.savez_compressed(out / f'population_{s}.npz', C=C, labels=np.array([f'{a_}|{b_}|{k_}' for a_, b_, k_ in labels]))
        Cj = jnp.asarray(C)

        def values(kind, rule, chunk):
            """Tested advection of every population state. Off-mesh rules are streamed over point chunks (the
            sum over q is split, never the rule), so rules of 10^5-10^6 points never hold an (m, M) table."""
            f = jax.jit(jax.vmap(Q.tested_value(kind, L), in_axes=(0, None)))
            if kind in ('point', 'flux'):
                X, w = Q.offmesh_rule(rule)
                o = ops[st['M']]
                acc = np.zeros((len(C), st['M']))
                for q0 in range(0, len(X), cfg.get('point_chunk', 32768)):
                    q1 = q0 + cfg.get('point_chunk', 32768)
                    d = Q.offmesh_data(mdl, st['Rp'], X[q0:q1], w[q0:q1], L, o['kx'], o['ky'], kind)
                    acc += np.concatenate([np.asarray(f(Cj[i:i + chunk], d)) for i in range(0, len(C), chunk)])
                    del d
                return acc, len(X)
            d, m = rule_blocks(s, kind, rule)
            data = dict(st['base'], **d)
            return np.concatenate([np.asarray(f(Cj[i:i + chunk], data)) for i in range(0, len(C), chunk)]), m

        t1 = time.perf_counter()
        Tm, _ = values('dense', None, cfg.get('dense_chunk', 8))
        Tc, mref = values('point', cfg['gref'], 64)
        Tk, _ = values('point', cfg['gref_check'], 64)
        Tf, _ = values('flux', cfg['gref'], 64)
        rho = lambda V, T: np.linalg.norm(V - T, axis=1) / np.maximum(np.linalg.norm(T, axis=1), 1e-300)
        conv, cflux = rho(Tc, Tk), rho(Tf, Tc)
        rep['gates'][f'continuum_target_converged_{s}'] = dict(
            gref=cfg['gref'], check=cfg['gref_check'], rho_max=float(conv.max()), rho_median=float(np.median(conv)),
            flux_vs_point_rho_max=float(cflux.max()), flux_vs_point_rho_median=float(np.median(cflux)),
            bar=cfg['gref_bar'], passed=bool(conv.max() <= cfg['gref_bar'] and cflux.max() <= cfg['gref_bar']))
        res = {}
        perstate = {}
        for spec in cfg['rho_rules'][s]:
            if spec['kind'] == 'dense':
                V, m = Tm, (L - 1) ** 2
            else:
                V, m = values(spec['kind'], spec.get('rule'), 64 if spec['kind'] != 'mesh' else 16)
            r_c, r_m = rho(V, Tc), rho(V, Tm)
            perstate[spec['name']] = np.stack((r_c, r_m))
            ent = dict(kind=spec['kind'], rule=spec.get('rule'), m=m)
            for tag, r_ in (('cont', r_c), ('mesh', r_m)):
                i = int(np.argmax(r_))
                ent[tag] = dict(max=float(r_.max()), p95=float(np.quantile(r_, .95)), median=float(np.median(r_)),
                                argmax=list(labels[i]), pass_primary=bool(r_.max() <= .116), pass_tight=bool(r_.max() <= .06))
            res[spec['name']] = ent
        np.savez_compressed(out / f'rho_per_state_{s}.npz', **perstate)
        rep['rho'][s] = dict(states=len(C), population_arm=cfg['population_arm'], seconds=time.perf_counter() - t1,
                             mesh_vs_continuum_gap=dict(max=float(rho(Tm, Tc).max()), median=float(np.median(rho(Tm, Tc)))),
                             rules=res)
        print('RHO', s, len(C), round(time.perf_counter() - t1, 1), el(), flush=True)
        save()

    # ------------------------------------------------------ phase 3: timing ----
    tcfg = cfg['timing']
    rng = np.random.default_rng(int(tcfg.get('seed', 20261001)))
    tcases = [x for x in cases if x[0] == tcfg['cohort']][:tcfg['cases']]
    subjects = []
    for s in cfg['settings']:
        for name in tcfg['arms'][s]:
            if name in S[s]['arms']:
                subjects.append(('rom', s, name))
        subjects.append(('decode', s, None))
    for fs in cfg['fom']['timed']:
        subjects.append(('fom', fs['name'], fs))
    inv = []
    for rep_i in range(tcfg['reps']):
        order = [(sj, x) for sj in subjects for x in range(len(tcases))]
        rng.shuffle(order)
        for sj, x in order:
            coh, c, ph = tcases[x]
            u0 = jnp.asarray(Q.e.initial(L, ph))
            nu = float(ph[4])
            if sj[0] == 'decode':
                dfn, dd = S[sj[1]]['decode']
                Wd = jnp.asarray(.1 * np.random.default_rng(x).normal(size=(6, dd)))
                fn = lambda dfn=dfn, Wd=Wd: dfn(Wd)
            elif sj[0] == 'rom':
                fn = lambda sj=sj, u0=u0, nu=nu: S[sj[1]]['arms'][sj[2]]['call'](u0, nu)
            else:
                fs = sj[2]
                fn = lambda fs=fs, u0=u0, nu=nu: fom(u0, nu, fs['ntol'], fs['ltol'])
            if rep_i == 0 and x == 0:
                jax.block_until_ready(fn())          # warm (already compiled in phase 1; decode/fom may compile)
            Q.e.burn(tcfg['burn'])
            t1 = time.perf_counter()
            jax.block_until_ready(fn())
            inv.append(dict(subject=list(sj[:2]) + [sj[2] if sj[0] == 'rom' else (sj[2]['name'] if sj[0] == 'fom' else None)],
                            case=x, rep=rep_i, seconds=time.perf_counter() - t1))
        save_t = {}
        for d_ in inv:
            save_t.setdefault('|'.join(str(v) for v in d_['subject']), []).append(d_['seconds'])
        rep['timing'] = dict(cases=[f'{a_}{b_}' for a_, b_, _ in tcases], reps=tcfg['reps'], burn=tcfg['burn'],
                             invocations=inv, median_ms={k: 1e3 * float(np.median(v)) for k, v in save_t.items()})
        save()
    print('TIMING', el(), flush=True)
    rep['elapsed_seconds'] = el()
    rep['checkpoint_sha256_after'] = sha_file(Q.CKPT)
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('QSTUDY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
