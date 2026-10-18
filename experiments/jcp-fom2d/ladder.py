"""jcp-fom2d C1 ladder job (DESIGN §3). One job runs a list of FD groups (family, L, dt) and PS configurations.

FD group: calibration on the calibration cases at every tolerance level of the config (t0 = reference; DESIGN §3 rule,
evaluated offline again by match2.py from the saved evidence), then production on every case at the chosen level
(calibration outputs at the chosen level are reused for the calibration cases: identical program and inputs), then
warm screening timings (lattice contract). PS configuration: production on every case, warm screening timings.
Outputs per configuration: F_<key>.npz (lattice fields, case ids) and its record in result.json (per-case convergence
evidence, in-job E0 scores, hashes, timings, prepared-input seconds, compile seconds). Resumable: a configuration whose
npz exists is loaded, not rerun.

    python ladder.py --config configs/<attempt>.json --out <dir>
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
import jax.numpy as jnp

import fomrun2 as FR
import refstore as RS
import cohorts as CH
import gates as G

TOL_CAL = 1e-6        # calibration bar: lattice difference to t0 <= 1e-6 n0 at every output time on every calibration case


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


_BURN = jax.jit(lambda a: jnp.tanh(a @ a) * .5)


def burn(seconds):
    a = jnp.ones((512, 512), jnp.float64) * .01
    jax.block_until_ready(_BURN(a))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def mem():
    try:
        s = jax.devices()[0].memory_stats() or {}
        return dict(in_use=int(s.get('bytes_in_use', -1)), peak=int(s.get('peak_bytes_in_use', -1)))
    except Exception:  # noqa: BLE001
        return dict(in_use=-1, peak=-1)


def case_list(which):
    """[(cohort, case, descriptor)]: all38 = dev6 then val32; or an explicit [[cohort, case], ...] list."""
    co = {c: CH.Cohort(2, c) for c in ('dev6', 'val32')}
    if which == 'all38':
        keys = [('dev6', j) for j in range(6)] + [('val32', j) for j in range(32)]
    else:
        keys = [tuple(k) for k in which]
    assert all(k[0] in co for k in keys), 'only dev6 / val32 are used'
    return [(c, int(j), co[c].D[int(j)]) for c, j in keys]


def score(F, coh, j, kinds):
    out = {}
    for kind in kinds:
        s = RS.score(np.asarray(F), 2, coh, j, kind)
        out[kind] = dict(e=s['e'], e_t=s['e_t'], eps_ref=s['eps_ref'], eligible=s['eligible'], outcome=s['outcome'])
    return out


def n0_of(coh, j):
    _, meta = RS.load(2, coh, j, 'ST')
    return float(meta['judge']['n0'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest' and jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    rp = out / 'result.json'
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    gpu = jax.devices()[0].device_kind
    rep.update(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=gpu, jax_version=jax.__version__, store_version=RS.store_version(),
               complete=False)
    rep.setdefault('configs', {})
    rep.setdefault('gates', {})
    rep.setdefault('restarts', 0)
    rep['restarts'] += 1 if rp.exists() else 0
    save = lambda: rp.write_text(json.dumps(clean(rep), indent=1) + '\n')
    cases = case_list(cfg.get('cases', 'all38'))
    cal = case_list(cfg.get('calibration_cases', [['dev6', 0], ['dev6', 1], ['dev6', 2], ['dev6', 3]]))
    warm = case_list(cfg.get('warm_cases', [['dev6', 0], ['dev6', 1]]))
    reps, bsec = int(cfg.get('warm_reps', 3)), float(cfg.get('burn', 0.05))
    levels = [str(x) for x in cfg.get('levels', ['0', '1', '2', '3'])]
    assert levels[0] == '0'
    cid = lambda c: f'{c[0]}|{c[1]}'
    foms = FR.FOMs()
    N0 = {cid(c): n0_of(c[0], c[1]) for c in cases + cal}

    # ---- gates (smoke) -------------------------------------------------------------------------------------------
    for gname in cfg.get('gates', []):
        if gname in rep['gates']:
            continue
        t = time.perf_counter()
        d6 = CH.Cohort(2, 'dev6').D
        if gname == 'fd1':
            r = G.g_fd1()
        elif gname == 'fd2':
            r = G.g_fd2(d6[0])
        elif gname == 'e0':
            r = dict(cases=[G.g_e0('dev6', j, d6[j]) for j in (0, 1)])
            r['passed'] = all(x['passed'] for x in r['cases'])
        elif gname == 'ps1':
            r = G.g_ps1(d6[:2])
        elif gname == 'ps2':
            r = G.g_ps2(CH.Cohort(2, 'val32').D[0])
        elif gname == 'int':
            r = G.g_int()
        elif gname == 'ct':
            r = G.g_ct(foms, [('fd', 'CC', 128, 0.01, '2'), ('fd', 'CC', 1024, 0.01, '2'), ('fd', 'UB', 2048, 0.005, '2'),
                              ('ps', 32, 0.01)], d6[0], M=1024)
            r4 = G.g_ct(foms, [('fd', 'CC', 256, 0.01, '2')], d6[0], M=4096)
            r = dict(values=dict(r['values'], **{k_ + '@4096': v for k_, v in r4['values'].items()}),
                     passed=bool(r['passed'] and r4['passed']))
        else:
            raise KeyError(gname)
        r['seconds'] = time.perf_counter() - t
        rep['gates'][gname] = r
        print('GATE', gname, r.get('passed', r.get('fires')), round(r['seconds'], 1), flush=True)
        save()

    # ---- configurations -------------------------------------------------------------------------------------------
    def run_cases(sp, clist, keep_fields=True):
        """Untimed runs; returns per-case records and the lattice fields."""
        recs, Fs = [], []
        for c in clist:
            x, tprep = FR.prepare(sp, c[2])
            t = time.perf_counter()
            F, info = foms.run(sp, x, float(c[2][4]), 'lattice')
            F = jax.block_until_ready(F)
            ts = time.perf_counter() - t
            ok, ev = FR.FOMs.converged(sp, info, F)
            Fn = np.asarray(F)
            recs.append(dict(case=cid(c), converged=ok, evidence=ev, sha256=sha(Fn), seconds_first=ts, prep_seconds=tprep))
            Fs.append(Fn if keep_fields else None)
        return recs, Fs

    def warm_times(sp, Fprod):
        rows = []
        for c in warm:
            x, _ = FR.prepare(sp, c[2])
            fn = foms.callable(sp, x, float(c[2][4]), 'lattice')
            jax.block_until_ready(fn())                         # warm (compiled already; first-call effects)
            for r_ in range(reps):
                burn(bsec)
                t = time.perf_counter()
                F = jax.block_until_ready(fn())
                ts = time.perf_counter() - t
                Fn = np.asarray(F)
                ref = Fprod[cid(c)]
                rows.append(dict(case=cid(c), rep=r_, seconds=ts, bitwise=sha(Fn) == sha(ref), rel=G.rel(Fn, ref)))
        hok = all(r['bitwise'] or r['rel'] <= 1e-12 for r in rows)
        return dict(times=rows, hashes_ok=hok, all_bitwise=all(r['bitwise'] for r in rows),
                    median=float(np.median([r['seconds'] for r in rows])))

    def finish(k, sp, prod, Fs, extra):
        kinds = ['ST'] + (['S'] if sp[0] == 'fd' and sp[1] == 'UB' and abs(sp[3] - 0.005) < 1e-12 else [])
        for r, c, F in zip(prod, cases, Fs):
            assert r['case'] == cid(c)
            r['scores'] = score(F, c[0], c[1], kinds)
        np.savez(out / f"F_{k.replace('|', '_')}.npz", case_ids=np.array([cid(c) for c in cases]),
                 fields=np.stack(Fs))
        Fprod = {cid(c): F for c, F in zip(cases, Fs)}
        rec = dict(key=k, spec=list(sp), production=prod, warm=warm_times(sp, Fprod), mem=mem(), **extra)
        if sp[0] == 'ps':                                  # A1.1 E0 stability label (lattice proxy for u_max)
            um = [float(np.max(np.abs(F))) for F in Fs]
            bound = [0.66 * 2.8 / (2 * 2 * np.pi * sp[1] * u) if u > 0 else np.inf for u in um]
            rec['stability'] = dict(umax_lattice=um, dt_bound=bound, label_pass=bool(all(sp[2] <= b for b in bound)),
                                    note='E0 stability label (lattice proxy); not an admissibility condition')
        rec['admissible_injob'] = bool(all(r['converged'] for r in prod) and rec['warm']['hashes_ok'])
        rep['configs'][k] = rec
        save()
        print('CONFIG', k, 'adm', rec['admissible_injob'], 'warm', round(rec['warm']['median'], 4), el(), mem(), flush=True)

    for fam, L, dt in [tuple(g) for g in cfg.get('fd', [])]:
        dt = FR.canon_dt(dt)
        gkey = f'fd|{fam}|L{L}|{dt:.8g}'
        gold = rep.get('groups', {}).get(gkey)
        glev = None if gold is None else (gold.get('recovery') or {}).get('level', gold['chosen'])
        if gold is not None and (gold['chosen'] is None or FR.key(('fd', fam, int(L), dt, glev)) in rep['configs']):
            continue                                       # resumable: finished group (calibration is simply redone otherwise)
        t0 = time.perf_counter()
        calib = {}
        Fcal = {}
        for lev in levels:
            sp = ('fd', fam, int(L), dt, lev)
            recs, Fs = run_cases(sp, cal)
            calib[lev] = recs
            Fcal[lev] = Fs
        # DESIGN §3 rule (recomputed offline by match2.py from this evidence)
        t0ok = all(r['converged'] for r in calib['0'])
        diffs = {}
        for lev in levels[1:]:
            diffs[lev] = [max(float(np.linalg.norm(Fcal[lev][i][k] - Fcal['0'][i][k])) / N0[cid(c)] for k in range(1, 6))
                          for i, c in enumerate(cal)]
        chosen = None
        if t0ok:
            chosen = '0'
            for lev in sorted(levels[1:], key=lambda s: -int(s)):        # loosest first
                if all(r['converged'] for r in calib[lev]) and max(diffs[lev]) <= TOL_CAL:
                    chosen = lev
                    break
        group = dict(key=gkey, calibration={lev: calib[lev] for lev in levels}, diffs=diffs, t0_ok=t0ok, chosen=chosen,
                     calib_seconds=time.perf_counter() - t0)
        np.savez(out / f"C_{gkey.replace('|', '_')}.npz", levels=np.array(levels), case_ids=np.array([cid(c) for c in cal]),
                 n0=np.array([N0[cid(c)] for c in cal]), **{f'F{lev}': np.stack(Fcal[lev]) for lev in levels})
        rep.setdefault('groups', {})[gkey] = group
        save()
        print('CALIB', gkey, 't0_ok', t0ok, 'chosen', chosen, {k_: max(v) for k_, v in diffs.items()}, el(), flush=True)
        if chosen is None:
            continue
        sp = ('fd', fam, int(L), dt, chosen)
        k = FR.key(sp)
        calset = {cid(c): (r, F) for c, r, F in zip(cal, calib[chosen], Fcal[chosen])}
        rest = [c for c in cases if cid(c) not in calset]
        recs, Fs = run_cases(sp, rest)
        bykey = {r['case']: (r, F) for r, F in zip(recs, Fs)}
        bykey.update({kk: (dict(v[0], reused_from_calibration=True), v[1]) for kk, v in calset.items()})
        prod = [bykey[cid(c)][0] for c in cases]
        Fp = [bykey[cid(c)][1] for c in cases]
        extra = dict(group=gkey)
        if not all(r['converged'] for r in prod) and chosen != '0':
            # A1.1/A2.6 production recovery: rerun once, all cases, at the next tighter level that itself passed the
            # calibration predicate (t0 passes by definition)
            bad = [r['case'] for r in prod if not r['converged']]
            passing = ['0'] + [lv for lv in levels[1:] if all(r['converged'] for r in calib[lv]) and max(diffs[lv]) <= TOL_CAL]
            lev2 = max((lv for lv in passing if int(lv) < int(chosen)), key=int)
            sp = ('fd', fam, int(L), dt, lev2)
            k = FR.key(sp)
            prod, Fp = run_cases(sp, cases)
            extra.update(recovered_from=chosen, recovery_level=lev2, unconverged_at_chosen=bad)
            group['recovery'] = dict(level=lev2, unconverged_at_chosen=bad)
        finish(k, sp, prod, Fp, extra)

    for N, dt in [tuple(g) for g in cfg.get('ps', [])]:
        sp = ('ps', int(N), FR.canon_dt(dt))
        k = FR.key(sp)
        if k in rep['configs']:
            continue
        recs, Fs = run_cases(sp, cases)
        finish(k, sp, recs, Fs, dict())

    # ---- controls / pilots (smoke) ---------------------------------------------------------------------------------
    if cfg.get('k_loose') and 'k_loose' not in rep:
        fam, L, dt = cfg['k_loose']
        sp0, spx = ('fd', fam, L, FR.canon_dt(dt), '0'), ('fd', fam, L, FR.canon_dt(dt), 'X')
        r0, F0 = run_cases(sp0, cal)
        rx, Fx = run_cases(spx, cal)
        dx = [max(float(np.linalg.norm(Fx[i][k] - F0[i][k])) / N0[cid(c)] for k in range(1, 6)) for i, c in enumerate(cal)]
        rep['k_loose'] = dict(spec=list(spx), converged=[r['converged'] for r in rx], diffs=dx, evidence=[r['evidence'] for r in rx],
                              fires=bool(max(dx) > TOL_CAL), rejecting_predicate='agreement (1e-6 n0)',
                              convergence_would_also_reject=not all(r['converged'] for r in rx))
        save()
        print('K_LOOSE', rep['k_loose']['fires'], dx, flush=True)
    if cfg.get('k_swap') and 'k_swap' not in rep:
        k = next(iter(rep['configs']))
        F = np.load(out / f"F_{k.replace('|', '_')}.npz")['fields'][0]
        c = cases[0]
        rep['k_swap'] = dict(config=k, **G.k_swap(F, c[0], c[1], 1 if c[1] != 1 else 2))
        save()
    for pk in cfg.get('pilots', []):                       # memory / cost pilot: one case each, peak memory recorded
        if pk in rep.setdefault('pilots', {}):
            continue
        sp = FR.parse(pk)
        recs, _ = run_cases(sp, cases[:1], keep_fields=False)
        recs2, _ = run_cases(sp, cases[:1], keep_fields=False)       # warm second call (compile excluded)
        rep['pilots'][pk] = dict(record=recs[0], warm=recs2[0], mem=mem())
        save()
        print('PILOT', pk, round(recs[0]['seconds_first'], 2), round(recs2[0]['seconds_first'], 3), mem(), flush=True)
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('LADDER COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
