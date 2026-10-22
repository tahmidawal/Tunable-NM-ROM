"""jcp-p1-scaling 2D kernel job (P1a; DESIGN 2.4, 2.5, 4.2, 6; A1.10). One GPU, mesh 1024^2.

Fixtures: combined W_R512.npz (R512, gauss192, LSPG-CN-R dt0), run 'LSPG|CNR|0.005|prod', dev6 cases 0-5, output
indices 1 and 5, truncated to each r ("kernel fixtures, not reached states of the smaller-r solver"); the archive
sha256 is pinned in the config. Panels:
  fixed   (M, m) = (frozen 1536-mode list, gauss96 = 9216 points), every r       -> the H-cost-r verdict panel
  fixedm  m = 9216 (gauss96), M = 4 r (frozen list prefix)                     -> descriptive
  nominal M = 4 r with the frozen rule per r (r >= 128)                         -> descriptive
Per (panel, r, fixture, component): 0.1 s burn, 3 warm, 30 synchronised calls (all raw times saved). Reference-kernel
drift (fixed-panel chol at the largest r) at the start and end of each panel; a drifting panel is re-run once.
Dispatch-sync control: two separately synchronised dispatches vs one, for each verdict component at the largest r.

    python p1kern.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2
import p1comp as PC
import jax
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, panels={}, sync={}, drift={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    fx = cfg['fixtures']
    assert QS.sha_file(fx['file']) == fx['sha256'], 'fixture file hash mismatch'
    z = np.load(fx['file'])
    fixtures, flabels = np.asarray(z['W']), [str(x) for x in z['labels']]
    rep['fixtures'] = dict(labels=flabels, case_of=[l_.rsplit('|', 1)[0] for l_ in flabels], R=int(fixtures.shape[1]))
    L = int(cfg['mesh'])
    bank = P.Bank(L, cfg['edges'])
    ops = bank.ops('frozen')
    nu, dt = float(cfg['nu']), float(P.DT0)
    ranks = [int(r) for r in cfg['ranks']]
    Rmax = max(ranks)
    rule_cache = {}

    def rule_tabs(rule, M):
        """Gq, Gs at Rmax columns and Psi at M columns for a rule (built once per (rule, M))."""
        key = (rule, M)
        if key not in rule_cache:
            X, w, _ = P.rule_points(rule)
            blk = Q.offmesh_data(bank.mdl, Rmax, X, w, L, ops['kx'][:M], ops['ky'][:M], 'point')
            rule_cache[key] = blk
        return rule_cache[key]

    def make_tabs(panel):
        cache = {}

        def tabs(r):
            if r not in cache:
                if panel == 'fixed':
                    M, rule = int(cfg['fixed_M']), cfg['fixed_rule']
                elif panel == 'fixedm':
                    M, rule = 4 * r, cfg['fixed_rule']
                else:
                    M, rule = 4 * r, cfg['nominal_rules'][str(r)]
                blk = rule_tabs(rule, M)
                cache.clear()       # keep one rank resident at a time (memory)
                cache[r] = tuple(jax.block_until_ready(jnp.array(x)) for x in
                                 (ops['Arot'][:M, :r], ops['lam'][:M], blk['Gq'][:, :r], blk['Gs'][:, :r], blk['Psi']))
                cache[r] = cache[r] + (M, len(blk['Gq']), rule)
            return cache[r][:5]

        def meta(r):
            tabs(r)
            return cache[r][5:]
        return tabs, meta

    def ref_kernel():
        tabs, _ = make_tabs('fixed')
        Ar, lr, Gq, Gs, Psi = tabs(Rmax)
        c = jnp.asarray(fixtures[0][:Rmax])
        Jf = Ar + (dt / (1 + dt * nu * lr))[:, None] * PC.k_J(c, Gq, Gs, Psi)
        Hm = PC.k_JTJ(Jf)
        g = Jf.T @ PC.k_res(c, Ar, lr, Gq, Gs, Psi, nu, dt)
        return float(np.median(PC._time(PC.k_chol, (Hm, g, jnp.asarray(1e-6)), 30, cfg['burn'])))

    QS.burn(.05)
    for panel in cfg['panels']:
        pr = [r for r in ranks if panel != 'nominal' or r >= 128]
        for attempt in range(2):
            d0 = ref_kernel()
            tabs, meta = make_tabs(panel)
            rows = PC.kernel_panel(None, None, tabs, fixtures, pr, nu, dt, reps=int(cfg['reps']), burn=float(cfg['burn']),
                                   seed=int(cfg['seed']) + attempt)
            for r_ in rows:
                M, m, rule = meta(r_['r'])
                r_.update(M=M, m=m, rule=rule, case=rep['fixtures']['case_of'][r_['fixture']],
                          flops=PC.flops_bytes(r_['comp'], m, M, r_['r'])[0] if r_['comp'] in ('N', 'J', 'JTJ', 'chol') else None,
                          bytes=PC.flops_bytes(r_['comp'], m, M, r_['r'])[1] if r_['comp'] in ('N', 'J', 'JTJ', 'chol') else None)
            d1 = ref_kernel()
            ok = (1 / 1.1) <= d1 / d0 <= 1.1
            rep['drift'].setdefault(panel, []).append(dict(start=d0, end=d1, ratio=d1 / d0, ok=ok))
            rep.setdefault('all_attempts', {}).setdefault(panel, []).append(rows)   # every attempt kept (code audit 1)
            if ok:
                break
        rep['panels'][panel] = dict(rows=rows, drift_ok=ok, attempts=attempt + 1, ranks=pr)
        save()
        print('PANEL', panel, el(), flush=True)
    # dispatch-sync control at the largest rank of the fixed panel
    tabs, _ = make_tabs('fixed')
    Ar, lr, Gq, Gs, Psi = tabs(Rmax)
    c = jnp.asarray(fixtures[0][:Rmax])
    Jf = Ar + (dt / (1 + dt * nu * lr))[:, None] * PC.k_J(c, Gq, Gs, Psi)
    Hm = PC.k_JTJ(Jf)
    g = Jf.T @ PC.k_res(c, Ar, lr, Gq, Gs, Psi, nu, dt)
    for comp, (fn, args) in dict(N=(PC.k_N, (c, Gq, Gs, Psi)), J=(PC.k_J, (c, Gq, Gs, Psi)), JTJ=(PC.k_JTJ, (Jf,)),
                                 chol=(PC.k_chol, (Hm, g, jnp.asarray(1e-6)))).items():
        s_ = PC.dispatch_sync(fn, args, reps=10, burn=float(cfg['burn']))
        s_['passed'] = bool(1.8 <= s_['ratio_median'] <= 2.2)
        rep['sync'][comp] = s_
    # A3.3: the kernel components equal the solver's tested advection and its Jacobian (qcore point form) on a fixture
    data = dict(Gq=Gq, Gs=Gs, Psi=Psi)
    nl, nlJ = Q.tested_value('point', L), Q.tested_jac('point', L)
    eqN, eqJ = [], []
    for cf in fixtures:                                   # A3.3: on every fixture, at the largest rank
        cc_ = jnp.asarray(cf[:Rmax])
        vN, vJ = nl(cc_, data), nlJ(cc_, data, None)[1]
        eqN.append(float(jnp.max(jnp.abs(PC.k_N(cc_, Gq, Gs, Psi) - vN)) / jnp.max(jnp.abs(vN))))
        eqJ.append(float(jnp.max(jnp.abs(PC.k_J(cc_, Gq, Gs, Psi) - vJ)) / jnp.max(jnp.abs(vJ))))
    rep['component_equivalence'] = dict(N=max(eqN), J=max(eqJ), per_fixture=dict(N=eqN, J=eqJ),
                                        passed=bool(max(eqN) <= 1e-12 and max(eqJ) <= 1e-12))
    rep['memory'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P1KERN COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
