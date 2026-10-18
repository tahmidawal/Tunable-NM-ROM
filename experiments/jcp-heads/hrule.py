"""jcp-heads rule selection (DESIGN 3.2 as amended by A1.4): one mesh, LSPG-BE dt0, the common ladder policy for every
arm group (a head size K = its seeds; a linear rung r = one arm).

For each group, rungs p = ladder[i] (selectable while p <= selectable_max) are tried in ascending order; the group's main
rule is the first q = Gauss p^2 at which at least `min_pass` arms of the group (heads: 2 of the 3 seeds, DESIGN A2.5;
a linear rung: its one arm) have, on the population (sub16): valid q and q+ solves,
rollout distance d(q, q+) <= 2.5e-4 on every case, rho_max(q -> q+) <= 0.116 on its own reached states (six outputs);
and on dev6: valid q++ solves and d(q+, q++) <= 2.5e-4 ("pass", never "unverified"). Solutions are cached per
(arm, rule) so q+ of one rung is reused as q of the next. Exhausted ladder -> "not quadrature-qualified".
Every saved array allows the offline recomputation (hscore.rule_check). One JSON + one npz per group.

    python hrule.py --config configs/<attempt>.json --out <dir>
config: {"mesh", "edges", "ladder": [16, 24, ...], "selectable_max": 192, "population": "sub16", "check": "dev6",
         "groups": [{"name", "M"?, "min_pass"?, "arms": [{"name", "kind", "head" | "Rp", "M"?}]}], "allow_cpu_smoke"?}
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2, DT0
import harm as HA
import jax
import jax.numpy as jnp

BAR_Q2, BAR_RHO = 2.5e-4, .116


def group_pass(passed, min_pass):
    """A2.5: a rung is the group's rule iff at least min_pass arms pass every predicate there."""
    return sum(bool(x) for x in passed) >= int(min_pass)


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
    L = int(cfg['mesh'])
    ladder = [int(x) for x in cfg['ladder']]
    smax = int(cfg['selectable_max'])
    assert ladder == sorted(ladder) and smax in ladder and ladder.index(smax) + 2 < len(ladder), 'selectable rung lacks q+/q++'
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT), cohorts=P.cohort_hashes(),
               groups={}, memory={}, elapsed={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    bank = P.Bank(L, cfg['edges'], nrb=cfg.get('bank_blocks'))
    rep['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem())
    cases = P.case_list(cfg)
    pop = P.select(cases, cfg.get('population', 'sub16'))
    chk = [i for i in P.select(cases, cfg.get('check', 'dev6')) if i in pop]
    s256 = L // 256
    U0 = {}
    for ci in pop:
        u0n = Q.e.initial(L, cases[ci][2])
        U0[ci] = (jnp.asarray(u0n), float(cases[ci][2][4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    cid = lambda ci: f'{cases[ci][0]}|{cases[ci][1]}'
    heads = {}
    sch = HA.sched('BE', DT0, 'prod')
    save()
    for grp in cfg['groups']:
        assert 'min_pass' in grp and 1 <= int(grp['min_pass']) <= len(grp['arms']), grp['name']
        t_g = time.perf_counter()
        arms = []
        for spec in grp['arms']:
            sp = dict(spec, rule=f'gauss{ladder[0]}', M=spec.get('M', grp.get('M')))
            arms.append(HA.make_arm(bank, sp, heads, HA.steps_of(DT0)))
        sol = {}                                   # (arm index, rule, case index) -> (W, f257, valid, coef)

        def solve(k, rule, ci):
            key = (k, rule, ci)
            if key not in sol:
                arm = arms[k]
                role = f'r_{rule}'
                if role not in arm.rules:
                    arm.add_role(role, rule)
                u0, nu, n0 = U0[ci]
                v = arm.run(role, 'BE', 'prod', u0, nu, DT0, record=False)
                jax.block_until_ready(v)
                W = np.asarray(v['W'])
                st_ = v['stats']
                S_ = HA.steps_of(DT0)
                diag = dict(exits=np.asarray(st_['exits']), step_gn=np.asarray(st_['step_gn'])[:S_],
                            step_reason=np.asarray(st_['step_reason'])[:S_], w0=np.asarray(v['w0']),
                            ic_reason=int(st_['ic_reason']) if 'ic_reason' in st_ else -1,
                            ic_gn=float(st_['ic_gn']) if 'ic_gn' in st_ else np.nan,
                            fields_finite=bool(np.isfinite(np.asarray(v['fields'])).all()))
                sol[key] = (W, P.restrict(v['fields'], L), HA.run_valid(v), arm.coef(W), diag)
            return sol[key]

        def rho_rows(k, rule, rule_p):
            arm = arms[k]
            for rl in (rule, rule_p):
                if f'r_{rl}' not in arm.rules:
                    arm.add_role(f'r_{rl}', rl)
            R = np.full((len(pop), 6), np.nan)
            for j, ci in enumerate(pop):
                C = solve(k, rule, ci)[3]
                for kk in range(6):
                    R[j, kk] = P.rho(arm.nl(C[kk], f'r_{rule}'), arm.nl(C[kk], f'r_{rule_p}'))
            return R

        def save_sol():
            """Archive every solve of the group so far (offline recomputation of every bar and of validity); called
            after every rung, so a timeout keeps the evidence of the finished rungs."""
            keys = sorted(sol)
            if not keys:
                return
            D = lambda f: np.array([sol[x][4][f] for x in keys])
            np.savez_compressed(out / f"SOL_{grp['name']}.npz",
                                keys=np.array([f'{arms[k].name}|{rl}|{cid(ci)}' for k, rl, ci in keys]),
                                W=np.array([sol[x][0] for x in keys]),      # fields are re-derivable (nphead / decode257)
                                valid=np.array([sol[x][2] for x in keys]), n0=np.array([U0[x[2]][2] for x in keys]),
                                exits=D('exits'), step_gn=D('step_gn'), step_reason=D('step_reason'), ic_reason=D('ic_reason'),
                                ic_gn=D('ic_gn'), fields_finite=D('fields_finite'),
                                w0=np.array([sol[x][4]['w0'] for x in keys], dtype=object) if len({sol[x][4]['w0'].shape for x in keys}) > 1
                                else D('w0'))
            rep['groups'][grp['name']] = dict(partial=True, trail=trail, population=[cid(c) for c in pop])
            save()

        trail = []
        chosen = None
        for i, p in enumerate(ladder):
            if p > smax:
                break
            q, qp, qpp = (f'gauss{x}' for x in ladder[i:i + 3])
            step = dict(rule=q, successor=qp, check=qpp, arms={})
            npass = 0
            for k, arm in enumerate(arms):
                r_ = dict(dist=[], valid_q=[], valid_qp=[], dist_check=[], valid_qpp=[])
                for ci in pop:
                    Wq, fq, vq = solve(k, q, ci)[:3]
                    Wp, fp, vp = solve(k, qp, ci)[:3]
                    r_['dist'].append(P.fdist(fq, fp, U0[ci][2]))
                    r_['valid_q'].append(vq)
                    r_['valid_qp'].append(vp)
                roll = all(r_['valid_q']) and all(r_['valid_qp']) and max(r_['dist']) <= BAR_Q2
                rho = rho_rows(k, q, qp) if roll else None
                r_['rho_max'] = None if rho is None else float(np.max(rho)) if np.isfinite(rho).all() else float('inf')
                rho_ok = rho is not None and r_['rho_max'] <= BAR_RHO
                if roll and rho_ok:
                    for ci in chk:
                        fp = solve(k, qp, ci)[1]
                        _, fpp, vpp = solve(k, qpp, ci)[:3]
                        r_['dist_check'].append(P.fdist(fp, fpp, U0[ci][2]))
                        r_['valid_qpp'].append(vpp)
                check_ok = bool(r_['dist_check']) and all(r_['valid_qpp']) and max(r_['dist_check']) <= BAR_Q2
                r_.update(rollout_pass=bool(roll), rho_pass=bool(rho_ok), check_pass=bool(check_ok),
                          passed=bool(roll and rho_ok and check_ok), max_dist=max(r_['dist']))
                step['arms'][arm.name] = r_
                npass += int(r_['passed'])
                if rho is not None:
                    np.savez_compressed(out / f"RHO_{grp['name']}_{arm.name}_{q}.npz", rho=rho, case_ids=np.array([cid(c) for c in pop]))
            need = int(grp['min_pass'])                 # required (A2.5: heads 2 of 3 seeds; a linear rung 1)
            ok_all = group_pass([r_['passed'] for r_ in step['arms'].values()], need)
            step.update(group_pass=bool(ok_all), n_pass=npass, min_pass=need)
            trail.append(step)
            print('RUNG', grp['name'], q, 'pass' if ok_all else 'fail', el(), flush=True)
            save_sol()
            if ok_all:
                chosen = dict(main=q, qp=qp, qpp=qpp, passing_arms=[n_ for n_, r_ in step['arms'].items() if r_['passed']],
                              failing_arms=[n_ for n_, r_ in step['arms'].items() if not r_['passed']])
                break
            for k, arm in enumerate(arms):        # tables of rule q are no longer needed
                arm.drop_role(f'r_{q}')
            gc.collect()
        save_sol()
        rep['groups'][grp['name']] = dict(arms=[a_.describe() for a_ in arms], trail=trail, chosen=chosen, partial=False,
                                          status='selected' if chosen else 'not quadrature-qualified (ladder exhausted)',
                                          population=[cid(c) for c in pop], check_population=[cid(c) for c in chk],
                                          seconds=time.perf_counter() - t_g, mem=P.mem())
        save()
        del arms, sol
        gc.collect()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HRULE COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
