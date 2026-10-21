"""imp-multimesh timing job J6 (DESIGN 5, A1.7, A2.3, A2.5, A2.7). One H200, one job, per mesh:

  A1  ours: bank build, operators, settings (timed setup), AOT compile of the recording fused query per scheme,
      warm call, one timed call per case (38); identity vs the J5 accuracy run (W, every state, exits, iterations)
  S   sentinel: ours R128 BE dev6/0 x 3 on a prefix bank (edges [0, 128]) kept resident during B
  B   competitors, one arm at a time (basis loaded or resampled, trust from transformed codes, rules from the build
      job), per configuration: arm data (timed), AOT compile, warm call, two timed calls per case; identity vs the
      build job (C trajectory, exits, iterations); a sentinel after every arm
  A2  ours again (bank rebuilt), one timed call per case
Drift: every sentinel median and every A2/A1 median ratio within [1/1.1, 1.1]; otherwise the mesh is repeated once,
then labelled invalid. Timed region: device-resident input -> six full-field outputs on the device.

    python mmtime.py <config.json> <outdir>
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

BAND = (1 / 1.1, 1.1)


def in_band(x):
    return bool(np.isfinite(x) and BAND[0] <= x <= BAND[1])


def time_calls(fn, xs, inject=0.0, burn_s=0.05, reps=1):
    """One timed call per input per rep; burn before each; `inject` seconds of host sleep inside the timed region
    (control 7 only)."""
    import pipeline as PL
    out = []
    for _ in range(reps):
        for u, nu in xs:
            PL.burn(burn_s)
            t0 = time.perf_counter()
            v = fn(u, nu)
            jax.block_until_ready(v)
            if inject:
                time.sleep(inject)
            out.append(time.perf_counter() - t0)
            del v
    return out


def main():
    import mmcore as M
    import mmbuild as B
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'jcp-p1-scaling'))
    import p1common as P
    H, BD, RM = M.H, M.BD, M.RM
    D = 2
    log = B.log
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    B.check_env(cfg)
    rec = B.Rec(out / 'time.json', dict(config=cfg, gpu=jax.devices()[0].device_kind, job=os.environ.get('SLURM_JOB_ID'),
                                        commit=os.environ.get('SOURCE_COMMIT'), meshes={}, complete=False))
    fams = B.families()
    cases = B.cases_all('all38')
    NREC = 51
    sch_c = RM.schedule(D, 'CNR', 0.005, 'prod')

    def ours_block(L, mc, tag, reps=1):
        res = {}
        t0 = time.perf_counter()
        bank = P.Bank(L, mc['edges'])
        res['bank_seconds'] = time.perf_counter() - t0
        res['bank_build_seconds'] = bank.build_seconds
        t0 = time.perf_counter()
        bank.ops('frozen')
        res['ops_seconds'] = time.perf_counter() - t0
        allc = P.case_list({})
        U = [(jnp.asarray(P.Q.e.initial(L, ph)), float(ph[4])) for (_, _, ph) in allc]
        res['mem_bank'] = B.dev_mem()
        for X in mc['configs']:
            t0 = time.perf_counter()
            st = P.Setting(bank, X['name'], X['Rp'], X['rule'], 'frozen', roles={}, nrec=NREC)
            jax.block_until_ready(st.data[('main', 'LSPG')])
            setup = time.perf_counter() - t0
            st.check('main', bank)
            q = st.q['LSPG']['fused']
            dat = st.data[('main', 'LSPG')]
            for sc in X['schemes']:
                key = f"{X['name']}|{sc}"
                sch = st.sched('LSPG', sc, P.DT0, 'prod')
                t0 = time.perf_counter()
                q.lower(U[0][0], jnp.float64(U[0][1]), dat, st.cold, sch).compile()
                comp = time.perf_counter() - t0
                jax.block_until_ready(q(U[0][0], jnp.float64(U[0][1]), dat, st.cold, sch))
                fn = lambda u, nu: q(u, jnp.float64(nu), dat, st.cold, sch)
                obs = time_calls(fn, U, reps=reps)
                ident = None
                j5 = mc.get('j5_dir')
                if j5 and tag == 'A1':
                    Rz = np.load(Path(j5) / f"R_{X['name']}_{sc}.npz")
                    bad = []
                    for ci, (u, nu) in enumerate(U):
                        v = q(u, jnp.float64(nu), dat, st.cold, sch)
                        Wm = np.asarray(v['W'])
                        sts = np.asarray(v['stats']['states'])
                        dW = float(np.max(np.abs(Wm - Rz['W'][ci])) / max(np.max(np.abs(Rz['W'][ci])), 1e-300))
                        dS = float(np.max(np.abs(sts - Rz['states'][ci])) / max(np.max(np.abs(Rz['states'][ci])), 1e-300))
                        it_ok = bool(np.array_equal(np.asarray(v['stats']['step_it'])[:NREC - 1], Rz['it'][ci]))
                        rs_ok = bool(np.array_equal(np.asarray(v['stats']['step_reason'])[:NREC - 1], Rz['reason'][ci]))
                        if not (dW <= 1e-10 and dS <= 1e-10 and it_ok and rs_ok):
                            bad.append(dict(case=ci, dW=dW, dS=dS, it_ok=it_ok, reason_ok=rs_ok))
                    ident = dict(n=len(U), mismatches=bad, ok=not bad)
                res[key] = dict(setup_seconds=setup, compile_seconds=comp, obs=obs, mean=float(np.mean(obs)),
                                median=float(np.median(obs)), identity=ident, table_bytes=st.table_bytes)
                log(f'  ours {tag} L={L} {key}: mean {res[key]["mean"]*1e3:.2f} ms compile {comp:.1f}s '
                    f'identity {None if ident is None else ident["ok"]}')
            del st, q, dat
            gc.collect()
        res['mem_peak'] = B.dev_mem()
        del bank, U
        gc.collect()
        return res

    def sentinel(sb, L, U0):
        st = sb['st']
        sch = st.sched('LSPG', 'BE', P.DT0, 'prod')
        q = st.q['LSPG']['fused']
        dat = st.data[('main', 'LSPG')]
        fn = lambda u, nu: q(u, jnp.float64(nu), dat, st.cold, sch)
        jax.block_until_ready(fn(*U0))
        o = time_calls(fn, [U0] * 3)
        return float(np.median(o))

    def arm_block(L, arm, sb, U0):
        res = dict(configs={})
        Phi_s, C_s, rules_s, trust_s = B.load_source(arm['source_dir'])
        s = int(arm['source'])
        rmax = max(int(c.split('|')[0][1:]) for c in arm['configs'])
        t0 = time.perf_counter()
        if s == L:
            Phi = jax.block_until_ready(jnp.asarray(np.asarray(Phi_s[:, :rmax])))
            R = None
        else:
            Phi, R, info = M.resample_basis(np.asarray(Phi_s[:, :rmax]), s, L)
            assert info['orth_ok']
        res['basis_seconds'] = time.perf_counter() - t0
        lat = np.load(Path(arm['build_dir']) / f"Phi_lat_{arm['name']}.npy", mmap_mode='r')
        mine = np.asarray(Phi[jnp.asarray(H.lattice_nodes(D, L))])
        res['basis_identity'] = float(np.max(np.abs(mine - np.asarray(lat[:, :rmax]))))
        trust = {}
        for cid in arm['configs']:
            r = int(cid.split('|')[0][1:])
            if r in trust:
                continue
            trust[r] = trust_s[r] if s == L else BD.trust_radii(M.transform_codes(C_s, s, L, r, R), [r], D)[r]
        cell = M.Cell(L, Phi, trust, log=log)
        xs = [(jnp.asarray(fams[coh].ic(L, j)), fams[coh].nu(j)) for coh, j in cases]
        for cid in arm['configs']:
            r = int(cid.split('|')[0][1:])
            k = int(cid.split('|')[1][1:])
            z = np.load(Path(arm['build_dir']) / f"W_{arm['name']}_r{r}_k{k}.npz")
            t0 = time.perf_counter()
            data, _ = cell.ecsw_data(r, z['nodes'], z['weights'])
            t_data = time.perf_counter() - t0
            q = RM.make_query('ecsw', D, L, 'common', trust=trust[r])
            t0 = time.perf_counter()
            q.lower(xs[0][0], jnp.float64(xs[0][1]), data, sch_c).compile()
            comp = time.perf_counter() - t0
            jax.block_until_ready(q(xs[0][0], jnp.float64(xs[0][1]), data, sch_c)['fields'])
            fn = lambda u, nu: q(u, jnp.float64(nu), data, sch_c)['fields']
            obs = time_calls(fn, xs, reps=2)
            bad = []
            for (coh, j), (u, nu) in zip(cases, xs):
                v = q(u, jnp.float64(nu), data, sch_c)
                Cm = np.asarray(v['C'])[:51]
                Cr = z[f'C|{coh}|{j}']
                dC = float(np.max(np.abs(Cm - Cr)) / max(np.max(np.abs(Cr)), 1e-300))
                it_ok = bool(np.array_equal(np.asarray(v['stats']['it']), z[f'it|{coh}|{j}']))
                rs_ok = bool(np.array_equal(np.asarray(v['stats']['reason']), z[f'reason|{coh}|{j}']))
                if not (dC <= 1e-10 and it_ok and rs_ok):
                    bad.append(dict(case=[coh, j], dC=dC, it_ok=it_ok, reason_ok=rs_ok))
            res['configs'][cid] = dict(armdata_seconds=t_data, compile_seconds=comp, obs=obs, mean=float(np.mean(obs)),
                                       median=float(np.median(obs)), identity=dict(ok=not bad, mismatches=bad),
                                       outliers=int(np.sum(np.asarray(obs) > 3 * np.median(obs))))
            log(f'  {arm["name"]} L={L} {cid}: mean {res["configs"][cid]["mean"]*1e3:.2f} ms compile {comp:.1f}s '
                f'identity {not bad}')
            del data, q
            gc.collect()
        res['mem_peak'] = B.dev_mem()
        del cell, Phi
        gc.collect()
        res['sentinel_after'] = sentinel(sb, L, U0)
        return res

    for Ls, mc in cfg['meshes'].items():
        L = int(Ls)
        for attempt in (1, 2):
            log(f'MESH {L} attempt {attempt}')
            M_ = dict(attempt=attempt)
            M_['A1'] = ours_block(L, mc['ours'], 'A1')
            rec.d['meshes'][Ls] = M_
            rec.save()
            sbank = P.Bank(L, [0, 128])
            sst = P.Setting(sbank, 'sentinel', 128, 'fib4181', 'frozen', roles={}, nrec=NREC)
            sb = dict(bank=sbank, st=sst)
            ph = P.case_list({})[0][2]
            U0 = (jnp.asarray(P.Q.e.initial(L, ph)), float(ph[4]))
            M_['sentinel0'] = sentinel(sb, L, U0)
            M_['arms'] = {}
            for arm in mc['arms']:
                M_['arms'][arm['name']] = arm_block(L, arm, sb, U0)
                rec.save()
            del sb, sbank, sst
            gc.collect()
            M_['A2'] = ours_block(L, mc['ours'], 'A2')
            s0 = M_['sentinel0']
            sent_ok = all(in_band(a['sentinel_after'] / s0) for a in M_['arms'].values())
            ratios = {k: M_['A2'][k]['median'] / M_['A1'][k]['median'] for k in M_['A1'] if isinstance(M_['A1'][k], dict)
                      and 'median' in M_['A1'][k]}
            M_['drift'] = dict(sentinel_ok=sent_ok, a2_a1=ratios, a_ok=all(in_band(v) for v in ratios.values()))
            M_['valid'] = bool(sent_ok and M_['drift']['a_ok'])
            rec.d['meshes'][Ls] = M_
            rec.save()
            if M_['valid']:
                break
    rec.d['complete'] = True
    rec.save()
    log('MMTIME COMPLETE')


if __name__ == '__main__':
    main()
