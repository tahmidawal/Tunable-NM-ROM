"""imp-fronts E10-q follow-up (audit codex-e10-5): successor-rule rho at EVERY accepted state of every POL-K1 prod run.

fpolicy / r2acc record rho_succ only at the six output states (W). This script recomputes rho with exactly their code
path -- the same Bank (K1, mesh 1024, edges), Setting(Rp = 384, 'frozen' test modes, M = 4 r), rule tables through
Setting.add_role (roles 'd' = the run's rule, 'ds' = policy_sm.Q2_SUCCESSOR[rule]), the same query's nl, and
p1common.rho -- at every row of each cert run's recorded `states` (row k = accepted state after step k; row 0 = the
initial fit). Validation first: at the six output states (row k * keep) it must reproduce the recorded rho_succ of the
run's successor record to ~1e-12 (relative).

    JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest bash <jaxslot.sh> 3600 <venv python> rom/rho_every_step.py \
        [<POL-K1 fpolicy output dir>] [--out checks/phaseb/e10q_every_step.json]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
ROOT = HERE.parents[2]
for _p in (ROOT / 'experiments' / 'separable-decoder', ROOT / 'experiments' / 'mr-burgers2d'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'deps'))

import numpy as np  # noqa: E402
import policy_sm as SM  # noqa: E402
import bankguard as BG  # noqa: E402

BAR_RHO = .116
VAL_TOL = 1e-6                    # registered (E10-DESIGN.md section 10): recorded values are GPU, the recompute is CPU
DEFAULT = LANE / 'runs' / 'rom' / 'POL-K1-out' / 'output' / 'fpolicy_pol1'


def main():
    args = [a for a in sys.argv[1:]]
    out = Path(args[args.index('--out') + 1]) if '--out' in args else LANE / 'checks' / 'phaseb' / 'e10q_every_step.json'
    pos = [a for i, a in enumerate(args) if a != '--out' and (i == 0 or args[i - 1] != '--out')]
    src = Path(pos[0]) if pos else DEFAULT
    res = json.loads((src / 'result.json').read_text())
    cfg, recs = res['config'], res['records']
    import archrun
    archrun.pin()
    import jax
    import jax.numpy as jnp
    import p1common as P
    t0 = time.perf_counter()
    bank_cfg = BG.verify(cfg['bank'], HERE, expect_label=cfg.get('bank_label'))
    assert bank_cfg['ckpt_sha256'] == res['bank']['ckpt_sha256'], 'bank differs from the job\'s'
    bank = P.Bank(int(cfg['mesh']), cfg['edges'], ckpt=HERE / bank_cfg['ckpt'], rotation=HERE / bank_cfg['rotation'])
    st = P.Setting(bank, 'POL', int(cfg.get('Rp', 384)), 'fib4181', 'frozen', roles={}, forms=('LSPG',), nrec=0)
    assert st.mode_hash == res['setup']['mode_hash'], 'test-mode list differs from the job\'s'
    nl = st.q['LSPG']['nl']
    pair = jax.jit(jax.vmap(lambda c, dd, ds: (nl(c, dd), nl(c, ds)), in_axes=(0, None, None)))
    cert = sorted(k for k, v in recs.items() if '|cert|' in k and v.get('valid') is not None and not v.get('skipped'))
    by_rule = {}
    for k in cert:
        by_rule.setdefault(SM.parse_key(k)[2], []).append(k)
    rows, val_err = [], []
    for rule, keys in sorted(by_rule.items()):
        succ = SM.Q2_SUCCESSOR[rule]
        st.add_role(bank, 'd', rule, forms=('LSPG',))
        st.add_role(bank, 'ds', succ, forms=('LSPG',))
        dd, ds = st.data[('d', 'LSPG')], st.data[('ds', 'LSPG')]
        for k in keys:
            cid, _, _, f = SM.parse_key(k)
            fn = src / 'runs' / f"{k.replace('|', '__').replace('.', 'p')}.npz"
            z = np.load(fn)
            S = np.asarray(z['states'])
            ok = np.isfinite(S).all(axis=1)
            Nd, Ns = pair(jnp.asarray(np.where(ok[:, None], S, 0.)), dd, ds)
            Nd, Ns = np.asarray(Nd), np.asarray(Ns)
            rho = np.array([P.rho(Nd[i], Ns[i]) if ok[i] else float('inf') for i in range(len(S))])
            keep = int(round(.05 / (P.DT0 * f)))
            sk = SM.run_key(cid, 'succ', succ, f)
            if sk in recs and recs[sk].get('rho_succ') is not None:      # validation at the six output states
                want = np.array([np.nan if x is None else x for x in recs[sk]['rho_succ']], float)
                got = rho[[i * keep for i in range(6)]]
                err = float(np.nanmax(np.abs(got - want) / np.maximum(np.abs(want), 1e-300)))
                val_err.append(dict(run=k, succ=sk, max_rel_err=err))
            fin = rho[np.isfinite(rho)]
            rows.append(dict(run=k, case=cid, cohort=cid.split('|')[0], rule=rule, succ=succ, factor=f, states=int(len(S)),
                             finite_states=int(len(fin)), rho_max=float(fin.max()) if len(fin) else None,
                             rho_argmax=int(np.argmax(np.where(np.isfinite(rho), rho, -1.))),
                             n_over=int(np.sum(~(rho <= BAR_RHO))), validated=sk in recs))
        st.drop_role('d')
        st.drop_role('ds')
        for kk in ('d', 'ds'):
            st.meta.pop(kk, None)
            st.rules.pop(kk, None)
    errs = [v['max_rel_err'] for v in val_err]
    fc1 = [r for r in rows if r['cohort'] == 'FC1' and r['rho_max'] is not None]
    allr = [r for r in rows if r['rho_max'] is not None]
    rep = dict(source=str(src), result_sha256=hashlib.sha256((src / 'result.json').read_bytes()).hexdigest(),
               bank=bank_cfg, backend=jax.default_backend(), precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
               bar_rho=BAR_RHO, n_runs=len(rows), validation=dict(n=len(val_err), max_rel_err=max(errs) if errs else None,
                                                                  tolerance=VAL_TOL, passed=bool(errs) and max(errs) <= VAL_TOL,
                                                                  per_run=val_err),
               summary=dict(rho_max_all=max(r['rho_max'] for r in allr), rho_max_FC1=max(r['rho_max'] for r in fc1) if fc1 else None,
                            states_over_bar=sum(r['n_over'] for r in rows), runs_over_bar=sum(r['n_over'] > 0 for r in rows),
                            states_total=sum(r['states'] for r in rows),
                            accepted_step_states=sum(r['states'] for r in rows) - len(rows),
                            initial_fit_states=len(rows)),
               runs=rows, seconds=time.perf_counter() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(dict(rep['summary'], validation_max_rel_err=rep['validation']['max_rel_err'], n_validated=len(val_err))))
    if not rep['validation']['passed']:
        sys.exit(f'validation failed: max relative error {rep["validation"]["max_rel_err"]} > {VAL_TOL}')


if __name__ == '__main__':
    main()
