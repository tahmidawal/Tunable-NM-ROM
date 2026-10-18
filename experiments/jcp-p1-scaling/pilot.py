"""jcp-p1-scaling 4096^2 memory pilot (DESIGN 9.2). One H200.

Builds, co-resident: the 256^2 and 1024^2 banks (the timing job's footprint) and the 4096^2 bank with the configured
edges; operators for the frozen 2048-mode list at 4096^2 (and nominal); the dev6 + val32 initial fields at 4096^2 on the
device (the accuracy job's footprint); per r in the config one Setting with its rule and successor roles (and the
heaviest historical rule, gauss640 at r = 384); one prod BE, one prod CN-R (recording) and one tight run on dev6|0.
Records bytes_in_use / peak_bytes_in_use after every step and per-run wall times (input to the cost certificate).
Decision (offline): peak > 120e9 bytes -> the 4096^2 arms use the prefix-retaining bank (DESIGN 2.2, 9.2).

    python pilot.py --config configs/pilot.json --out <dir>
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
import jax
import jax.numpy as jnp

NREC = int(round(.25 / DT0)) + 1


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
               jax_version=jax.__version__, steps=[], runs=[], complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    mark = lambda name: (rep['steps'].append(dict(step=name, t=el(), **P.mem())), save(), print('MARK', name, el(), P.mem(), flush=True))
    mark('start')
    banks = {}
    for Ls, mc in cfg['meshes'].items():
        banks[int(Ls)] = P.Bank(int(Ls), mc['edges'])
        mark(f'bank_{Ls}')
    L = int(cfg['pilot_mesh'])
    bank = banks[L]
    for pol in ('frozen', 'nominal'):
        bank.ops(pol)
        mark(f'ops_{pol}_{L}')
    cases = P.case_list(cfg)
    s = L // 256
    U0 = {}
    for ci in P.select(cases, 'all38'):
        u0n = Q.e.initial(L, cases[ci][2])
        U0[ci] = (jnp.asarray(u0n), float(cases[ci][2][4]), float(np.linalg.norm(u0n[::s, ::s])))
    mark('U0_all38_on_device')
    ci = P.select(cases, [['dev6', 0]])[0]
    kept = []
    u0, nu, n0 = U0[ci]
    for sp in cfg['settings']:
        st = P.Setting(bank, f"R{sp['Rp']}", sp['Rp'], sp['rule'], sp.get('policy', 'frozen'), roles=sp.get('roles', {}),
                       nrec=NREC)
        mark(f"setting_R{sp['Rp']}_{sp['rule']}")
        cert = T2.make_certifier(st.q['LSPG']['nl'], st.Rp)
        jac = T2.make_jac(st.q['LSPG']['nl'])
        for sc, lev, rec in (('BE', 'prod', True), ('CNR', 'prod', True), ('BE', 'tight', False), ('CNR', 'tight', False),
                             ('CNR', 'tighter', False)):
            for k in range(2):           # first call includes compilation
                t0 = time.perf_counter()
                v = jax.block_until_ready(P.run_query(bank, st, 'main', 'LSPG', sc, DT0, lev, u0, nu, record=rec))
                rep['runs'].append(dict(Rp=sp['Rp'], rule=sp['rule'], scheme=sc, level=lev, call=k,
                                        seconds=time.perf_counter() - t0, valid=P.run_valid(v),
                                        it_sum=int(v['stats']['it_sum'])))
                if rec:
                    t0 = time.perf_counter()
                    jax.block_until_ready(cert(v['stats']['states'], nu, st.data[('main', 'LSPG')], st.sched('LSPG', sc, DT0, 'prod')))
                    rep['runs'][-1]['cert_seconds'] = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    np.linalg.svd(np.asarray(jac(v['W'][3], jnp.asarray(T2.SCHEMES[sc][:5]), DT0, nu, st.data[('main', 'LSPG')])),
                                  compute_uv=False)
                    rep['runs'][-1]['q3_seconds'] = time.perf_counter() - t0
                del v
            mark(f"runs_R{sp['Rp']}_{sc}_{lev}")
        for role in sp.get('roles', {}):
            t0 = time.perf_counter()
            v = jax.block_until_ready(P.run_query(bank, st, role, 'LSPG', 'BE', DT0, 'prod', u0, nu, record=False))
            rep['runs'].append(dict(Rp=sp['Rp'], rule=st.rules[role], scheme='BE', level='prod', call=0,
                                    seconds=time.perf_counter() - t0, valid=P.run_valid(v)))
            del v
            mark(f"role_R{sp['Rp']}_{role}")
        kept.append(st)                  # settings stay resident, as in the timing job
        mark(f"kept_R{sp['Rp']}")
    rep['peak_bytes'] = P.mem()['peak']
    rep['trigger_bytes'] = 120e9
    rep['prefix_retaining_required'] = bool(rep['peak_bytes'] > 120e9)
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PILOT COMPLETE', el(), rep['peak_bytes'], flush=True)


if __name__ == '__main__':
    main()
