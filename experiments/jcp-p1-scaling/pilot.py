"""jcp-p1-scaling 4096^2 memory pilot (DESIGN 9.2; code audits 1-2). One H200.

Upper-bound footprint: the timing job's complete resident state (every mesh's bank, every P1c subject's setting with
its successor and check roles, the split recording evolve compiled and run, certifier, tight run) PLUS the 4096^2
accuracy job's extras (all 38 initial fields on the device, the heaviest historical rule gauss640 at r = 384, Q3
Jacobian). bytes_in_use / peak_bytes_in_use are recorded after every step; warm per-run seconds (after a 0.1 s burn,
second call) feed the cost certificate. Decision (offline and here): peak > 120e9 bytes -> the 4096^2 arms use the
prefix-retaining bank; an unavailable peak (-1) is treated as exceeding the trigger.

    python pilot.py --config configs/pilot.json --out <dir>
"""
from __future__ import annotations

import argparse
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
    tcfg = json.loads((Path(a.config).parent / cfg['timing_config']).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    rep = dict(config=cfg, timing_config=tcfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, steps=[], runs=[], complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)

    def mark(name):
        rep['steps'].append(dict(step=name, t=el(), **P.mem()))
        save()
        print('MARK', name, el(), P.mem(), flush=True)

    mark('start')
    banks = {int(L): P.Bank(int(L), mc['edges']) for L, mc in cfg['meshes'].items()}
    mark('banks')
    for b in banks.values():
        b.ops('frozen')
    mark('ops')
    cases = P.case_list(cfg)
    ci = P.select(cases, [['dev6', 0]])[0]
    U1 = {}
    for L in banks:
        u0n = Q.e.initial(L, cases[ci][2])
        U1[L] = (jnp.asarray(u0n), float(cases[ci][2][4]))
    settings = {}

    def timed(fn):
        jax.block_until_ready(fn())
        QS.burn(.1)
        t0 = time.perf_counter()
        o = jax.block_until_ready(fn())
        return time.perf_counter() - t0, o

    for sp in [tcfg['A']] + tcfg['subjects']:            # the timing job's resident settings, all meshes
        L = int(sp['mesh'])
        k = (L, int(sp['Rp']), sp['rule'], sp.get('policy', 'frozen'))
        if k not in settings:
            settings[k] = P.Setting(banks[L], str(k), k[1], k[2], k[3], roles={r: sp[r] for r in ('qp', 'qpp') if sp.get(r)},
                                    nrec=NREC)
        st = settings[k]
        q = st.q['LSPG']
        d = st.data[('main', 'LSPG')]
        sch = st.sched('LSPG', sp['scheme'], DT0, 'prod')
        u0, nu = U1[L]
        w0, scale = jax.block_until_ready(q['init'](u0, st.cold))
        te, (W, s_) = timed(lambda: q['evolve'](w0, nu, scale, d, sch))
        cert = T2.make_certifier(q['nl'], st.Rp)
        tc, _ = timed(lambda: cert(s_['states'], nu, d, sch))
        tt, _ = timed(lambda: P.run_query(banks[L], st, 'main', 'LSPG', sp['scheme'], DT0, 'tight', u0, nu, record=False))
        rep['runs'].append(dict(subject=list(k) + [sp['scheme']], mesh=L, Rp=k[1], scheme=sp['scheme'], evolve_seconds=te,
                                cert_seconds=tc, tight_seconds=tt, valid=P.run_valid(dict(W=W, stats=s_))))
        mark(f'subject_{k}_{sp["scheme"]}')
    # 4096^2 accuracy-job extras on top: all 38 initial fields, gauss640 at r = 384, Q3 Jacobian
    L = int(cfg['pilot_mesh'])
    s = L // 256
    U0 = []
    for cj in P.select(cases, 'all38'):
        U0.append(jnp.asarray(Q.e.initial(L, cases[cj][2])))
    mark('U0_all38_on_device')
    k = (L, 384, 'gauss128', 'frozen')
    st = settings[k]
    st.add_role(banks[L], 'hist', 'gauss640')
    mark('gauss640_role')
    u0, nu = U1[L]
    th, v = timed(lambda: P.run_query(banks[L], st, 'hist', 'LSPG', 'BE', DT0, 'prod', u0, nu, record=False))
    jac = T2.make_jac(st.q['LSPG']['nl'])
    tq, J = timed(lambda: jac(v['W'][3], jnp.asarray(T2.SCHEMES['BE'][:5]), DT0, nu, st.data[('main', 'LSPG')]))
    np.linalg.svd(np.asarray(J), compute_uv=False)
    rep['runs'].append(dict(subject='4096|R384|gauss640|BE', seconds=th, q3_seconds=tq, valid=P.run_valid(v)))
    mark('extras_run')
    pk = P.mem()['peak']
    rep['peak_bytes'] = pk
    rep['trigger_bytes'] = 120e9
    rep['prefix_retaining_required'] = bool(pk < 0 or pk > 120e9)
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PILOT COMPLETE', el(), pk, flush=True)


if __name__ == '__main__':
    main()
