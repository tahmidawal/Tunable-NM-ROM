"""jcp-qualify 4096^2 memory pilot (DESIGN 8, A1.6, A2.4): the full time4096 / acc4096 footprint on one H200.

Builds the 4096^2 full R512 bank and the 1024^2 bank (reference A), then per r (128 ... 512) the main-rule setting kept
resident (timing needs all four), and one at a time each further rule of the r (q+, q++ and the R512 fallback q+++),
dropped after use; one recorded prod rollout per (form, scheme) at R512 with each R512 rule. Records bytes_in_use /
peak_bytes_in_use after every step. Verdict (offline): peak <= 120e9 bytes -> feasible.

    python qpilot.py --config configs/pilot.json --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, DT0
import qacc as QA
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
    t0 = time.perf_counter()
    rep = dict(config=cfg, gpu=jax.devices()[0].device_kind, steps=[], runs=[], complete=False)
    step = lambda s: (rep['steps'].append(dict(step=s, t=round(time.perf_counter() - t0, 1), **P.mem())), print(s, P.mem(), flush=True))
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    step('start')
    L = int(cfg['mesh'])
    bank = P.Bank(L, cfg['edges'])
    step('bank4096')
    bank2 = P.Bank(int(cfg['also_bank']), cfg['edges'])
    st2 = P.Setting(bank2, 'A', 384, 'gauss128', 'frozen', forms=('LSPG', 'GAL'))
    step('bank1024+A')
    cases = P.case_list(cfg)
    dev = P.select(cases, 'dev6')                 # qtime keeps every dev6 initial field resident on both meshes
    U0 = {(LL, ci): jnp.asarray(Q.e.initial(LL, cases[ci][2])) for LL in (L, int(cfg['also_bank'])) for ci in dev}
    u0 = U0[(L, dev[0])]
    nu = float(cases[dev[0]][2][4])
    step('dev6 initial fields (both meshes)')
    keep = {}
    st2.add_role(bank2, 'qp', 'gauss160', forms=('LSPG', 'GAL'))          # A's own Q2 tables, as qtime certifies them
    st2.add_role(bank2, 'qpp', 'gauss192', forms=('LSPG', 'GAL'))
    step('A Q2 tables')
    for kk in ('qp', 'qpp'):
        st2.drop_role(kk)
    gc.collect()
    plan = [(f'R{int(s_["Rp"])}', int(s_['Rp']), s_['rules'][0], s_['rules'][1:3]) for s_ in cfg['settings']]
    plan += [(f'R{int(s_["Rp"])}fb', int(s_['Rp']), s_['rules'][1], s_['rules'][2:4]) for s_ in cfg['settings'] if len(s_['rules']) > 3]
    for name, r, main, pair in plan:              # every original AND fallback main stays resident (timing retains them);
        keep[name] = P.Setting(bank, name, r, main, 'frozen', forms=('LSPG', 'GAL'))     # one setting's successors at a time
        step(f'{name} main {main}')
        for k, rl in enumerate(pair):
            keep[name].add_role(bank, f'x{k}', rl, forms=('LSPG', 'GAL'))
        step(f'{name} +{"+".join(pair)}')
        if r == 512:
            for role, rl in [('main', main)] + [(f'x{k}', rl) for k, rl in enumerate(pair)]:
                for form, sc, f in (('LSPG', 'BE', 1), ('LSPG', 'CNR', 2.5), ('GAL', 'CN', 5)):
                    t1 = time.perf_counter()
                    v = QA.run(bank, keep[name], role, form, sc, DT0 * f, 'prod', u0, nu, record=True)
                    jax.block_until_ready(v)
                    # qtime's live pattern: split query (init / recording evolve / full-field decode) and the certifier,
                    # their outputs held together
                    st_ = keep[name]
                    S = QA.steps_of(DT0 * f)
                    q = QA.QQ.get(st_, form, S + 1, nstep=max(QA.T2.NSTEP, S + 1))
                    g_, t_ = P.tol_of(form, 'prod')
                    sch = QA.T2.sched(sc, DT0 * f, g_, t_)
                    d_ = st_.data[(role, form)]
                    w0, scale = jax.block_until_ready(q['init'](u0, st_.cold))
                    W, s_ = jax.block_until_ready(q['evolve'](w0, nu, scale, d_, sch))
                    F = jax.block_until_ready(q['decode'](W, d_))
                    cert = (QA.T2.make_certifier(q['nl'], r) if form == 'LSPG' else QA.QC.make_gal_certifier(q['nl'], r))
                    o = jax.block_until_ready(cert(s_['states'], nu, d_, sch))
                    # qtime keeps the tight, tighter, successor and check replays' outputs referenced together with F
                    held = [jax.block_until_ready(QA.run(bank, st_, rr, form, sc, DT0 * f, lev, u0, nu))
                            for rr, lev in ((role, 'tight'), (role, 'tighter'), (role, 'prod'), (role, 'prod'))]
                    rep['runs'].append(dict(setting=name, rule=rl, form=form, scheme=sc, factor=f,
                                            seconds=time.perf_counter() - t1, valid=P.run_valid(v)))
                    step(f'{name} {rl} {form} {sc} split+certifier live')
                    del v, W, s_, F, o, held
            step(f'{name} rollouts')
        for k in range(len(pair)):
            keep[name].drop_role(f'x{k}')
        gc.collect()
        step(f'{name} done')
        save()
    rep['peak'] = max(x['peak'] for x in rep['steps'])
    rep['feasible'] = bool(0 < rep['peak'] <= 120e9)
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('QPILOT COMPLETE peak', rep['peak'], flush=True)


if __name__ == '__main__':
    main()
