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
    coh, c, ph = cases[0]
    u0 = jnp.asarray(Q.e.initial(L, ph))
    nu = float(ph[4])
    keep = {}
    st2.add_role(bank2, 'qp', 'gauss160', forms=('LSPG', 'GAL'))          # A's own Q2 tables, as qtime certifies them
    st2.add_role(bank2, 'qpp', 'gauss192', forms=('LSPG', 'GAL'))
    step('A Q2 tables')
    for kk in ('qp', 'qpp'):
        st2.drop_role(kk)
    gc.collect()
    for s in cfg['settings']:
        r = int(s['Rp'])
        keep[r] = P.Setting(bank, f'R{r}', r, s['rules'][0], 'frozen', forms=('LSPG', 'GAL'))
        step(f'R{r} main {s["rules"][0]}')
        # successor tables resident together, as qtime / qacc hold them: (q+, q++); and q+, q++, q+++ (fallback setting)
        pairs = [s['rules'][1:3]] + ([s['rules'][1:4]] if len(s['rules']) > 3 else [])
        for pr in pairs:
            for k, rl in enumerate(pr):
                keep[r].add_role(bank, f'x{k}', rl, forms=('LSPG', 'GAL'))
            step(f'R{r} +{"+".join(pr)}')
            if r == 512:
                for k, rl in enumerate(pr):
                    for form, sc, f in (('LSPG', 'BE', 1), ('LSPG', 'CNR', 2.5), ('GAL', 'CN', 5)):
                        t1 = time.perf_counter()
                        v = QA.run(bank, keep[r], f'x{k}', form, sc, DT0 * f, 'prod', u0, nu, record=True)
                        jax.block_until_ready(v)
                        rep['runs'].append(dict(rule=rl, form=form, scheme=sc, factor=f, seconds=time.perf_counter() - t1,
                                                valid=P.run_valid(v)))
                        del v
                step(f'R512 rollouts {"+".join(pr)}')
            for k in range(len(pr)):
                keep[r].drop_role(f'x{k}')
            gc.collect()
        for form, sc in (('LSPG', 'CNR'), ('GAL', 'CN')):
            v = QA.run(bank, keep[r], 'main', form, sc, DT0, 'prod', u0, nu, record=True)
            jax.block_until_ready(v)
            rep['runs'].append(dict(Rp=r, rule=s['rules'][0], form=form, scheme=sc, valid=P.run_valid(v)))
            del v
        step(f'R{r} done')
        save()
    rep['peak'] = max(x['peak'] for x in rep['steps'])
    rep['feasible'] = bool(0 < rep['peak'] <= 120e9)
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('QPILOT COMPLETE peak', rep['peak'], flush=True)


if __name__ == '__main__':
    main()
