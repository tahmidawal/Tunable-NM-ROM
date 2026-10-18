"""jcp-heads 4096^2 memory pilot (DESIGN 8, A1.9): build the mesh bank (all 512 rotated columns) and the 1024^2 bank of the
timing reference A, then the arms of every planned 4096^2 subject at an upper-bound rule (main rule `rule`, plus the
largest concurrently live successor / check tables on one arm), run one head and one linear rollout, and report device
memory (peak bytes in use) against the 120 GB decimal bar. Heads not yet trained are stood in for by the incumbent (an
arm's tables do not depend on the head's weights, only on K, M and the rule).

    python hpilot.py --config configs/<attempt>.json --out <dir>
config: {"mesh", "edges", "edges_A", "rule", "qp", "qpp", "heads": [{"name", "head", "M"?}], "lin": [r, ...], "A": {...}}
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

BAR = 120e9


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
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(), stages=[], complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    stage = lambda name: (rep['stages'].append(dict(stage=name, seconds=time.perf_counter() - t0, mem=P.mem())), save(),
                          print('STAGE', name, P.mem(), flush=True))
    L = int(cfg['mesh'])
    bank = P.Bank(L, cfg['edges'])
    stage('bank_mesh')
    bankA = P.Bank(1024, cfg['edges_A'])
    stage('bank_1024_for_A')
    S = HA.steps_of(DT0)
    heads, arms = {}, []
    A = HA.make_arm(bankA, dict(cfg['A']), heads, S)
    stage('A')
    for h in cfg['heads']:
        arms.append(HA.make_arm(bank, dict(name=h['name'], kind='head', head=h['head'], rule=cfg['rule'], M=h.get('M')), heads, S))
    stage('heads')
    for r in cfg['lin']:
        arms.append(HA.make_arm(bank, dict(name=f'L{r}', kind='lin', Rp=int(r), rule=cfg['rule']), heads, S))
    stage('linear')
    big = max(arms, key=lambda x: x.M)
    big.add_role('qp', cfg['qp'])
    big.add_role('qpp', cfg['qpp'])
    stage('successor_check_tables_on_largest_M')
    cases = P.case_list(cfg)
    ci = P.select(cases, 'dev6')[0]
    u0 = jax.numpy.asarray(Q.e.initial(L, cases[ci][2]))
    nu = float(cases[ci][2][4])
    for arm in (arms[0], arms[-1]):
        v = arm.run('main', 'BE', 'prod', u0, nu, DT0, record=True)
        jax.block_until_ready(v)
        rep[f'rollout_{arm.name}'] = dict(valid=HA.run_valid(v), it_sum=int(v['stats']['it_sum']))
        stage(f'rollout_{arm.name}')
        del v
    va = A.run('main', 'CNR', 'prod', jax.numpy.asarray(Q.e.initial(1024, cases[ci][2])), nu, DT0, record=True)
    jax.block_until_ready(va)
    stage('rollout_A')
    peak = max(s_['mem']['peak'] for s_ in rep['stages'])
    rep.update(peak_bytes=peak, bar_bytes=BAR, within_bar=bool(0 < peak <= BAR), complete=True,
               elapsed_seconds=time.perf_counter() - t0)
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HPILOT COMPLETE peak', peak, flush=True)


if __name__ == '__main__':
    main()
