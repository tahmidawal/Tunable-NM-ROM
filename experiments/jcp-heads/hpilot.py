"""jcp-heads 4096^2 memory pilot (DESIGN 8, A1.9): build the mesh bank (all 512 rotated columns) and the 1024^2 bank of the
timing reference A, then the arms of every planned 4096^2 subject at an upper-bound rule (main rule `rule`, plus the
largest concurrently live successor / check tables on one arm), run one recorded rollout per head size K, one linear and
one A rollout, and report device memory (peak bytes in use) against the 120 GB decimal bar. Heads not yet trained are
stand-ins with the ACTUAL dimensions (code audit 1): for every planned (K, seed) a separate compact head archive with a
randomly initialised 512 x 2 MLP of latent size K (sep_hfit.init_head), 8192 random candidate codes and its own initial-
fit tables, so storage is independent per subject exactly as in the timing job; the incumbent is loaded as itself.

    python hpilot.py --config configs/<attempt>.json --out <dir>
config: {"mesh", "edges", "edges_A", "rule", "qp", "qpp", "heads": [{"name", "K" | "head"}], "lin": [r, ...], "A": {...}}
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
    import hcommon as HM
    import sep_hfit as hf
    syn = out / 'synthetic'
    syn.mkdir(exist_ok=True)
    inc_sha = HM.sha_file(HM.INCUMBENT)
    first_of_k = {}
    for i, h in enumerate(cfg['heads']):
        if 'head' in h:
            path = h['head']
        else:
            K = int(h['K'])
            hp = hf.init_head(jax.random.PRNGKey(1000 + i), K, 512, hidden=512, layers=2)
            Zs = np.asarray(jax.random.normal(jax.random.PRNGKey(2000 + i), (8192, K))) * .1
            zd = np.linalg.norm(Zs - Zs.mean(0), axis=1)
            hw = {f'h{j}_w': np.asarray(w) for j, (w, b) in enumerate(hp['h'])}
            hw.update({f'h{j}_b': np.asarray(b) for j, (w, b) in enumerate(hp['h'])})
            path = syn / f"{h['name']}.npz"
            np.savez(path, h_lin=np.asarray(hp['h_lin']), n_layers=len(hp['h']), Zsub=Zs, trust=.01 * float(zd.max()),
                     code_std=float(Zs.std()), code_max_dev=float(zd.max()), n_codes=len(Zs), k=K, seed0=-1, steps=0,
                     name=h['name'], source_ckpt_sha256=inc_sha, npz_sha256='synthetic', pkl_sha256='synthetic', **hw)
        a_ = HA.make_arm(bank, dict(name=h['name'], kind='head', head=str(path), rule=cfg['rule']), heads, S)
        arms.append(a_)
        first_of_k.setdefault(a_.K, a_)
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
    for arm in list(first_of_k.values()) + [arms[-1]]:
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
