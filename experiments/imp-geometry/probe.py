"""imp-geometry production-size cost probe (DESIGN A1.10): measured seconds of one unit of every stage at production
sizes on the cluster GPU, for the cost certificate (resources.py). Writes <out>/probe.json.
  python probe.py --out <dir> --dom L --lev S2"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geom as G          # noqa: E402
import fd                 # noqa: E402
import sem                # noqa: E402
import rom as RM          # noqa: E402
import pod as PD          # noqa: E402
import ladder as LD       # noqa: E402
import run_cell as RC     # noqa: E402

R = {}


def tm(name, f, *a, **k):
    t0 = time.perf_counter()
    out = f(*a, **k)
    jax.block_until_ready(out) if not isinstance(out, (dict, list, tuple, type(None), float, int)) else None
    R[name] = time.perf_counter() - t0
    print(f'{name}: {R[name]:.2f}s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--dom', default='L')
    ap.add_argument('--lev', default='S2')
    ap.add_argument('--sections', default='ref,pod,fd,sem,bank')
    a = ap.parse_args()
    secs = a.sections.split(',')
    dom, lev = a.dom, a.lev
    D = G.draw('wide', lev, 'val')
    D8 = D[:8]
    n, nt = RC.DATA_GRID[lev]
    gr = fd.Grid(dom, n)
    m = gr.mask_np
    if 'ref' in secs:
        ref_data(dom, lev, D8, n, nt, m)
    if 'pod' in secs:
        pod_sec(dom, lev, D, n, gr)
    if 'fd' in secs:
        fd_sec(dom, D)
    if 'sem' in secs and dom != 'B':
        sem_sec(dom, D)
    if 'bank' in secs:
        bank_sec(dom, lev, D, n, gr, m)
    R['env'] = RC.env_record()
    R['gpu_mem_peak_bytes'] = max((d.memory_stats() or {}).get('peak_bytes_in_use', 0) for d in jax.devices())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'probe_{dom}_{lev}_{"-".join(secs)}.json').write_text(json.dumps(R, indent=1, default=float))


def ref_data(dom, lev, D8, n, nt, m):
    # references: each level, 8 cases batched (compile included in first, measured separately by a second call)
    for (n, nt) in RC.REF_LEVELS[lev]:
        q = fd.make_sbdf2(dom, n, nt, 1e-10, maxit=60)
        tm(f'ref_n{n}_compile+8', lambda: LD.run_cohort(q, D8)[1])
        tm(f'ref_n{n}_8', lambda: LD.run_cohort(q, D8)[1])
    # training data: 8 trajectories on the data grid with 21 outputs
    qd = fd.make_sbdf2(dom, n, nt, 1e-10, n_out=20, out_idx=np.nonzero(m), maxit=60)
    tm('data_8traj_compile', lambda: jax.vmap(qd)(jnp.asarray(D8))[0])
    tm('data_8traj', lambda: jax.vmap(qd)(jnp.asarray(D8))[0])


def pod_sec(dom, lev, D, n, gr):
    n_in = int(gr.mask_np.sum())
    R['n_in_data'] = n_in
    # POD at production size: random snapshot set of 4032 x n_in (memory + time)
    S = np.random.default_rng(0).standard_normal((4032, n_in))
    Phi, sv = tm('pod_4032', PD.pod, S, gr.h, 512)
    sp = tm('pod_space', PD.pod_space, dom, n, Phi, 128)
    del S
    # reduced queries at r = 384 (tensor, LSPG) on 32 cases
    for st, form in (('lspg_cnr', 'tensor'), ('gal_etdrk4', 'tensor'), ('gal_cnr', 'tensor')):
        q = RM.make_query(sp, 384, st, 80, form=form)
        tm(f'podg_r384_{st}_compile+32', lambda: LD.run_cohort(q, D)[1])
        tm(f'podg_r384_{st}_32', lambda: LD.run_cohort(q, D)[1])
    # ECSW fit at r = 384, Mt = 512 (LSPG) and Mt = 384
    Z = np.random.default_rng(1).standard_normal((132, 512)) * sv[:512][None] / sv[0]
    for Mt in (384, 512):
        fit, info = tm(f'ecsw_fit_r384_Mt{Mt}', PD.ecsw_fit, sp, 384, Mt, Z, max_seconds=1200)
        R[f'ecsw_info_Mt{Mt}'] = info


def fd_sec(dom, D):
    # FD-MG and RK4 at the largest ladder size
    for nn, ntt in ((768, 1280), (384, 640)):
        q = fd.make_sbdf2(dom, nn, ntt, 1e-8)
        tm(f'fdmg_n{nn}_nt{ntt}_compile+32', lambda: LD.run_cohort(q, D)[1])
        tm(f'fdmg_n{nn}_nt{ntt}_32', lambda: LD.run_cohort(q, D)[1])


def sem_sec(dom, D):
    # SEM: largest and a middle configuration, fine steps, 32 cases; eig factorisation
    if True:
        for p, k in ((12, 8), (8, 4)):
            q, Ssp = sem.make_sem(dom, p, k, 2560, 1e-10, order=3, grade=0.5 if dom == 'L' else None)
            R[f'sem_p{p}_k{k}_Ng'] = int(Ssp.Ng)
            tm(f'sem_p{p}_k{k}_nt2560_compile+32', lambda: LD.run_cohort(q, D)[1])
            st = tm(f'sem_p{p}_k{k}_nt2560_32', lambda: LD.run_cohort(q, D)[1])
            o, s, _ = LD.run_cohort(q, D[:2])
            R[f'sem_p{p}_k{k}_pcg_its_per_step'] = float(np.max(s[:, 0]) / 2560)
        S2 = sem.SEMSpace(dom, 10, 6, 0.5 if dom == 'L' else None)
        R['sem_p10_k6_interior'] = int(S2.dm_np.sum())
        tm('sem_eig_factor_p10_k6', S2.eig_factor)


def bank_sec(dom, lev, D, n, gr, m):
    # bank: training steps at production size (sub and full), bank space and LSPG r = 384
    import bank as BK
    nb = RC.BANK_SAMPLE[lev]
    s_ = n // nb
    sub = np.zeros_like(m)
    sub[s_ - 1::s_, s_ - 1::s_] = True
    sub &= m
    Xb = np.stack([gr.X[sub], gr.Y[sub]], -1)
    Ub = np.random.default_rng(2).standard_normal((4032, len(Xb))) * 0.1
    R['bank_points'] = int(len(Xb))
    BK.RECIPE['full_last'] = 20
    tm('bank_train_220steps', BK.train, dom, Xb, Ub, 0, steps=220, log_every=100000)
    params, Zc, info = BK.train(dom, Xb, Ub, 0, steps=30, log_every=100000)
    Xq, wq = G.composite_rule(dom, RC.bank_rule_k(dom), 16)
    T, *_ = tm('bank_rotation', BK.rotation, dom, params, Zc, Xq, wq)
    bs = tm('bank_space_q12', BK.bank_space, dom, params, T, RC.bank_rule_k(dom), 12)
    q = RM.make_query(bs, min(384, bs.R), 'lspg_cnr', 80, form='point')
    tm('bank_r384_lspg_compile+32', lambda: LD.run_cohort(q, D)[1])
    tm('bank_r384_lspg_32', lambda: LD.run_cohort(q, D)[1])


if __name__ == '__main__':
    main()
