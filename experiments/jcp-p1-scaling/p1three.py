"""jcp-p1-scaling 3D job (DESIGN 2.4, 2.5 P1b and P1e-3D; A1.10, A1.12). One GPU, mesh 65^3, W1024 bank.

The 3D setup is taken from jcp-combined `c3common.py` (bank / rule loading and lean tables, copied below with the
sha256 pins; provenance in PROVENANCE.json), using the vendored `offmesh` unchanged.

  mode 'kern'  P1b: kernel components N, J, JTJ, chol (verdict, fixed panel) and res, evalJ, g (descriptive) on the
               W1024 BE fixtures (W_grid.npz 'W1024|lat65536|fs|BE|0.01|K1', cases 0-5 of 923801, outputs 1 and 5)
               truncated to each r; panels fixed (M = complete_M(65, 2048) frozen shell list, Gauss 24^3) and nominal
               (M = complete_M(65, 4 r), the W1024-prefix rule per width); dispatch-sync control at the largest r.
  mode 'p1e'   P1e-3D: W512 (W1024 prefix 512), deployed fixed-sweep BE (vendor make_fsc_rule, a = 6, dt 0.01), rules
               lat8192 (failing) -> lat32768 (deployed) -> gl56 (finer) -> gl64 (check), cases 0-5 of 923801;
               distances on the 63^3 interior lattice / ||u0||, own-state rho vs the next rule. Descriptive (A1.12).

    python p1three.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
sys.path.insert(0, str(HERE / 'vendor' / 'quad2d'))
import offmesh as OM  # noqa: E402  (vendored, unchanged)
from offmesh import C, TB  # noqa: E402
import p1core  # noqa: E402,F401  (sets the import paths of the 2D vendored code that qstudy needs)
import qstudy as QS  # noqa: E402
import p1comp as PC  # noqa: E402

DT3 = .01
BANK = (HERE / 'inputs' / 'model_W1024' / 'bank.pkl', '86094f6884e55643240197e560f7dc1924bac3885aed79b20faf2e54446aa036')
RULE_FILES = [(HERE / 'vendor' / 'quad3d' / 'rules' / 'rules.npz', '099a79976d03e993dc3fbdcc153a57dae0a2dbc5f917d7ca3674bef28daf859b'),
              (HERE / 'rules3d' / 'rules_extra.npz', '8deacf019839b8cf42993d5d8926126541db8152b811f411eceeb69940bb38cf')]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_bank():
    p, want = BANK
    assert sha_file(p) == want
    b = pickle.loads(p.read_bytes())
    return dict(params=jax.tree_util.tree_map(jnp.asarray, b['params']), T=np.asarray(b['rotation']),
                spread=b['coefficient_rms_spread'], sha256=want)


def load_rules(names):
    out = {}
    for p, want in RULE_FILES:
        assert sha_file(p) == want, p
        z = np.load(p)
        for nm in names:
            if nm + '_X' in z.files:
                X, w = np.asarray(z[nm + '_X']), np.asarray(z[nm + '_w'])
                assert nm not in out or (np.array_equal(out[nm][0], X) and np.array_equal(out[nm][1], w)), nm
                out[nm] = (X, w)
    missing = set(names) - set(out)
    assert not missing, missing
    return out


def rho_a24(a, b):
    """A2.4 semantics (same as p1common.rho): both zero -> nan (trivial); b = 0 != a -> inf; non-finite -> inf."""
    a, b = np.asarray(a), np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('inf')
    nb = float(np.linalg.norm(b))
    if nb == 0.:
        return float('nan') if float(np.linalg.norm(a)) == 0. else float('inf')
    return float(np.linalg.norm(a - b) / nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg['_out'] = str(out)
    import os
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu'
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    n = 65
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, jax_version=jax.__version__,
               offmesh_sha256=sha_file(OM.__file__), complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    bank = load_bank()
    order, lam_all = C.mode_order(n)
    ranks = [int(r) for r in cfg['ranks']]
    M_of = {r: int(C.complete_M(n, 4 * r, order, lam_all)) for r in ranks}
    Mfix = int(C.complete_M(n, 4 * int(cfg.get('fixed_r', 512)), order, lam_all))
    Mmax = max(list(M_of.values()) + [Mfix])
    tb = TB.build_tables(n, bank['params'], bank['T'], sorted(set(ranks)), lambda r: Mmax, log=lambda s: None, tensor=False)
    kx = tb['kxyz']
    rep['M_of'], rep['M_fixed'], rep['M_max'] = {str(k): v for k, v in M_of.items()}, Mfix, Mmax
    rep['setup_seconds'] = el()
    print('SETUP', el(), flush=True)
    if cfg['mode'] == 'kern':
        kern(cfg, rep, save, bank, tb, kx, M_of, Mfix, ranks, n, el)
    else:
        p1e(cfg, rep, save, bank, tb, kx, M_of, n, el)
    try:
        rep['peak_bytes'] = int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1))
    except Exception:  # noqa: BLE001
        rep['peak_bytes'] = -1
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P1THREE COMPLETE', el(), flush=True)


def kern(cfg, rep, save, bank, tb, kx, M_of, Mfix, ranks, n, el):
    fx = cfg['fixtures']
    assert sha_file(fx['file']) == fx['sha256'], 'fixture file hash mismatch'
    z = np.load(fx['file'])
    fixtures, labels = np.asarray(z['W']), [str(x) for x in z['labels']]
    rep['fixtures'] = dict(labels=labels, case_of=[l_.rsplit('|', 1)[0] for l_ in labels], R=int(fixtures.shape[1]))
    rules = load_rules(sorted({cfg['fixed_rule']} | set(cfg['nominal_rules'].values())))
    Rmax = max(ranks)
    nu, dt = float(cfg['nu']), DT3
    blk = {}

    def rule_blk(rule, M):
        if (rule, M) not in blk:
            blk.clear()
            X, w = rules[rule]
            om = OM.offmesh_tables(n, bank['params'], bank['T'][:, :Rmax], X, w, kx[:M])
            blk[(rule, M)] = om
        return blk[(rule, M)]

    def make_tabs(panel):
        def tabs(r):
            M, rule = (Mfix, cfg['fixed_rule']) if panel == 'fixed' else (M_of[r], cfg['nominal_rules'][str(r)])
            om = rule_blk(rule, M)
            return (jnp.asarray(tb['A'][:M, :r]), jnp.asarray(tb['lam'][:M]), om['B'][:, :r], om['D'][:, :r], om['P'])
        return tabs

    def meta(panel, r):
        M, rule = (Mfix, cfg['fixed_rule']) if panel == 'fixed' else (M_of[r], cfg['nominal_rules'][str(r)])
        return M, len(rules[rule][1]), rule

    def ref_kernel():
        Ar, lr, B, D, P = make_tabs('fixed')(Rmax)
        c = jnp.asarray(fixtures[0][:Rmax])
        Jf = Ar + (dt / (1 + dt * nu * lr))[:, None] * PC.k_J(c, B, D, P)
        Hm = PC.k_JTJ(Jf)
        g = Jf.T @ PC.k_res(c, Ar, lr, B, D, P, nu, dt)
        return float(np.median(PC._time(PC.k_chol, (Hm, g, jnp.asarray(1e-6)), 30, cfg['burn'])))

    rep['panels'], rep['drift'], rep['sync'] = {}, {}, {}
    QS.burn(.05)
    for panel in cfg['panels']:
        for attempt in range(2):
            d0 = ref_kernel()
            rows = PC.kernel_panel(None, None, make_tabs(panel), fixtures, ranks, nu, dt, reps=int(cfg['reps']),
                                   burn=float(cfg['burn']), seed=int(cfg['seed']) + attempt)
            for r_ in rows:
                M, m, rule = meta(panel, r_['r'])
                fb = PC.flops_bytes(r_['comp'], m, M, r_['r']) if r_['comp'] in ('N', 'J', 'JTJ', 'chol') else (None, None)
                r_.update(M=M, m=m, rule=rule, case=rep['fixtures']['case_of'][r_['fixture']], flops=fb[0], bytes=fb[1])
            d1 = ref_kernel()
            ok = (1 / 1.1) <= d1 / d0 <= 1.1
            rep['drift'].setdefault(panel, []).append(dict(start=d0, end=d1, ratio=d1 / d0, ok=ok))
            rep.setdefault('all_attempts', {}).setdefault(panel, []).append(rows)
            if ok:
                break
        rep['panels'][panel] = dict(rows=rows, drift_ok=ok, attempts=attempt + 1, ranks=ranks)
        save()
        print('PANEL', panel, el(), flush=True)
    Ar, lr, B, D, P = make_tabs('fixed')(Rmax)
    c = jnp.asarray(fixtures[0][:Rmax])
    Jf = Ar + (dt / (1 + dt * nu * lr))[:, None] * PC.k_J(c, B, D, P)
    Hm = PC.k_JTJ(Jf)
    g = Jf.T @ PC.k_res(c, Ar, lr, B, D, P, nu, dt)
    for comp, (fn, args) in dict(N=(PC.k_N, (c, B, D, P)), J=(PC.k_J, (c, B, D, P)), JTJ=(PC.k_JTJ, (Jf,)),
                                 chol=(PC.k_chol, (Hm, g, jnp.asarray(1e-6)))).items():
        s_ = PC.dispatch_sync(fn, args, reps=10, burn=float(cfg['burn']))
        s_['passed'] = bool(1.8 <= s_['ratio_median'] <= 2.2)
        rep['sync'][comp] = s_
    # equivalence of the kernel components with the vendored 3D advection and Jacobian (A3.3)
    data = dict(B=B, D=D, P=P)
    eqN, eqJ = [], []
    for cf in fixtures:
        cc_ = jnp.asarray(cf[:Rmax])
        vN, vJ = OM.adv_offmesh_batch(data, cc_[None])[0], OM.contract_offmesh(data, cc_)
        eqN.append(float(jnp.max(jnp.abs(PC.k_N(cc_, B, D, P) - vN)) / jnp.max(jnp.abs(vN))))
        eqJ.append(float(jnp.max(jnp.abs(PC.k_J(cc_, B, D, P) - vJ)) / jnp.max(jnp.abs(vJ))))
    rep['component_equivalence'] = dict(N=max(eqN), J=max(eqJ), per_fixture=dict(N=eqN, J=eqJ),
                                        passed=bool(max(eqN) <= 1e-12 and max(eqJ) <= 1e-12))
    save()


def p1e(cfg, rep, save, bank, tb, kx, M_of, n, el):
    tab = C.table(int(cfg['cohort_seed']), int(cfg['cohort_count']))
    assert int(cfg['cohort_seed']) == 923801
    cases = list(cfg['cases'])
    rules = load_rules(cfg['chain'])
    Rp = int(cfg['Rp'])
    M = M_of[Rp]
    trust = float(cfg['trust_fraction'] * bank['spread'][str(Rp)])
    vq = OM.make_fsc_rule(n, Rp, None, OM.contract_offmesh, dt=DT3, gtol=float(cfg['gtol']), trust=trust,
                          adaptive_first=int(cfg['a']))
    lean = dict(GTb=TB.rows_prefix(tb['GTb'], Rp), L=tb['L'][:Rp, :Rp], A=tb['A'][:M, :Rp], lam=tb['lam'][:M])
    lean = jax.tree_util.tree_map(lambda x: jax.block_until_ready(jnp.asarray(x)), lean)
    keep = int(round(.05 / DT3))
    rep['p1e'] = dict(Rp=Rp, M=M, trust=trust, chain=cfg['chain'], m={k: int(len(rules[k][1])) for k in cfg['chain']}, rows=[])
    U0 = {j: jax.block_until_ready(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases}
    NU = {j: float(tab['nu'][j]) for j in cases}
    Wst, Fst, valid = {}, {}, {}
    nl3 = lambda c, d: ((d['B'] @ c) * (d['D'] @ c)) @ d['P']
    chain = cfg['chain']
    for i, rule in enumerate(chain):
        X, w = rules[rule]
        om = OM.offmesh_tables(n, bank['params'], bank['T'][:, :Rp], X, w, kx[:M])
        data = dict(lean, B=om['B'], D=om['D'], P=om['P'])
        nxt = None
        if i + 1 < len(chain):
            X2, w2 = rules[chain[i + 1]]
            nxt = OM.offmesh_tables(n, bank['params'], bank['T'][:, :Rp], X2, w2, kx[:M])
        for j in cases:
            o = jax.block_until_ready(vq(U0[j], NU[j], data, {}))
            W = np.asarray(o[1])[::keep]
            Wst[(rule, j)], Fst[(rule, j)] = W, np.asarray(o[0])
            rs = np.asarray(o[3])
            valid[(rule, j)] = bool(np.isfinite(np.asarray(o[1])).all() and np.isfinite(np.asarray(o[0])).all()
                                    and np.isfinite(np.asarray(o[4])).all() and int((rs == 3).sum()) == 0)
            if nxt is not None:
                rh = []
                for k in range(6):
                    a_ = np.asarray(nl3(jnp.asarray(W[k]), data))
                    b_ = np.asarray(nl3(jnp.asarray(W[k]), nxt))
                    rh.append(rho_a24(a_, b_))
                rep['p1e'].setdefault('rho', {}).setdefault(f'{rule}|{chain[i + 1]}', {})[str(j)] = rh
            print('P1E', rule, j, el(), flush=True)
        del om, nxt, data
        gc.collect()
    for j in cases:
        n0 = float(np.linalg.norm(np.asarray(U0[j])))
        d = {}
        for x_, y_ in zip(chain[:-1], chain[1:]):
            d[f'{x_}|{y_}'] = max(float(np.linalg.norm(Fst[(x_, j)][k] - Fst[(y_, j)][k])) / n0 for k in range(1, 6))
        rep['p1e']['rows'].append(dict(case=j, dist=d, valid={r: valid[(r, j)] for r in chain}))
    np.savez_compressed(Path(cfg['_out']) / 'P1E3D.npz',
                        **{f'W|{r}|{j}': Wst[(r, j)] for r in chain for j in cases})
    save()


if __name__ == '__main__':
    main()
