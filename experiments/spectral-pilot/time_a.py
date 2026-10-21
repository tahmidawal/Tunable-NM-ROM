"""spectral-pilot Part A timing (DESIGN 6, A1.5, A2.3, A3.1, A4). PILOT: small, not a result.

Run on ONE core with single-thread flags (asserted):
  taskset -c 4 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python time_a.py A2 screen
  ... python time_a.py A2 final

screen: (1) cost model R4: per-evaluation time of the FOM nonlinear term at every ladder N, the packed tensor at every r,
        and the grid form at every (r, Mq) run; (2) candidates per target: for each size (and form), the smallest
        observed step count that MEETS tau (three-way rule, A3.1); for ROM candidates, N_init is selected (A2.3) and the
        candidate re-scored with it before timing (A4); (3) each candidate is timed once (whole query, case 0) and fully
        timed (5 repetitions on each of the 3 timing cases) unless that single time exceeds 3x the method's best
        fully-timed query so far for that tau (heuristic, A5, reported); candidates alternate FOM/ROM; sentinel at the
        start and end of the screen.
final:  fresh process; the cost-optimal FOM and ROM per target re-timed interleaved F, R, F, R with a sentinel at the
        start and end (drift > 10 % -> re-run once, then invalid), the secondary native contract, plus diagnostic H1 (two-case call vs the sum of single calls).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import common as C
import romlib as RL
import score as SC
import tutil as TU

REPS = 5
TIMING_CASES = [0, 1, 2]
NINIT_LADDER = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]


def check_env():
    xf = os.environ.get('XLA_FLAGS', '')
    assert 'intra_op_parallelism_threads=1' in xf and 'multi_thread_eigen=false' in xf, xf
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(k) == '1', k
    aff = os.sched_getaffinity(0)
    assert len(aff) == 1, aff
    return dict(xla_flags=xf, affinity=sorted(aff))


def tmed(fn, arg, reps=REPS):
    jax.block_until_ready(fn(arg))
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(arg))
        out.append(time.perf_counter() - t)
    return out


# ------------------------------------------------------------------------------------------------- queries -----

def fom_query(N, n):
    f = C.make_fom(N, n, 5, lattice=True)
    return lambda d: f(d[None])


class RomOps:
    def __init__(self, d, Nref, r):
        z = np.load(d / f'ops_r{r}.npz')
        self.Psi, self.D, self.Tp, self.iu, self.ju = z['Psi'], z['D'], z['Tp'], z['iu'], z['ju']
        self.Nref, self.r = Nref, r
        n = Nref - 1
        lat = jax.vmap(lambda b: C.synth_lattice(b, Nref))(jnp.asarray(self.Psi.T.reshape(r, n, n)))
        self.Plat = jnp.asarray(np.asarray(lat).reshape(r, -1).T)         # (257^2, r)

    def nl(self, form, Mq=None):
        if form == 'T':
            return RL.nl_tensor(self.Tp, self.iu, self.ju)
        Bq, Dq = RL.q_values(self.Psi, self.Nref, Mq)
        return RL.nl_quad(Bq, Dq, Mq)

    def query(self, form, n, Ninit, Mq=None, native=False):
        NL = self.nl(form, Mq)
        fn = NL.fn
        n_ = self.Nref - 1
        rule = C.ICRule(Ninit)
        Pinit = jnp.asarray(self.Psi.T.reshape(self.r, n_, n_)[:, :Ninit - 1, :Ninit - 1])
        Dj = jnp.asarray(self.D)
        Plat = self.Plat
        X, SW = rule.X, rule.SW

        @jax.jit
        def q(d, ops, Pinit, Dj, Plat, X, SW):              # operator arrays are explicit jit arguments
            cx, cy, w, a = d[0], d[1], d[2], d[3]
            gx = jnp.exp(-(X - cx) ** 2 / (2 * w * w)) @ SW
            gy = jnp.exp(-(X - cy) ** 2 / (2 * w * w)) @ SW
            z0 = a * jnp.einsum('kij,i,j->k', Pinit, gx, gy)
            Z = C.rollout(z0, d[4] * Dj, lambda z: fn(z, ops), n, 5)
            return Z if native else Z @ Plat.T
        return lambda d: q(d, NL.ops, Pinit, Dj, Plat, X, SW)


# ----------------------------------------------------------------------------------------------- screening -----

def candidates(sizes_rows, U, tau):
    """{key: smallest n meeting tau} over sizes; rows = list of dict(n, worst)."""
    out = {}
    for key, rows in sizes_rows.items():
        ok = [r['n'] for r in rows if SC.classify(r['worst'], U, tau) == 'meets']
        if ok:
            out[key] = min(ok)
    return out


def ninit_select(Psi, Nref, te, Bref0, tau):
    """Smallest N_init with ||b0 - trunc_{Ninit}(b0)|| / ||b0|| <= tau / 20 on every test case (A2.3)."""
    for Ni in [x for x in NINIT_LADDER if x <= Nref] + [Nref]:
        m = Ni - 1
        tail = np.sqrt(np.sum(Bref0 ** 2, axis=(-1, -2)) - np.sum(Bref0[:, :m, :m] ** 2, axis=(-1, -2)))
        if np.all(tail / np.sqrt(np.sum(Bref0 ** 2, axis=(-1, -2))) <= tau / 20):
            return Ni
    return Nref


def screen(lv):
    d = C.RUNS / 'A' / lv
    ref = json.loads((d / 'ref.json').read_text())
    fom = json.loads((d / 'fom.json').read_text())
    rom = json.loads((d / 'rom.json').read_text())
    Nref = ref['Nref']
    te = np.asarray(ref['test_desc'])
    nu = te[:, 4]
    U = fom['U']
    Bref = np.load(d / 'ref_coef.npy')
    bflat = Bref.reshape(len(Bref), 6, -1)
    bnorm2 = np.sum(bflat ** 2, axis=-1)
    den = np.sqrt(bnorm2[:, 0])
    rep = dict(level=lv, env=C.env_record(), threads=check_env(), U=U, label=C.PILOT, cost_model={}, targets={})
    if U is None:
        rep['status'] = 'unresolved: allowance unavailable'
        C.dump(d / 'time_screen.json', rep)
        return
    sent = fom_query(32, 80)
    rep['sentinel_start'] = [tmed(sent, jnp.asarray(te[c])) for c in TIMING_CASES]
    # (1) cost model
    rng = np.random.default_rng(0)
    cm = dict(fom={}, rom_t={}, rom_q={})
    for N in sorted(int(k) for k in fom['sizes']):
        b = jnp.asarray(rng.standard_normal((N - 1, N - 1)) * 1e-3)
        f = jax.jit(C.fom_nl(N))
        ts_ = tmed(f, b, 20)
        cm['fom'][N] = float(np.median(ts_))
        cm.setdefault('reps', {})[f'fom|{N}'] = ts_
    ops = {}
    for r in sorted(int(k) for k in rom['rom_t']):
        ops[r] = RomOps(d, Nref, r)
        z = jnp.asarray(rng.standard_normal(r) * 1e-2)
        nlo = ops[r].nl('T')
        ts_ = tmed(lambda zz, fj=jax.jit(nlo.fn), o=nlo.ops: fj(zz, o), z, 20)
        cm['rom_t'][r] = float(np.median(ts_))
        cm.setdefault('reps', {})[f'rom_t|{r}'] = ts_
        for Mq in rom['rom_q'].get(str(r), {}):
            nlq = ops[r].nl('Q', int(Mq))
            ts_ = tmed(lambda zz, fj=jax.jit(nlq.fn), o=nlq.ops: fj(zz, o), z, 20)
            cm['rom_q'][f'{r}|{Mq}'] = float(np.median(ts_))
            cm['reps'][f'rom_q|{r}|{Mq}'] = ts_
    rep['cost_model'] = cm
    print(lv, 'cost model done', flush=True)
    fom_rows = {int(k): v['rows'] for k, v in fom['sizes'].items()}
    romt_rows = {('T', int(r), None): v['rows'] for r, v in rom['rom_t'].items()}
    romq_rows = {('Q', int(r), int(Mq)): rows for r, dd in rom['rom_q'].items() for Mq, rows in dd.items()}
    for tau in SC.TARGETS:
        cf = candidates(fom_rows, U, tau)
        cr = candidates({**romt_rows, **romq_rows}, U, tau)
        keep_f, keep_r = list(cf), list(cr)
        entry = dict(fom_candidates={str(k): v for k, v in cf.items()},
                     rom_candidates={'|'.join(map(str, k)): v for k, v in cr.items()},
                     fom_timed=[], rom_timed=[], not_timed_heuristic=dict(fom=[], rom=[]))
        # ROM: N_init selection + re-score before timing (A4)
        rescored = {}
        for k in keep_r:
            form, r, Mq = k
            o = ops[r]
            Ni = ninit_select(o.Psi, Nref, te, Bref[:, 0], tau)
            z0 = RL.ic_project(o.Psi, Nref, te, Ninit=Ni)
            Cref = np.einsum('btn,nr->btr', bflat, o.Psi)
            f2 = np.maximum(bnorm2 - np.sum(Cref ** 2, axis=-1), 0)
            NL = o.nl(form, Mq)
            Z = np.asarray(RL.make_rom_batch(NL, o.D, cr[k])(jnp.asarray(z0), jnp.asarray(nu)))
            e = RL.rom_errors(Z, Cref, f2, den)[0]
            w = float(np.max(e))
            st = SC.classify(w, U, tau)
            if st != 'meets' and Ni != Nref:          # fall back to the next larger Ninit until it meets (A2.3)
                for Ni2 in [x for x in NINIT_LADDER if Ni < x <= Nref] + [Nref]:
                    z0 = RL.ic_project(o.Psi, Nref, te, Ninit=Ni2)
                    Z = np.asarray(RL.make_rom_batch(NL, o.D, cr[k])(jnp.asarray(z0), jnp.asarray(nu)))
                    e = RL.rom_errors(Z, Cref, f2, den)[0]
                    w, Ni = float(np.max(e)), Ni2
                    st = SC.classify(w, U, tau)
                    if st == 'meets':
                        break
            rescored[k] = dict(Ninit=Ni, worst=w, status=st)
        # (3) call-level interleaved whole-query timing of all candidates (tutil.roundrobin; one-rep pruning, A5)
        cands = []
        for N in keep_f:
            cands.append((('F', N), 'F', fom_query(N, cf[N])))
        for k in keep_r:
            if rescored[k]['status'] == 'meets':
                form, r, Mq = k
                cands.append((('R',) + k, 'R', ops[r].query(form, cr[k], rescored[k]['Ninit'], Mq)))
        fc = [c for c in cands if c[1] == 'F']
        rc = [c for c in cands if c[1] == 'R']
        alt = [x for pair in zip(fc, rc) for x in pair] + fc[len(rc):] + rc[len(fc):]
        xs = [jnp.asarray(te[c]) for c in TIMING_CASES]
        times, one, pruned = TU.roundrobin(alt, xs, REPS)
        for key in pruned:
            entry['not_timed_heuristic']['fom' if key[0] == 'F' else 'rom'].append(dict(key=str(key), t_one=one[key]))
        for key, ts in times.items():
            t = float(np.median([np.median(x) for x in ts]))
            if key[0] == 'F':
                N = key[1]
                entry['fom_timed'].append(dict(N=N, n=cf[N], times=ts, t=t,
                                               worst=min(r['worst'] for r in fom_rows[N] if r['n'] == cf[N])))
            else:
                k = key[1:]
                entry['rom_timed'].append(dict(form=k[0], r=k[1], Mq=k[2], n=cr[k], Ninit=rescored[k]['Ninit'],
                                               times=ts, t=t, worst=rescored[k]['worst']))
        entry['rescored'] = {'|'.join(map(str, k)): v for k, v in rescored.items()}
        bf = min(entry['fom_timed'], key=lambda x: x['t']) if entry['fom_timed'] else None
        br = min(entry['rom_timed'], key=lambda x: x['t']) if entry['rom_timed'] else None
        entry['fom_best'], entry['rom_best'] = bf, br
        entry['S_screen'] = (bf['t'] / br['t']) if (bf and br) else None
        rep['targets'][str(tau)] = entry
        print(lv, 'tau', tau, 'FOM', bf and (bf['N'], bf['n'], '%.2e s' % bf['t']),
              'ROM', br and (br['form'], br['r'], br['Mq'], br['n'], '%.2e s' % br['t']), 'S', entry['S_screen'], flush=True)
        C.dump(d / 'time_screen.json', rep)
    s1 = [tmed(sent, jnp.asarray(te[c])) for c in TIMING_CASES]
    rep['sentinel_end'] = s1
    rep['sentinel_drift'] = abs(np.median(s1) / np.median(rep['sentinel_start']) - 1)
    rep['screen_valid'] = bool(rep['sentinel_drift'] <= 0.10)
    C.dump(d / 'time_screen.json', rep)


def final(lv, attempt=1):
    d = C.RUNS / 'A' / lv
    ref = json.loads((d / 'ref.json').read_text())
    scr = json.loads((d / 'time_screen.json').read_text())
    Nref = ref['Nref']
    te = np.asarray(ref['test_desc'])
    rep = dict(level=lv, env=C.env_record(), threads=check_env(), label=C.PILOT, targets={},
               screen_valid=scr.get('screen_valid', False))
    sent = fom_query(32, 80)
    s0 = [tmed(sent, jnp.asarray(te[c])) for c in TIMING_CASES]
    ops = {}
    for tau, e in scr['targets'].items():
        bf, br = e['fom_best'], e['rom_best']
        if not (bf and br):
            rep['targets'][tau] = dict(S=None, reason='a method did not reach tau (censored)',
                                       fom=bool(bf), rom=bool(br))
            continue
        qf = fom_query(bf['N'], bf['n'])
        if br['r'] not in ops:
            ops[br['r']] = RomOps(d, Nref, br['r'])
        qr = ops[br['r']].query(br['form'], br['n'], br['Ninit'], br['Mq'])
        tf, tr = [], []
        for c in TIMING_CASES:
            x = jnp.asarray(te[c])
            jax.block_until_ready(qf(x))
            jax.block_until_ready(qr(x))
            a_, b_ = [], []
            for _ in range(REPS):
                t = time.perf_counter()
                jax.block_until_ready(qf(x))
                a_.append(time.perf_counter() - t)
                t = time.perf_counter()
                jax.block_until_ready(qr(x))
                b_.append(time.perf_counter() - t)
            tf.append(a_)
            tr.append(b_)
        mf = float(np.median([np.median(x) for x in tf]))
        mr = float(np.median([np.median(x) for x in tr]))
        # secondary "native" contract (no output synthesis): FOM coefficients, ROM coefficients z
        fn_ = C.make_fom(bf['N'], bf['n'], 5, lattice=False)
        qfn = lambda dd: fn_(dd[None])
        qrn = ops[br['r']].query(br['form'], br['n'], br['Ninit'], br['Mq'], native=True)
        nf = float(np.median([np.median(tmed(qfn, jnp.asarray(te[c]))) for c in TIMING_CASES]))
        nr = float(np.median([np.median(tmed(qrn, jnp.asarray(te[c]))) for c in TIMING_CASES]))

        def outl(ts):
            ts = np.asarray(ts).ravel()
            return int(np.sum(ts > 1.2 * np.median(ts)))
        rep['targets'][tau] = dict(fom=dict(N=bf['N'], n=bf['n'], times=tf, t=mf, spread=float(np.max(tf) / np.min(tf)),
                                            outliers=outl(tf), worst=bf['worst']),
                                   rom=dict(form=br['form'], r=br['r'], Mq=br['Mq'], n=br['n'], Ninit=br['Ninit'],
                                            times=tr, t=mr, spread=float(np.max(tr) / np.min(tr)), outliers=outl(tr),
                                            worst=br['worst']),
                                   S=mf / mr, native=dict(fom=nf, rom=nr, S=nf / nr), order_changed=bool((mf / mr >= 1) != (e['S_screen'] >= 1)))
        print(lv, 'final tau', tau, 'S=%.3g' % (mf / mr), flush=True)
    s1 = [tmed(sent, jnp.asarray(te[c])) for c in TIMING_CASES]
    m0, m1 = float(np.median(s0)), float(np.median(s1))
    rep['sentinel'] = dict(start=s0, end=s1, drift=abs(m1 / m0 - 1), valid=bool(abs(m1 / m0 - 1) <= 0.10))
    # H1 diagnostic: a two-case call vs the sum of the single calls (FOM sentinel configuration)
    f2 = C.make_fom(32, 80, 5, lattice=True)
    two = tmed(f2, jnp.asarray(te[:2]))
    one = [tmed(f2, jnp.asarray(te[c:c + 1])) for c in (0, 1)]
    rep['H1'] = dict(two=float(np.median(two)), sum_single=float(np.median(one[0]) + np.median(one[1])),
                     ratio=float(np.median(two) / (np.median(one[0]) + np.median(one[1]))))
    rep['attempt'] = attempt
    if not rep['sentinel']['valid'] and attempt == 1:          # drift > 10 %: re-run once (A1.5)
        print(lv, 'sentinel drift', rep['sentinel']['drift'], '-> re-run', flush=True)
        C.dump(d / 'time_final_attempt1.json', rep)
        return final(lv, attempt=2)
    rep['timing_valid'] = rep['sentinel']['valid']
    C.dump(d / 'time_final.json', rep)
    print(lv, 'sentinel drift', rep['sentinel']['drift'], 'H1', rep['H1']['ratio'], flush=True)


if __name__ == '__main__':
    lv, phase = sys.argv[1], sys.argv[2]
    screen(lv) if phase == 'screen' else final(lv)
