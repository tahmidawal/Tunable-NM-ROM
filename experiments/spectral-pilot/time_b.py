"""spectral-pilot Part B timing (DESIGN 6, A1.5, A2.3, A3.2, A4). PILOT: small, not a result.

One core, single-thread flags (asserted), as time_a.py:
  taskset -c 5 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python time_b.py screen|final

Queries (A2.3, A3.2): F-IMEX = nodal IC, both SuperLU factorisations, n SBDF2 steps, i/64 lattice output (the sparse
Laplacian of each L is case-independent and prebuilt for both FD families' and the ROM's benefit alike);
F-RK4 = nodal IC, n RK4 steps, lattice output; ROM-T = IC projection on the L_init sub-lattice
z0 = (512/L_init)^2 Psi_sub^T u0_sub, ETDRK4 coefficients for r modes, n steps, decode Plat z.
screen: per target and family, the smallest observed n meeting tau at each size; each candidate is first timed once
on case 0 and fully timed (3 cases x 5 repetitions) only if that single time is <= 3x the family-and-target best so
far (heuristic, reported). final: fresh process, best FOM (over both families) vs best ROM per target, interleaved,
sentinel at start and end.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import common as C
import fd_l as F
import romlib as RL
import score as SC
import tutil as TU
from rom_b import lat_rows
from time_a import check_env, REPS, TIMING_CASES

D_ = C.RUNS / 'B'
LINIT = [32, 64, 128, 256, 512]


def t_np(fn, x, reps=REPS):
    jax.block_until_ready(fn(x))
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(x))
        out.append(time.perf_counter() - t)
    return out


_AM = {}


def fom_query(fam, L, n):
    if fam == 'IMEX':
        if L not in _AM:
            _AM[L] = F.lap_matrix(L)
        A, m = _AM[L]
        return lambda d: F.imex_query(np.asarray(d), L, n, A, m, out_lattice=True)
    q = F.make_rk4(L, n, out_lattice=True)
    return lambda d: q(jnp.asarray(d))


class RomB:
    def __init__(self, r):
        z = np.load(D_ / f'ops_r{r}.npz')
        self.Psi, self.D, self.Tp, self.iu, self.ju, self.Plat = (z[k] for k in ('Psi', 'D', 'Tp', 'iu', 'ju', 'Plat'))
        self.r = r
        self.m, X, Y = F.mask(512)
        self.X, self.Y = X[self.m], Y[self.m]

    def sub_rows(self, Li):
        """Rows of the L=512 interior vector on the i/Li sub-lattice (inside Omega_L)."""
        return lat_rows(self.m, 512, Li)

    def z0_np(self, d, Li):
        rows = self.sub_rows(Li)
        cx, cy, w, a = d[:4]
        u = a * np.exp(-((self.X[rows] - cx) ** 2 + (self.Y[rows] - cy) ** 2) / (2 * w * w))
        return (512 / Li) ** 2 * (self.Psi[rows].T @ u)

    def query(self, n, Li):
        rows = self.sub_rows(Li)
        Ps = jnp.asarray(self.Psi[rows])
        Xs, Ys = jnp.asarray(self.X[rows]), jnp.asarray(self.Y[rows])
        NL = RL.nl_tensor(self.Tp, self.iu, self.ju)
        fn = NL.fn
        Dj, Plat = jnp.asarray(self.D), jnp.asarray(self.Plat)
        sc = (512 / Li) ** 2

        @jax.jit
        def q(d, ops, Ps, Xs, Ys, Dj, Plat):                # operator arrays are explicit jit arguments
            cx, cy, w, a = d[0], d[1], d[2], d[3]
            u = a * jnp.exp(-((Xs - cx) ** 2 + (Ys - cy) ** 2) / (2 * w * w))
            z0 = sc * (Ps.T @ u)
            Z = C.rollout(z0, d[4] * Dj, lambda z: fn(z, ops), n, 5)
            return Z @ Plat.T
        return lambda d: q(jnp.asarray(d), NL.ops, Ps, Xs, Ys, Dj, Plat)


def screen():
    ref = json.loads((D_ / 'ref.json').read_text())
    fom = json.loads((D_ / 'fom.json').read_text())
    rom = json.loads((D_ / 'rom.json').read_text())
    from gen_b import sub
    R = np.load(D_ / 'ref_lat256.npy')[..., sub(256, 64)]
    den = np.sqrt(np.sum(R[:, 0] ** 2, axis=-1))
    te = np.asarray(ref['test_desc'])
    nu = te[:, 4]
    U = fom['U']
    rep = dict(env=C.env_record(), threads=check_env(), U=U, scoring=fom['scoring'], label=C.PILOT, targets={},
               basis_label='unestablished (A3.4)')
    if U is None:
        rep['status'] = 'unresolved: allowance unavailable'
        C.dump(D_ / 'time_screen.json', rep)
        return
    sent = fom_query('IMEX', 128, 80)
    rep['sentinel_start'] = [t_np(sent, te[c]) for c in TIMING_CASES]
    u0n = {}
    m512, _, _ = F.mask(512)
    for j, d in enumerate(te):
        u0n[j] = np.linalg.norm(F.ic(512, d)[m512])
    roms = {}
    m512 = F.mask(512)[0]
    from rom_b import lat_rows as _lr
    u0lat = {j: np.linalg.norm(F.lat_vals(F.ic(512, d), 512)) for j, d in enumerate(te)}
    xs = [te[c] for c in TIMING_CASES]
    for tau in SC.TARGETS:
        e = dict(fom={}, rom={}, skipped=[])
        cands, meta = [], {}
        for fam, sizes in fom['families'].items():
            for L, v in sorted(sizes.items(), key=lambda kv: int(kv[0])):
                if v['diagnostic_only']:
                    continue
                ok = [r['n'] for r in v['rows'] if SC.classify(r['worst'], U, tau) == 'meets']
                if not ok:
                    continue
                n = min(ok)
                key = ('F', fam, int(L), n)
                meta[key] = dict(fam=fam, L=int(L), n=n, worst=min(r['worst'] for r in v['rows'] if r['n'] == n))
                cands.append((key, 'F', fom_query(fam, int(L), n)))
        for r, v in sorted(rom['rom_t'].items(), key=lambda kv: int(kv[0])):
            r = int(r)
            ok = [x['n'] for x in v['rows'] if SC.classify(x['worst'], U, tau) == 'meets']
            if not ok:
                continue
            n = min(ok)
            if r not in roms:
                roms[r] = RomB(r)
            o = roms[r]
            # L_init selection in the scoring norm (decoded lattice values / ||u0|| on the lattice), then re-score (A4)
            zfull = np.stack([o.z0_np(d, 512) for d in te])
            Li = 512
            for Lc in LINIT:
                zc = np.stack([o.z0_np(d, Lc) for d in te])
                dl = np.linalg.norm((zc - zfull) @ o.Plat.T, axis=1) / np.array([u0lat[j] for j in range(len(te))])
                if np.all(dl <= tau / 20):
                    Li = Lc
                    break
            for Lc in [x for x in LINIT if x >= Li]:
                z0 = np.stack([o.z0_np(d, Lc) for d in te])
                Z = np.asarray(RL.make_rom_batch(RL.nl_tensor(o.Tp, o.iu, o.ju), o.D, n)(jnp.asarray(z0), jnp.asarray(nu)))
                w = float(np.max(F.err_lat(np.einsum('btr,pr->btp', Z, o.Plat), R)))
                if SC.classify(w, U, tau) == 'meets':
                    Li = Lc
                    break
            else:
                continue
            key = ('R', r, n, Li)
            meta[key] = dict(r=r, n=n, Linit=Li, worst=w)
            cands.append((key, 'R', o.query(n, Li)))
        fc = [c for c in cands if c[1] == 'F']
        rc = [c for c in cands if c[1] == 'R']
        alt = [x for pair in zip(fc, rc) for x in pair] + fc[len(rc):] + rc[len(fc):]
        times, one, pruned = TU.roundrobin(alt, xs, REPS)
        e['leaders'] = TU.roundrobin.last
        e['skipped'] = [dict(key=str(k), t_one=one[k]) for k in pruned]
        for key, ts in times.items():
            t = float(np.median([np.median(x) for x in ts]))
            tgt = e['fom'] if key[0] == 'F' else e['rom']
            tgt['|'.join(map(str, key[1:]))] = dict(meta[key], t=t, times=ts)
        bf = min(e['fom'].values(), key=lambda x: x['t']) if e['fom'] else None
        br = min(e['rom'].values(), key=lambda x: x['t']) if e['rom'] else None
        e['fom_best'], e['rom_best'] = bf, br
        e['S_screen'] = bf['t'] / br['t'] if (bf and br) else None
        rep['targets'][str(tau)] = e
        print('B tau', tau, 'FOM', bf and (bf['fam'], bf['L'], bf['n'], '%.3e' % bf['t']),
              'ROM', br and (br['r'], br['n'], br['Linit'], '%.3e' % br['t']), 'S', e['S_screen'], flush=True)
        C.dump(D_ / 'time_screen.json', rep)
    rep['sentinel_end'] = [t_np(sent, te[c]) for c in TIMING_CASES]
    rep['sentinel_drift'] = abs(np.median(rep['sentinel_end']) / np.median(rep['sentinel_start']) - 1)
    rep['screen_valid'] = bool(rep['sentinel_drift'] <= 0.10)
    C.dump(D_ / 'time_screen.json', rep)


def final(attempt=1):
    ref = json.loads((D_ / 'ref.json').read_text())
    scr = json.loads((D_ / 'time_screen.json').read_text())
    te = np.asarray(ref['test_desc'])
    rep = dict(env=C.env_record(), threads=check_env(), label=C.PILOT, targets={},
               screen_valid=scr.get('screen_valid', False), screen_time=scr['env']['time'], native_contract='not implemented for Part B (A6)')
    sent = fom_query('IMEX', 128, 80)
    s0 = [t_np(sent, te[c]) for c in TIMING_CASES]
    for tau, e in scr['targets'].items():
        bf, br = e['fom_best'], e['rom_best']
        if not (bf and br):
            rep['targets'][tau] = dict(S=None, reason='a method did not reach tau (censored)', fom=bool(bf), rom=bool(br))
            continue
        qf = fom_query(bf['fam'], bf['L'], bf['n'])
        qr = RomB(br['r']).query(br['n'], br['Linit'])
        tf, tr = [], []
        for c in TIMING_CASES:
            x = te[c]
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
        sp = lambda T_: float(np.max(T_) / np.min(T_))
        ol = lambda T_: int(np.sum(np.asarray(T_) > 1.2 * np.median(T_)))
        rep['targets'][tau] = dict(fom=dict(bf, times=tf, t=mf, spread=sp(tf), outliers=ol(tf)),
                                   rom=dict(br, times=tr, t=mr, spread=sp(tr), outliers=ol(tr)), S=mf / mr,
                                   order_changed=bool((mf / mr >= 1) != (e['S_screen'] >= 1)))
        print('B final tau', tau, 'S=%.3g' % (mf / mr), flush=True)
    s1 = [t_np(sent, te[c]) for c in TIMING_CASES]
    m0, m1 = float(np.median(s0)), float(np.median(s1))
    rep['sentinel'] = dict(start=s0, end=s1, drift=abs(m1 / m0 - 1), valid=bool(abs(m1 / m0 - 1) <= 0.10))
    rep['attempt'] = attempt
    if not rep['sentinel']['valid'] and attempt == 1:
        C.dump(D_ / 'time_final_attempt1.json', rep)
        return final(attempt=2)
    rep['timing_valid'] = rep['sentinel']['valid']
    rep['scoring'] = scr['scoring']
    C.dump(D_ / 'time_final.json', rep)
    print('B sentinel drift', rep['sentinel']['drift'], flush=True)


if __name__ == '__main__':
    screen() if sys.argv[1] == 'screen' else final()
