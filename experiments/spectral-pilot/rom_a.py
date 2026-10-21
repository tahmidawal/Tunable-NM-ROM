"""spectral-pilot Part A: build and score the POD-Galerkin feasibility comparator for one level (accuracy only; timing
is time_a.py). PILOT: small, not a result.

  python rom_a.py A2 [--rmax 384] [--no-quad] [--no-sine]

Writes runs/A/<level>/rom.json: floors (POD span, spectral truncation, sine span, POD from 48 trajectories, random
span), the ROM-T and ROM-Q ladders (per-case errors per (r, n) and (r, Mq, n)), the sine-ROM control ladder, gates G4,
G9, SW4, D2, and offline timings. Operators for timing are saved to runs/A/<level>/ops_r<r>.npz (untracked).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

import common as C
import romlib as RL

R_LADDER = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
N_STEPS = [5 * 2 ** k for k in range(10)]          # 5 .. 2560
MQ_LADDER = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 576, 1152]


def ladder_n(run, label, conv=0.02):
    """Run n ascending until time-converged (relative change of the worst error < conv) or the top (A1.5)."""
    rows, prev = [], None
    for n in N_STEPS:
        e = run(n)
        w = float(np.max(e))
        rows.append(dict(n=n, err=[float(x) for x in e], worst=w, median=float(np.median(e))))
        if prev is not None and np.isfinite(w) and np.isfinite(prev) and abs(w - prev) <= conv * w:
            break
        prev = w
    print(label, ' '.join('%d:%.2e' % (r['n'], r['worst']) for r in rows), flush=True)
    return rows


def floors_from_coefs(Cref, bnorm2, den):
    """Cref (B, 6, R) coefficients on an orthonormal basis; bnorm2 (B, 6). floor(r) = max_t sqrt(||b||^2 - sum_{<r}
    c^2)/den for every r = 1..R. Returns (B, R)."""
    cum = np.cumsum(Cref ** 2, axis=-1)
    f2 = np.maximum(bnorm2[..., None] - cum, 0.0)
    return np.max(np.sqrt(f2[:, 1:, :]), axis=1) / den[:, None], f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('level')
    ap.add_argument('--rmax', type=int, default=None)
    ap.add_argument('--no-quad', action='store_true')
    ap.add_argument('--no-sine', action='store_true')
    a = ap.parse_args()
    lv = a.level
    d = C.RUNS / 'A' / lv
    ref = __import__('json').loads((d / 'ref.json').read_text())
    Nref = ref['Nref']
    n = Nref - 1
    rmax = a.rmax or {'A1': 256, 'A2': 384, 'A3': 512, 'A4': 512}[lv]
    te = np.asarray(ref['test_desc'])
    nu = te[:, 4]
    Bref = np.load(d / 'ref_coef.npy')                       # (12, 6, n, n)
    X = np.load(d / 'train_coef.npy', mmap_mode='r')          # (ntr, 21, n, n)
    rep = dict(level=lv, env=C.env_record(), Nref=Nref, rmax=rmax, label=C.PILOT, timings={})
    bflat = Bref.reshape(len(Bref), 6, -1)
    bnorm2 = np.sum(bflat ** 2, axis=-1)
    den = np.sqrt(bnorm2[:, 0])

    # ---------------------------------------------------------------------------------------------- POD ----
    t0 = time.perf_counter()
    Phi, sv = RL.pod(np.asarray(X).reshape(-1, n, n), rmax)
    rep['timings']['pod'] = time.perf_counter() - t0
    rmax = Phi.shape[1]
    rep['rmax_eff'] = rmax
    rep['sv'] = (sv[:rmax] / sv[0]).tolist()
    rep['orth_dev'] = float(np.max(np.abs(Phi.T @ Phi - np.eye(rmax))))
    Phi48, _ = RL.pod(np.asarray(X[:48]).reshape(-1, n, n), rmax)
    rng = np.random.default_rng(11)
    Prand, _ = np.linalg.qr(rng.standard_normal((n * n, 128)))
    lam = RL.lam_flat(n)
    order = np.argsort(-lam, kind='stable')                   # Laplacian order: smallest |k|^2 first
    fl = {}
    for tag, B_ in (('pod', Phi), ('pod48', Phi48), ('random', Prand)):
        Cf = np.einsum('btn,nr->btr', bflat, B_)
        f, _ = floors_from_coefs(Cf, bnorm2, den)
        fl[tag] = dict(worst=f.max(0).tolist(), median=np.median(f, 0).tolist())
    Cs = bflat[:, :, order[:max(rmax, 4096)]]
    f, _ = floors_from_coefs(Cs, bnorm2, den)
    fl['sine'] = dict(worst=f.max(0).tolist(), median=np.median(f, 0).tolist())
    spec = {}
    for N in range(4, Nref + 1, 4):
        m = N - 1
        tail = bnorm2 - np.sum(Bref[:, :, :m, :m].reshape(len(Bref), 6, -1) ** 2, axis=-1)
        e = np.max(np.sqrt(np.maximum(tail[:, 1:], 0)), axis=1) / den
        spec[N] = dict(worst=float(e.max()), median=float(np.median(e)))
    fl['spectral'] = spec
    rep['floors'] = fl
    print(lv, 'POD done', rep['timings']['pod'], 'rmax', rmax, 'orth', rep['orth_dev'], flush=True)

    # ------------------------------------------------------------------------------------------- tensor ----
    t0 = time.perf_counter()
    Tphi = RL.tensor_full(Phi, Nref)
    rep['timings']['tensor_full'] = time.perf_counter() - t0
    print(lv, 'tensor built', rep['timings']['tensor_full'], flush=True)

    rlist = [r for r in R_LADDER if r <= rmax]
    rom_t, rom_q, rom_s, gates = {}, {}, {}, {}
    Bq_phi = {}
    if not a.no_quad:
        t0 = time.perf_counter()
        for Mq in MQ_LADDER:
            if Mq < 3 * Nref // 2 * 2 and (Mq - 1) ** 2 * 6 <= 2 * rmax ** 2:
                Bq_phi[Mq] = RL.q_values(Phi, Nref, Mq)
        rep['timings']['q_values'] = time.perf_counter() - t0
    for r in rlist:
        t0 = time.perf_counter()
        Psi, D, V = RL.rotate(Phi[:, :r], lam)
        T = RL.rotate_tensor(Tphi[:r, :r, :r], V)
        Tp, iu, ju = RL.pack(T)
        z0 = RL.ic_project(Psi, Nref, te)
        Cref = np.einsum('btn,nr->btr', bflat, Psi)
        floor2 = bnorm2 - np.sum(Cref ** 2, axis=-1)
        floor2 = np.maximum(floor2, 0)
        tb = time.perf_counter() - t0
        NLt = RL.nl_tensor(Tp, iu, ju)

        def run_t(nn, NL=NLt):
            Z = np.asarray(RL.make_rom_batch(NL, D, nn)(jnp.asarray(z0), jnp.asarray(nu)))
            e, _ = RL.rom_errors(Z, Cref, floor2, den)
            return e
        rom_t[r] = dict(build_s=tb, rows=ladder_n(run_t, f'{lv} ROM-T r={r}'))
        # D2: every run >= its floor
        fr = np.max(np.sqrt(floor2[:, 1:]), axis=1) / den
        for row in rom_t[r]['rows']:
            assert all((not np.isfinite(e_)) or e_ >= f_ * (1 - 1e-9) for e_, f_ in zip(row['err'], fr)), 'D2'
        np.savez(d / f'ops_r{r}.npz', Psi=Psi, D=D, Tp=Tp, iu=iu, ju=ju)
        # ROM-Q at the admissible grid sizes (6 m r <= 2 r^3, DESIGN A1 pruning registered in A2)
        if not a.no_quad:
            rom_q[r] = {}
            for Mq, (Bq, Dq) in Bq_phi.items():
                m = (Mq - 1) ** 2
                if 6 * m * r > 2 * r ** 3:
                    continue
                NLq = RL.nl_quad(Bq[:, :r] @ V, Dq[:, :r] @ V, Mq)
                rom_q[r][Mq] = ladder_n(lambda nn, NL=NLq: run_t(nn, NL), f'{lv} ROM-Q r={r} Mq={Mq}')
        # gates at r in {32, 128}: G4 (tensor vs exact evaluation), SW4 (corrupted slice)
        if r in (32, 128):
            NLe = RL.nl_exact(Psi, Nref)
            nn = rom_t[r]['rows'][-1]['n']
            run2 = lambda NL: np.asarray(RL.make_rom_batch(NL, D, nn)(jnp.asarray(z0[:2]), jnp.asarray(nu[:2])))
            Zt, Ze = run2(NLt), run2(NLe)
            g4 = float(np.max(np.abs(Zt - Ze)) / np.max(np.abs(Ze)))
            zs = Ze.reshape(-1, r)
            Nv = np.stack([np.asarray(NLt(jnp.asarray(z))) for z in zs])
            ks = int(np.argmax(np.max(np.abs(Nv), axis=0)))
            Tc = Tp.copy()
            Tc[ks] = 0
            Zc = run2(RL.nl_tensor(Tc, iu, ju))
            sw4 = float(np.max(np.abs(Zc - Zt)) / np.max(np.abs(Zt)))
            gates[r] = dict(G4_rel=g4, G4_pass=bool(g4 <= 1e-9), SW4_rel=sw4, SW4_fired=bool(sw4 >= 1e-6),
                            SW4_slice=ks, n=nn)
            print(lv, 'gates r', r, gates[r], flush=True)
        # sine-ROM control (first r Laplacian-ordered sines), r <= 256
        if not a.no_sine and r <= 256:
            Phs = np.zeros((n * n, r))
            Phs[order[:r], np.arange(r)] = 1.0
            Ts = RL.tensor_full(Phs, Nref)
            Psi_s, D_s, V_s = RL.rotate(Phs, lam)
            Tps, ius, jus = RL.pack(RL.rotate_tensor(Ts, V_s))
            z0s = RL.ic_project(Psi_s, Nref, te)
            Crs = np.einsum('btn,nr->btr', bflat, Psi_s)
            f2s = np.maximum(bnorm2 - np.sum(Crs ** 2, axis=-1), 0)
            NLs = RL.nl_tensor(Tps, ius, jus)

            def run_s(nn):
                Z = np.asarray(RL.make_rom_batch(NLs, D_s, nn)(jnp.asarray(z0s), jnp.asarray(nu)))
                return RL.rom_errors(Z, Crs, f2s, den)[0]
            rom_s[r] = ladder_n(run_s, f'{lv} sine-ROM r={r}')
        C.dump(d / 'rom.json', dict(rep, rom_t=rom_t, rom_q=rom_q, rom_sine=rom_s, gates=gates))

    # G9: padding convergence of the tensor at r = 64 (M = 3N/2 vs 2N), rollout difference
    r = min(64, rmax)
    Psi, D, V = RL.rotate(Phi[:, :r], lam)
    z0 = RL.ic_project(Psi, Nref, te)
    nn = rom_t[r]['rows'][-1]['n']
    Za = np.asarray(RL.make_rom_batch(RL.nl_tensor(*RL.pack(RL.rotate_tensor(Tphi[:r, :r, :r], V))), D, nn)(
        jnp.asarray(z0), jnp.asarray(nu)))
    T2 = RL.rotate_tensor(tensor_full_padded(Phi[:, :r], Nref), V)
    Zb = np.asarray(RL.make_rom_batch(RL.nl_tensor(*RL.pack(T2)), D, nn)(jnp.asarray(z0), jnp.asarray(nu)))
    g9 = np.max(np.sqrt(np.sum((Za - Zb) ** 2, axis=-1))[:, 1:], axis=1) / den
    gates['G9'] = dict(r=r, n=nn, max=float(np.max(g9)))
    print(lv, 'G9', gates['G9'], flush=True)
    C.dump(d / 'rom.json', dict(rep, rom_t=rom_t, rom_q=rom_q, rom_sine=rom_s, gates=gates))
    print(lv, 'done', flush=True)


def tensor_full_padded(Phi, N):
    """The tensor with a 2N-point padded grid instead of 3N/2 (G9): embed coefficients in a larger N' = 4N/3 grid so
    that pscore's M = 3N'/2 = 2N; the extra modes are zero, the products are projected back onto the original modes."""
    n = N - 1
    N2 = 4 * N // 3
    assert 3 * N2 // 2 == 2 * N, N
    r = Phi.shape[1]
    Pb = Phi.T.reshape(r, n, n)
    Pb = np.pad(Pb, ((0, 0), (0, N2 - N), (0, N2 - N))).reshape(r, -1).T
    return RL.tensor_full(Pb, N2)


if __name__ == '__main__':
    main()
