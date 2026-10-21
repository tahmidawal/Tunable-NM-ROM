"""imp-multimesh core (DESIGN.md 3.2-3.4, 7): continuous-POD resampling, low-memory ECSW construction for large
meshes, rule reuse across nested meshes, and the competitor run/qualification loop.

Built on the byte-identical jcp-hr-baselines modules (hrcore, build, rom, pipeline, score), imported from
experiments/jcp-hr-baselines (staged into the job directory unchanged). Nothing here alters their arithmetic:
the blocked contribution builder reproduces build.ecsw_contrib (checked by control 5).

Grid conventions (hrcore): interior node (i, j), i, j = 0..L-2, at x = (i+1)/L, y = (j+1)/L, flat index
i (L-1) + j (x slowest). Basis Phi (n, r), n = (L-1)^2.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
HRB = HERE.parent / 'jcp-hr-baselines'
if str(HRB) not in sys.path:
    sys.path.insert(0, str(HRB))

import hrcore as H          # noqa: E402
import build as BD          # noqa: E402
import rom as RM            # noqa: E402
import pipeline as PL       # noqa: E402

D = 2
ORTH_TOL = 1e-10


def sync(x):
    return jax.block_until_ready(x)


# ------------------------------------------------------------------------------------------ resampling ----

def _dst1(x, mutate=None):
    """Orthonormal self-inverse DST-I on the last axis (engines.dst1 arithmetic). mutate='shift': control 2a."""
    n = x.shape[-1]
    zero = jnp.zeros(x.shape[:-1] + (1,), x.dtype)
    ext = jnp.concatenate((zero, x, zero, -x[..., ::-1]), axis=-1)
    y = -jnp.fft.fft(ext, axis=-1).imag
    y = y[..., 2:n + 2] if mutate == 'shift' else y[..., 1:n + 1]
    return y / jnp.sqrt(2. * (n + 1))


def _dst2(x, mutate=None):
    return _dst1(_dst1(x, mutate).swapaxes(-1, -2), mutate).swapaxes(-1, -2)


def resample_fn(Ls, Lt, mutate=None):
    """Jitted map of a block of modes (n_s, b) on mesh Ls to (n_t, b) on mesh Lt by the exact sine series:
    orthonormal DST-I coefficients, zero-padded (Lt > Ls) or truncated (Lt < Ls), inverse orthonormal DST-I.
    Prolongation is an isometry (Euclidean inner products preserved exactly in exact arithmetic)."""
    ms, mt = Ls - 1, Lt - 1

    def f(V):
        b = V.shape[1]
        G = V.T.reshape(b, ms, ms)
        if mutate == 'transpose':                       # control 2b: x/y swapped
            G = G.swapaxes(1, 2)
        C = _dst2(G, mutate)
        if mt >= ms:
            C = jnp.pad(C, ((0, 0), (0, mt - ms), (0, mt - ms)))
        else:
            C = C[:, :mt, :mt]
        return _dst2(C, mutate).reshape(b, mt * mt).T
    return jax.jit(f)


def resample(Phi_s, Ls, Lt, block=8, mutate=None):
    """Phi_s (n_s, r) host or device -> Phi_t (n_t, r) on the device (built block by block into a preallocated
    buffer, so the peak is Phi_t + one block)."""
    r = Phi_s.shape[1]
    f = resample_fn(Ls, Lt, mutate)
    out = jnp.zeros(((Lt - 1) ** 2, r), jnp.float64)
    put = jax.jit(lambda o, blk, a: jax.lax.dynamic_update_slice(o, blk, (0, a)), donate_argnums=(0,))
    for a in range(0, r, block):
        bsz = min(block, r - a)
        blk = f(jnp.asarray(np.asarray(Phi_s[:, a:a + bsz]) if not isinstance(Phi_s, jax.Array) else Phi_s[:, a:a + bsz]))
        if bsz != block:
            out = out.at[:, a:a + bsz].set(blk)
        else:
            out = put(out, blk, a)
    return sync(out)


@jax.jit
def _gram_block(P, a):
    B = jax.lax.dynamic_slice_in_dim(P, a, _GB, axis=0)
    return jnp.einsum('ni,nj->ij', B, B)


_GB = 1 << 18


def orth_dev(Phi):
    """max |Phi^T Phi - I|, accumulated over row blocks inside jit (an eager Phi.T materialises a transposed copy of
    the basis, 48 GB at 4096^2, smoke S1)."""
    n = Phi.shape[0]
    G = jnp.zeros((Phi.shape[1], Phi.shape[1]))
    a = 0
    while a + _GB <= n:
        G = G + _gram_block(Phi, a)
        a += _GB
    if a < n:
        B = Phi[a:]
        G = G + jnp.einsum('ni,nj->ij', B, B)
    G = sync(G)
    return float(jnp.max(jnp.abs(G - jnp.eye(G.shape[0]))))


def reorthonormalise(Phi):
    """Two Householder QR passes in mode order, sign fixed so diag(R) > 0 (nested prefixes preserved)."""
    Q = Phi
    for _ in range(2):
        Q, R = jnp.linalg.qr(Q)
        s = jnp.sign(jnp.diag(R))
        s = jnp.where(s == 0, 1., s)
        Q = Q * s[None, :]
    return sync(Q)


def resample_basis(Phi_s, Ls, Lt, mutate=None, qr=None):
    """Resampled basis + certificate (DESIGN A1.3). Restriction (Lt < Ls) is re-orthonormalised by QR (rank loss:
    min|R_ii| / max|R_ii| < 1e-10 -> not constructible); prolongation is an isometry and is not. Returns
    (Phi_t, R or None, info). qr=False forces no QR (control 2c)."""
    t0 = time.perf_counter()
    P = resample(Phi_s, Ls, Lt, mutate=mutate)
    do_qr = (Lt < Ls) if qr is None else qr
    dev_raw = orth_dev(P)
    R = None
    rank_ratio = None
    if do_qr:
        Q, R1 = jnp.linalg.qr(P)
        s1 = jnp.where(jnp.sign(jnp.diag(R1)) == 0, 1., jnp.sign(jnp.diag(R1)))
        Q, R1 = Q * s1[None, :], R1 * s1[:, None]
        Q2, R2 = jnp.linalg.qr(Q)
        s2 = jnp.where(jnp.sign(jnp.diag(R2)) == 0, 1., jnp.sign(jnp.diag(R2)))
        Q2, R2 = Q2 * s2[None, :], R2 * s2[:, None]
        R = np.asarray(R2 @ R1)                         # P = Q2 R (mode order kept, diag(R) > 0)
        d = np.abs(np.diag(R))
        rank_ratio = float(d.min() / d.max())
        P = sync(Q2)
        del Q, Q2
    dev = orth_dev(P)
    info = dict(Ls=Ls, Lt=Lt, qr=bool(do_qr), orth_raw=dev_raw, orth=dev, rank_ratio=rank_ratio,
                orth_ok=bool(np.isfinite(dev) and dev <= ORTH_TOL),     # rank acceptance is per prefix (rank_ok, A2.4)
                seconds=time.perf_counter() - t0, mutated=mutate)
    return P, R, info


def transform_codes(C_s, Ls, Lt, r, R=None):
    """Codes of the same physical fields (low-pass projected under restriction) in the rank-r resampled basis
    (DESIGN A1.3, A2.4): C_t = (Lt/Ls) C_s[:, :r] (prolongation); C_t = (Lt/Ls) C_s[:, :r] R[:r, :r]^T (restriction,
    Phi~ = Q R). Nodal values of a resampled vector are (Ls/Lt) x the physical field. C_s: (n_snap, >= r) rows."""
    f = Lt / Ls
    C = np.asarray(C_s)[:, :r]
    return f * (C if R is None else C @ np.asarray(R)[:r, :r].T)


def rank_ok(R, r, cond_max=1e10):
    s = np.linalg.svd(np.asarray(R)[:r, :r], compute_uv=False)
    return bool(s[-1] > 0 and s[0] / s[-1] <= cond_max), float(s[0] / s[-1]) if s[-1] > 0 else float('inf')


def node_consistency(Phi_s, Ls, Phi_t, Lt, cols=None):
    """Control 2: on the nodes of the coarser mesh (each is also a node of the finer, nested mesh),
    (Lt/Ls) Phi_t == Phi_s. Exact for prolongation (the sine interpolant reproduces the source nodes); for
    restriction it measures the low-pass difference (reported only). Relative Frobenius difference."""
    Lc, Lf = min(Ls, Lt), max(Ls, Lt)
    nc = (Lc - 1) ** 2
    fine = jnp.asarray(map_nodes(np.arange(nc), Lc, Lf))
    if Lt >= Ls:
        a, b = jnp.asarray(Phi_t)[fine], jnp.asarray(Phi_s)
    else:
        a, b = jnp.asarray(Phi_t), jnp.asarray(Phi_s)[fine]
    if cols is not None:
        a, b = a[:, :cols], b[:, :cols]
    return float(jnp.linalg.norm((Lt / Ls) * a - b) / jnp.linalg.norm(b))


def roundtrip_error(Phi_s, Ls, Lt, cols=8):
    """Control 2: prolong Ls -> Lt (> Ls) and restrict back; max abs difference on the first `cols` modes."""
    P = resample(np.asarray(Phi_s[:, :cols]), Ls, Lt)
    B = resample(P, Lt, Ls)
    return float(jnp.max(jnp.abs(B - jnp.asarray(np.asarray(Phi_s[:, :cols])))))


# ------------------------------------------------------------------------------------------ rule reuse ----

def map_nodes(nodes_s, Ls, Lt, mutate=False):
    """Flat interior node indices on mesh Ls -> the coincident nodes on the nested finer mesh Lt (node (i, j) at
    x = (i+1)/Ls is node ((i+1) Lt/Ls - 1, ...) on Lt). mutate: control 3 (indices used unmapped)."""
    assert Lt % Ls == 0 and Lt >= Ls, (Ls, Lt)
    nodes_s = np.asarray(nodes_s, np.int64)
    if mutate:
        return nodes_s.copy()
    k = Lt // Ls
    i, j = np.unravel_index(nodes_s, (Ls - 1, Ls - 1))
    return np.ravel_multi_index(((i + 1) * k - 1, (j + 1) * k - 1), (Lt - 1, Lt - 1))


# ------------------------------------------------------------------------------- low-memory ECSW (large L) ----

def gather_stencil(Phi, idx, r):
    """(k, 5, r) stencil rows of Phi[:, :r] with zero rows for wall neighbours (idx == n), without the
    (n + 1, r) concatenated copy that hrcore.stencil_rows makes (that copy does not fit at 4096^2)."""
    n = Phi.shape[0]
    idx = jnp.asarray(idx)
    ok = idx < n
    rows = jnp.take(Phi, jnp.minimum(idx, n - 1), axis=0)[..., :r]
    return rows * ok[..., None]


_gather = jax.jit(gather_stencil, static_argnums=(2,))


def ecsw_data_lowmem(Phi, r, nodes, weights, L):
    """build.ecsw_data equivalent (Pst, sw) from the full-rank device basis without slicing Phi."""
    idx = H.stencil_index(nodes, D, L)
    return sync(dict(Pst=_gather(Phi, jnp.asarray(idx), r), sw=jnp.sqrt(jnp.asarray(weights)))), \
        dict(sample=int(len(nodes)), stencil_nodes=int(len(np.unique(idx[idx < (L - 1) ** D]))))


def make_block_eval(L, mutation=None):
    """Per node block E: residual r_E and Jacobian rows J_E of the unreduced LSPG step (the 'none' arm's rows),
    evaluated by the ECSW arm's stencil algebra with unit weights. mutation 'drop_adv_jac': control 5."""
    sc_e, rJ_e = RM.make_arm('ecsw', D, L, mutation=mutation)

    @jax.jit
    def ev(Pst, c, cn, cm, nu, co, dt):
        data = dict(Pst=Pst, sw=jnp.ones(Pst.shape[0], Pst.dtype))
        const = sc_e(cn, cm, nu, co, dt, data)
        r, J = rJ_e(c, const, nu, co, dt, data)
        return r, J
    return ev


def ecsw_contrib_blocked(states, Phi, r, L, block=1 << 17, seed=BD.ECSW_SEED_CAND, n_cand=BD.CAND,
                         cand_nodes=None, mutation=None, log=print):
    """Same output as build.ecsw_contrib(states, Phi[:, :r], 2, L) (G, b, cand, info), computed in node blocks
    from the device basis Phi (n, >= r) so that no (n, r) Jacobian, Laplacian or derivative copy of the basis is
    formed. Candidate draw, block scaling and the floor are build.ecsw_contrib's code path, line for line."""
    n = (L - 1) ** D
    ev = make_block_eval(L, mutation)
    be = jnp.asarray(H.BE_CO)
    proj = jax.jit(lambda P, u: (u @ P)[:r])
    rows = []
    for s in states:
        sc, dt = s['option']
        a0, a1, a2, b0, b1, ns = H.SCHEMES[sc]
        co = be if s['k'] <= ns else jnp.asarray([a0, a1, a2, b0, b1])
        rows.append((s, co, dt))
    blocks = [(a, min(a + block, n)) for a in range(0, n, block)]
    bidx = [jnp.asarray(H.stencil_index(np.arange(a, b_), D, L)) for a, b_ in blocks]
    imp = np.zeros(n)
    bvec = []
    codes = []
    t0 = time.perf_counter()
    for si, (s, co, dt) in enumerate(rows):
        c, cn, cm = (proj(Phi, jnp.asarray(s[k])) for k in ('u', 'un', 'um'))
        codes.append((c, cn, cm))
        bacc = jnp.zeros(r)
        for (a, b_), ix in zip(blocks, bidx):
            Pst = _gather(Phi, ix, r)
            rr, J = ev(Pst, c, cn, cm, jnp.float64(s['nu']), co, jnp.float64(dt))
            bacc = bacc + J.T @ rr
            if cand_nodes is None and n > n_cand:
                imp[a:b_] += np.asarray(jnp.abs(rr) * jnp.linalg.norm(J, axis=1))
            del Pst, rr, J
        bvec.append(np.asarray(bacc))
        if log and (si % 30 == 0 or si == len(rows) - 1):
            log(f'   blocked contrib state {si + 1}/{len(rows)} [{time.perf_counter() - t0:.0f}s]')
    if cand_nodes is None:
        if n <= n_cand:
            cand_nodes = np.arange(n)
        else:
            rng = np.random.default_rng(seed)
            half = n_cand // 2
            uni = rng.choice(n, half, replace=False)
            rest = np.setdiff1d(np.arange(n), uni)
            pr = imp[rest] / imp[rest].sum()
            k_imp = min(n_cand - half, int(np.sum(pr > 0)))
            impn = rng.choice(rest, k_imp, replace=False, p=pr)
            cand_nodes = np.sort(np.concatenate([uni, impn]))
    cidx = jnp.asarray(H.stencil_index(cand_nodes, D, L))
    Pc = _gather(Phi, cidx, r)
    nc = len(cand_nodes)
    Gh = np.empty((len(rows) * r, nc))
    bh = np.empty(len(rows) * r)
    nb = np.empty(len(rows))
    for i, ((s, co, dt), (c, cn, cm)) in enumerate(zip(rows, codes)):
        rr, J = ev(Pc, c, cn, cm, jnp.float64(s['nu']), co, jnp.float64(dt))
        Gh[i * r:(i + 1) * r] = np.asarray((J * rr[:, None]).T)
        bh[i * r:(i + 1) * r] = bvec[i]
        nb[i] = float(np.linalg.norm(bvec[i]))
    floor = max(1e-8 * float(np.median(nb)), 1e-14 * float(np.max(nb)), 1e-300)
    keep = nb >= floor
    if not np.any(keep):
        raise RuntimeError('ECSW construction failure: every training target is below the floor')
    sc_ = 1. / np.maximum(nb, floor)
    rowsel = np.repeat(keep, r)
    Gh *= np.repeat(sc_, r)[:, None]
    bh *= np.repeat(sc_, r)
    if not keep.all():
        Gh, bh = Gh[rowsel], bh[rowsel]
    G = sync(jnp.asarray(Gh))
    b = jnp.asarray(bh)
    info = dict(n_cand=int(nc), dropped_blocks=int(np.sum(~keep)), block_norms=nb.tolist(), keep=keep.tolist(),
                floor=floor, blocked=True, block=int(block), seconds=time.perf_counter() - t0)
    return G, b, cand_nodes, info


# ------------------------------------------------------------------------------------ a competitor cell ----

class Cell(PL.Cell):
    """jcp-hr-baselines Cell on a given basis, without the (n, r) Laplacian / derivative copies (only the ECSW arm
    is run). Queries, certifier and lattice restriction are the parent's."""

    def __init__(self, L, Phi, trust, log=print, timer=None):
        super().__init__(D, L, log=log, timer=timer)
        self.Phi, self.trust, self.gen = Phi, dict(trust), 'central'
        self.constructible_r = set(int(k) for k in trust)

    def phi_r(self, r):
        """The r-column basis for init/decode. A prefix of the device basis is a copy (one (n, r) buffer), so only
        the most recent rank is cached; the caller releases arm data before moving to another rank."""
        if self.Phi.shape[1] == r:
            return self.Phi
        if getattr(self, '_slice_r', None) != r:
            self._slice = None
            gc.collect()
            self._slice = sync(self.Phi[:, :r])
            self._slice_r = r
        return self._slice

    def ecsw_data(self, r, nodes, weights):
        """(data for rom.make_query('ecsw'), info). data['Phi'] is the r-column basis used by init and decode."""
        dat, info = ecsw_data_lowmem(self.Phi, r, nodes, weights, self.L)
        return dict(dat, Phi=self.phi_r(r)), info


class budget_override:
    """Control 4: queries built inside this context use an LM budget of `budget` iterations (rom.make_common is
    looked up by make_evolve at build time; the original is restored on exit)."""

    def __init__(self, budget):
        self.budget = budget

    def __enter__(self):
        self.orig = RM.make_common
        RM.make_common = lambda rJ, trust: self.orig(rJ, trust, budget=self.budget)

    def __exit__(self, *a):
        RM.make_common = self.orig


class unsolved_override:
    """Control 4 fallback (DESIGN A6.2): every step returns half of its starting candidate, unsolved."""

    def __enter__(self):
        self.orig = RM.make_common

        def mk(rJ, trust):
            def solve(c0, args, tol, gtol, lam0):
                return 0.5 * c0, jnp.int32(0), jnp.int32(2), lam0
            return solve
        RM.make_common = mk

    def __exit__(self, *a):
        RM.make_common = self.orig


def assert_case_identity(fams, refs, case, desc):
    """Control 6 / A2.6: the run's descriptor row must be the cohort table row of the case it is scored against
    (score.Refs already ties each reference's t0 to the IC built from that row)."""
    coh, j = case
    assert np.array_equal(np.asarray(desc, float), np.asarray(fams[coh].D[j], float)), ('case identity', case)
    assert j in refs[coh].f, ('reference not loaded', case)


def e0_of(ref, fields_lat, j):
    """t0 output vs the reference t0 of case j, relative to n0 (reported identity diagnostic)."""
    F = np.asarray(fields_lat, float)
    return float(np.linalg.norm((F[0] - ref.f[j][0]).ravel()) / ref.n0(j))


def run_config(cell, r, data, option, cases, fams, refs, tag=None):
    """One competitor configuration on `cases` [(cohort, j)]: prod run (timed, certified with a fresh Jacobian at
    every step), tight anchor on every case, tighter on the first two; E0 scores; Q1 (pipeline.q1_config).
    fams / refs: per-cohort hrcore.Family / score.Refs. Returns (summary, arrays)."""
    recs, tight, tighter, n0 = {}, {}, {}, {}
    cases = sorted(cases)
    cohs = sorted({c[0] for c in cases})
    first2 = {coh: [c for c in cases if c[0] == coh][:2] for coh in cohs}        # each cohort's first two (3.4)
    anchor_set = [c for coh in cohs for c in first2[coh]]
    for c in cases:
        coh, j = c
        u0, nu = fams[coh].ic(cell.L, j), fams[coh].nu(j)
        recs[c] = cell.run('ecsw', r, data, 'common', option, 'prod', u0, nu)
        recs[c]['desc'] = np.asarray(fams[coh].D[j], float)
        tight[c] = cell.run('ecsw', r, data, 'native', option, 'tight', u0, nu)     # native GN anchors (hr-baselines)
        n0[c] = refs[coh].n0(j)
        if c in anchor_set:
            tighter[c] = cell.run('ecsw', r, data, 'native', option, 'tighter', u0, nu)
    per = {}
    for coh in cohs:
        cc = [c for c in cases if c[0] == coh]
        per[coh] = PL.q1_config({c: recs[c] for c in cc}, {c: tight[c] for c in cc},
                                {c: tighter[c] for c in first2[coh]}, {c: n0[c] for c in cc}, first2[coh], cc)
    ns = sum(recs[c]['cert']['nonstat'] for c in cases)
    st = sum(recs[c]['cert']['steps'] for c in cases)
    dist = {c: v for coh in cohs for c, v in per[coh].get('dist', {}).items()}
    q1 = dict(per_cohort={coh: {k: v for k, v in per[coh].items() if k not in ('dist', 'anchor')} for coh in cohs},
              all_valid=all(recs[c]['cert']['valid'] for c in cases), nonstat=ns, steps=st,
              max_dist=max(dist.values()) if dist else float('inf'),
              max_anchor=max(per[coh].get('max_anchor', float('inf')) for coh in cohs), dist=dist)
    # pooled stationarity (COMMON-BRIEF 3); every other predicate per cohort, as pipeline.q1_config defines it
    q1['q1'] = bool(st > 0 and ns <= PL.Q1_NONSTAT * st and q1['all_valid'] and all(
        per[coh].get('anchors_ok') and per[coh].get('max_dist', 1.) <= PL.Q1_DIST and
        per[coh].get('max_anchor', 1.) <= PL.Q1_ANCHOR and per[coh].get('reason') != 'incomplete anchors'
        for coh in cohs))
    rows = []
    for c in cases:
        coh, j = c
        assert_case_identity(fams, refs, c, recs[c]['desc'])
        sc = refs[coh].score_safe(recs[c]['fields_lat'], j)
        ce = recs[c]['cert']
        rows.append(dict(cohort=coh, case=int(j), e=sc['e'], eps_ref=sc['eps_ref'], eligible=sc['eligible'],
                         seconds=recs[c]['seconds'], valid=bool(ce['valid']), nonstat=int(ce['nonstat']),
                         steps=int(ce['steps']), dist_tight=float(q1.get('dist', {}).get(c, np.nan)),
                         it_sum=int(np.sum(np.maximum(recs[c]['it'], 0))), e0=e0_of(refs[coh], recs[c]['fields_lat'], j),
                         desc=recs[c]['desc'].tolist()))
    q1s = {k: v for k, v in q1.items() if k != 'dist'}
    arrays = {f'W|{c[0]}|{c[1]}': np.asarray(recs[c]['W']) for c in cases}
    arrays.update({f'C|{c[0]}|{c[1]}': np.asarray(recs[c]['C']) for c in cases})
    arrays.update({f'it|{c[0]}|{c[1]}': np.asarray(recs[c]['it']) for c in cases})
    arrays.update({f'reason|{c[0]}|{c[1]}': np.asarray(recs[c]['reason']) for c in cases})
    arrays.update({f'F|{c[0]}|{c[1]}': np.asarray(recs[c]['fields_lat']) for c in cases if c[0] == 'dev6'})
    return dict(tag=tag, r=int(r), option=list(option), q1=q1s, rows=rows), arrays


def summarise(rows):
    """worst / median over eligible cases (DESIGN 2)."""
    e = np.array([x['e'] for x in rows if x['eligible']], float)
    if len(e) == 0 or not np.all(np.isfinite(e)):
        return dict(worst=float('nan'), median=float('nan'), n=int(len(e)))
    return dict(worst=float(e.max()), median=float(np.median(e)), n=int(len(e)),
                eps_star=float(max(x['eps_ref'] for x in rows if x['eligible'])))


# ------------------------------------------------------------------------------------- compile measurement ----

_EVENTS = []
_INSTALLED = [False]


def _install_listener():
    if not _INSTALLED[0]:
        import jax.monitoring as jm
        jm.register_event_duration_secs_listener(lambda name, secs, **kw: _EVENTS.append((str(name), float(secs))))
        _INSTALLED[0] = True


def aot_compile(jitted, *args):
    """DESIGN A4.3: wall seconds of jitted.lower(*args).compile() on a jit object never compiled before, with JAX's
    own backend-compile events recorded during the call (none recorded -> 'cache hit suspected')."""
    _install_listener()
    n0 = len(_EVENTS)
    t0 = time.perf_counter()
    jitted.lower(*args).compile()
    sec = time.perf_counter() - t0
    ev = [e for e in _EVENTS[n0:] if 'backend_compile' in e[0]]
    return dict(seconds=sec, backend_events=len(ev), backend_seconds=float(sum(e[1] for e in ev)),
                cache_hit_suspected=len(ev) == 0)
