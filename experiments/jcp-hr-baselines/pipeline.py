"""jcp-hr-baselines cell pipeline shared by smoke.py (phase I) and the phase-II drivers: offline construction of one
(dimension, mesh) cell, configuration runs with timing, untimed Q1 certification, anchors and per-configuration Q1.

Configuration ID (A2.17): kind|r|hr-param|scheme|dt|solver|generator, hr-param = tau (ecsw), p (qdeim) or '-'.
"""
from __future__ import annotations

import json
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import hrcore as H
import rom as RM
import build as BD

Q1_DIST, Q1_ANCHOR, Q1_NONSTAT = 2.5e-4, 2.5e-5, 0.01


_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    """Keep the GPU busy for `seconds` before a timed call (combined lane's burn)."""
    a = jax.block_until_ready(_BURN(jnp.ones((512, 512), jnp.float64) * .01))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def cfg_id(kind, r, hrp, scheme, dt, solver, gen='central'):
    return f'{kind}|{r}|{hrp}|{scheme}|{dt:.8g}|{solver}|{gen}'


def sync(x):
    return jax.block_until_ready(x)


def device_peak():
    try:
        return int((jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use', -1))
    except Exception:  # noqa: BLE001
        return -1


def host_rss():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # noqa: BLE001
        return -1


class Cell:
    """One (d, L) cell: basis, trust radii, advection basis, Q-DEIM and ECSW artifacts, arm data, compiled queries."""

    def __init__(self, d, L, log=print, timer=None):
        self.d, self.L, self.log = d, L, log
        self.n = (L - 1) ** d
        self.T = timer or BD.Timer()
        self.info = dict(d=d, L=L, n=self.n)
        self.qcache, self.ccache, self.piv = {}, {}, {}
        self.ecsw_rec = {}

    # --------------------------------------------------------------------------------------- offline -----
    def build_basis(self, fam_train, rows, rmax, ranks, gen='central'):
        """Snapshots (gen 'central' = CN-R central FOM; 'upwind' = the banks' generator, upwind BE), POD, trust."""
        sch_gen = ('CNR', 'central') if gen == 'central' else ('BE', 'upwind')
        with self.T(f'snapshots[{gen}]'):
            S, sinfo = BD.snapshots(self.d, self.L, fam_train, rows, scheme=sch_gen[0], adv=sch_gen[1], log=self.log)
        assert rmax <= S.shape[0], f'rank {rmax} exceeds the snapshot count {S.shape[0]} (not constructible, A6.2)'
        with self.T(f'pod[{gen}]'):
            Phi, sig, dev = BD.pod(S, rmax)
        rl = [r for r in ranks if r <= rmax]
        if not (np.all(np.isfinite(np.asarray(Phi[:1]))) and np.isfinite(dev)):   # A6.2: non-finite factor -> fallback
            dev = np.inf
        with self.T(f'pod_check[{gen}]'):
            pe, C = BD.projection_errors(S, Phi, rl, sig)
        passed = basis_certified(pe, dev, rl)
        fallback = None
        if not all(passed.values()):          # A5.2: blocked randomised SVD; the tail always comes from the Gram
            Phi_host = np.asarray(Phi)             # A14.2: only one candidate basis on the device at a time
            del Phi
            with self.T(f'pod_fallback[{gen}]'):
                Phi2, sr, dev2 = BD.pod_rsvd(S, rmax)
                pe2, C2 = BD.projection_errors(S, Phi2, rl, sig)
            passed2 = basis_certified(pe2, dev2, rl)
            fallback = dict(passed=passed2, proj=pe2)
            if sum(passed2.values()) > sum(passed.values()):          # A6.1: one basis per cell, the one certifying more
                Phi, pe, C, dev, passed = Phi2, pe2, C2, dev2, passed2
                fallback['used'] = True
            else:
                del Phi2
                Phi = jax.block_until_ready(jnp.asarray(Phi_host))
            del Phi_host
        rho = int(np.sum(sig > 1e-10 * sig[0]))          # reported only (A5.2)
        trust = BD.trust_radii(C, rl, self.d)
        gate = dict(orthonormality=dev, orth_ok=dev <= 1e-10, numerical_rank_reported=rho, fallback=fallback,
                    certificate=passed, optimality_ok=all(passed.values()),
                    constructible=[r for r in rl if passed[r]])
        self.constructible_r = set(gate['constructible']) if dev <= 1e-10 and np.isfinite(dev) else set()
        self.info[f'basis[{gen}]'] = dict(snapshots=sinfo, sigma=sig[:max(ranks) + 1].tolist(), proj=pe, trust=trust,
                                          gate=gate, n_snap=int(S.shape[0]))
        self.log(f'   basis[{gen}]: n_snap {S.shape[0]} rank {rho} orth {dev:.1e} optimal {gate["optimality_ok"]}')
        return S, Phi, sig, trust

    def set_basis(self, Phi, trust, gen='central'):
        self.Phi, self.trust, self.gen = Phi, trust, gen
        with self.T('lap_dsum_cols'):
            self.LPhi = sync(H.lap_cols(Phi, self.d, self.L))
            self.DPhi = sync(H.dsum_cols(Phi, self.d, self.L))

    def build_adv(self, S, pmax, p_list):
        """Advection POD (overwrites S in place) -> U (n, pmax) on the device; optimality check at the registered p and
        the A4.7 fallback."""
        with self.T('adv_snapshots'):
            BD.advection_in_place(S, self.d, self.L)
        assert pmax <= S.shape[0], f'p {pmax} exceeds the snapshot count {S.shape[0]} (not constructible, A6.2)'
        with self.T('adv_pod'):
            U, sigN, devN = BD.pod(S, pmax)
        pl = sorted({p for p in p_list if p <= pmax})
        with self.T('adv_pod_check'):
            pe, _ = BD.projection_errors(S, U, pl, sigN)
        passed = basis_certified(pe, devN, pl)
        fallback = None
        if not all(passed.values()):
            U_host = np.asarray(U)                 # A14.2: only one candidate basis on the device at a time
            del U
            with self.T('adv_pod_fallback'):
                U2, sr, devN2 = BD.pod_rsvd(S, pmax)
                pe2, _ = BD.projection_errors(S, U2, pl, sigN)
            passed2 = basis_certified(pe2, devN2, pl)
            fallback = dict(passed=passed2)
            if sum(passed2.values()) > sum(passed.values()):
                U, pe, devN, passed = U2, pe2, devN2, passed2
                fallback['used'] = True
            else:
                del U2
                U = jax.block_until_ready(jnp.asarray(U_host))
            del U_host
        ok = all(passed.values())
        rhoN = int(np.sum(sigN > 1e-10 * sigN[0]))
        self.U, self.sigN = U, sigN
        self.constructible_p = {p for p in pl if passed[p]} if (devN <= 1e-10 and np.isfinite(devN)) else set()
        self.info['adv'] = dict(pmax=pmax, numerical_rank=rhoN, orthonormality=devN, sigma=sigN[:pmax + 1].tolist(),
                                proj=pe, optimality_ok=ok, fallback=fallback, certificate=passed,
                                constructible=[p for p in pl if passed[p]])
        self.log(f'   advection POD: p_max {pmax} numerical rank {rhoN} orth {devN:.1e}')
        return rhoN

    def qdeim(self, r, p, allow_uncertified_p=False):
        """allow_uncertified_p: only for the N-under ladder's small p (outside the registered p list)."""
        if r not in self.constructible_r:
            raise ConstructionError(f'rank {r} not constructible')
        if p not in getattr(self, 'constructible_p', set()) and not allow_uncertified_p:
            raise ConstructionError(f'advection rank p={p} not certified')
        if p not in self.piv:
            with self.T('qdeim_pivots'):
                self.piv[p] = BD.qdeim_points(self.U[:, :p])
        with self.T('qdeim_qr'):
            data, info = BD.qdeim_data(self.Phi[:, :r], self.LPhi[:, :r], self.U[:, :p], self.piv[p], self.d, self.L)
        if not info['solve_ok']:
            raise ConstructionError(f'DEIM solve backward error {info["solve_backward_error"]:.2e} > 1e-12 (r={r}, p={p})')
        return dict(data, Phi=self.Phi[:, :r]), info

    def ecsw_states(self, fam_train):
        with self.T('ecsw_states'):
            self.states, info = BD.ecsw_states(self.d, self.L, fam_train, self.Phi)
        self.info['ecsw_states'] = info
        return info

    def ecsw(self, r, kappas=BD.ECSW_KAPPA, nnls_log=None):
        """A15.1: weights at support targets kappa * r (keys 'k<kappa>'), tau crossings reported."""
        with self.T('ecsw_contrib'):
            G, b, cand, cinfo = BD.ecsw_contrib(self.states, self.Phi[:, :r], self.d, self.L)
            sync(G)
        with self.T('ecsw_nnls'):
            rec, summ = BD.ecsw_weights(G, b, cand, r, kappas, log=nnls_log)
        del G, b
        self.ecsw_rec[r] = {f'k{k}': v for k, v in rec.items()}
        self.info.setdefault('ecsw', {})[r] = dict(contrib=dict(cinfo, block_norms=None), paths=summ,
                                                  supports={f'k{k}': (None if v is None else dict(size=v['size'],
                                                            relres=v['relres'])) for k, v in rec.items()})
        return self.ecsw_rec[r]

    def arm_data(self, kind, r, hrp, allow_uncertified_p=False):
        if r not in self.constructible_r:
            raise ConstructionError(f'rank {r} not constructible')
        if kind == 'none':
            return dict(Phi=self.Phi[:, :r], LPhi=self.LPhi[:, :r], DPhi=self.DPhi[:, :r]), {}
        if kind == 'ecsw':
            rec = self.ecsw_rec[r][hrp] if not isinstance(hrp, tuple) else hrp[1]
            if rec is None:
                raise ConstructionError(f'ECSW support target {hrp} not reached at r={r}')
            dat, info = BD.ecsw_data(self.Phi[:, :r], rec['nodes'], rec['weights'], self.d, self.L)
            return dict(dat, Phi=self.Phi[:, :r]), info
        return self.qdeim(r, hrp, allow_uncertified_p=allow_uncertified_p)

    # ------------------------------------------------------------------------------------------ online -----
    def query(self, kind, r, solver, mutation=None):
        key = (kind, r, solver, mutation)
        if key not in self.qcache:
            self.qcache[key] = RM.make_query(kind, self.d, self.L, solver, trust=self.trust.get(r), mutation=mutation)
        return self.qcache[key]

    def certifier(self, kind):
        if kind not in self.ccache:
            self.ccache[kind] = RM.make_certifier(kind, self.d, self.L, H.NMAX[self.d])
        return self.ccache[kind]

    def run(self, kind, r, data, solver, option, level, u0, nu, mutation=None, certify=True, burn_s=0.):
        """One timed query + (untimed) certification. Returns a record with lattice fields and Q1 evidence.
        burn_s > 0: GPU burn before the timed call (screening timings, code audit 1 item 8)."""
        q = self.query(kind, r, solver, mutation)
        # A8.1: the native GN's production stopping tolerance is gamma <= 1e-6 ('prod' requests map to it)
        sch = RM.schedule(self.d, option[0], option[1], 'prod_native' if (solver == 'native' and level == 'prod') else level)
        u0 = jnp.asarray(u0)
        if burn_s > 0:
            burn(burn_s)
        t0 = time.perf_counter()
        out = q(u0, jnp.float64(nu), data, sch)
        sync(out['fields'])
        t = time.perf_counter() - t0
        rec = dict(seconds=t, W=np.asarray(out['W']), fields_lat=self.lattice(out['fields']))
        if certify:
            gam = self.certifier(kind)(out['C'], jnp.float64(nu), data, sch)
            rec['cert'] = RM.certify(out, gam, sch, level if level in ('tight', 'tighter') else 'prod')
            rec['gamma'] = np.asarray(gam)[:int(sch['steps'])]
        rec['C'] = np.asarray(out['C'])[:int(sch['steps']) + 1]
        rec['reason'] = np.asarray(out['stats']['reason'])
        rec['lam'] = np.asarray(out['stats']['lam'])
        rec['it'] = np.asarray(out['stats']['it'])
        rec['scale'] = float(out['scale'])
        return rec

    def lattice(self, fields):
        if not hasattr(self, '_lat_idx'):
            self._lat_idx = jnp.asarray(H.lattice_nodes(self.d, self.L))
        v = np.asarray(jnp.asarray(fields)[:, self._lat_idx])          # rows selected on the device (code audit 1, 18)
        return H.to_lattice(v, self.d)


class ConstructionError(RuntimeError):
    """A registered configuration cannot be constructed (POD rank not certified, DEIM solve inaccurate, ...)."""


def basis_certified(pe, dev, ranks):
    """Per-rank certificate of one candidate basis, including orthonormality and finiteness (code audit 2, item 13)."""
    ok_basis = bool(np.isfinite(dev) and dev <= 1e-10)
    return {r: bool(ok_basis and certificate_ok(pe[r])) for r in ranks}


def certificate_ok(v):
    """A5.2 / A6.1 POD certificate. tau^2 = (||S||^2 - sum_{i<=r} lambda_i) / ||S||^2 from the Gram eigenvalues:
    a negative radicand within -1e-12 is clipped to 0, a larger negative one fails (stable recomputation needed).
    Saturated regime tau < 1e-6: pass iff e <= 1e-6. Otherwise pass iff |e - tau| <= 0.01 tau + 1e-6."""
    t2, e = v['pod_tail_sq'], v['frob_rel']
    if not (np.isfinite(t2) and np.isfinite(e)) or t2 < -1e-12:
        return False
    tau = float(np.sqrt(max(t2, 0.)))
    if tau < 1e-6:
        return bool(e <= 1e-6)
    return bool(abs(e - tau) <= 0.01 * tau + 1e-6)


def q1_config(records, tight, tighter, n0, subset, expected):
    """Per-configuration Q1 (A1.7, A2.7) on a cohort. `expected` = the cohort's registered case identities; records /
    tight must cover exactly them; tighter exactly the prescribed `subset` (the cohort's first two cases); n0 positive
    and finite for each. Fail-closed."""
    import score as SC
    cases = sorted(expected)
    complete = (sorted(records) == cases and sorted(tight) == cases and sorted(tighter) == sorted(subset)
                and list(subset) == cases[:2] and all(c in n0 and np.isfinite(n0[c]) and n0[c] > 0 for c in cases))
    if not complete:
        return dict(q1=False, reason='incomplete anchors', cases=cases, tight=sorted(tight), tighter=sorted(tighter))
    allv = all(r['cert']['valid'] for r in records.values())
    ns = sum(r['cert']['nonstat'] for r in records.values())
    st = sum(r['cert']['steps'] for r in records.values())
    anchors_ok = all(tight[c]['cert']['anchor_ok'] for c in cases) and all(tighter[c]['cert']['anchor_ok'] for c in subset)
    dist = {c: SC.lattice_distance(records[c]['fields_lat'], tight[c]['fields_lat'], n0[c]) for c in cases}
    dtt = {c: SC.lattice_distance(tight[c]['fields_lat'], tighter[c]['fields_lat'], n0[c]) for c in subset}
    finite = all(np.isfinite(v) for v in list(dist.values()) + list(dtt.values()))
    ok = bool(finite and allv and st > 0 and ns <= Q1_NONSTAT * st and anchors_ok
              and all(v <= Q1_DIST for v in dist.values()) and all(v <= Q1_ANCHOR for v in dtt.values()))
    return dict(q1=ok, all_valid=allv, nonstat=ns, steps=st, anchors_ok=anchors_ok, max_dist=max(dist.values()),
                max_anchor=max(dtt.values()), dist=dist, anchor=dtt, subset=list(subset))


def dump(path, obj):
    def conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, tuple):
            return list(x)
        return str(x)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=conv)
