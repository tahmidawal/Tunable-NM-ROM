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
        with self.T(f'pod[{gen}]'):
            Phi, sig, dev = BD.pod(S, rmax)
        with self.T(f'pod_check[{gen}]'):
            pe, C = BD.projection_errors(S, Phi, [r for r in ranks if r <= rmax], sig)
        fallback = False
        if not all(abs(v['frob_rel'] - v['pod_tail']) <= 0.01 * v['pod_tail'] + 1e-6 for v in pe.values()):
            with self.T(f'pod_fallback[{gen}]'):            # A4.7: blocked randomised SVD, then the check again
                Phi, sr, dev = BD.pod_rsvd(S, rmax)
                sig = np.concatenate([sr, np.zeros(max(0, len(sig) - len(sr)))])
                sig = np.sqrt(np.maximum(sig ** 2, 0.))
                pe, C = BD.projection_errors(S, Phi, [r for r in ranks if r <= rmax], sig)
                fallback = True
        rho = int(np.sum(sig > 1e-10 * sig[0]))
        trust = BD.trust_radii(C, [r for r in ranks if r <= rmax], self.d)
        gate = dict(orthonormality=dev, orth_ok=dev <= 1e-10, numerical_rank=rho, fallback=fallback,
                    optimality_ok=all(abs(v['frob_rel'] - v['pod_tail']) <= 0.01 * v['pod_tail'] + 1e-6
                                      for v in pe.values()),
                    constructible=[r for r in ranks if r <= min(rho, rmax)])
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
        with self.T('adv_pod'):
            U, sigN, devN = BD.pod(S, pmax)
        pl = sorted({p for p in p_list if p <= pmax})
        with self.T('adv_pod_check'):
            pe, _ = BD.projection_errors(S, U, pl, sigN)
        ok = all(abs(v['frob_rel'] - v['pod_tail']) <= 0.01 * v['pod_tail'] + 1e-6 for v in pe.values())
        fallback = False
        if not ok:
            with self.T('adv_pod_fallback'):
                U, sr, devN = BD.pod_rsvd(S, pmax)
                sigN = np.concatenate([sr, np.zeros(max(0, len(sigN) - len(sr)))])
                pe, _ = BD.projection_errors(S, U, pl, sigN)
                ok = all(abs(v['frob_rel'] - v['pod_tail']) <= 0.01 * v['pod_tail'] + 1e-6 for v in pe.values())
                fallback = True
        rhoN = int(np.sum(sigN > 1e-10 * sigN[0]))
        self.U, self.sigN = U, sigN
        self.info['adv'] = dict(pmax=pmax, numerical_rank=rhoN, orthonormality=devN, sigma=sigN[:pmax + 1].tolist(),
                                proj=pe, optimality_ok=ok, fallback=fallback,
                                constructible=[p for p in pl if p <= rhoN and ok])
        self.log(f'   advection POD: p_max {pmax} numerical rank {rhoN} orth {devN:.1e}')
        return rhoN

    def qdeim(self, r, p):
        if p not in self.piv:
            with self.T('qdeim_pivots'):
                self.piv[p] = BD.qdeim_points(self.U[:, :p])
        with self.T('qdeim_qr'):
            data, info = BD.qdeim_data(self.Phi[:, :r], self.LPhi[:, :r], self.U[:, :p], self.piv[p], self.d, self.L)
        return dict(data, Phi=self.Phi[:, :r]), info

    def ecsw_states(self, fam_train):
        with self.T('ecsw_states'):
            self.states, info = BD.ecsw_states(self.d, self.L, fam_train, self.Phi)
        self.info['ecsw_states'] = info
        return info

    def ecsw(self, r, taus=BD.ECSW_TAUS, nnls_log=None):
        with self.T('ecsw_contrib'):
            G, b, cand, cinfo = BD.ecsw_contrib(self.states, self.Phi[:, :r], self.d, self.L)
            sync(G)
        with self.T('ecsw_nnls'):
            rec, summ = BD.ecsw_weights(G, b, cand, taus, log=nnls_log)
        del G, b
        self.ecsw_rec[r] = rec
        self.info.setdefault('ecsw', {})[r] = dict(contrib=dict(cinfo, block_norms=None), paths=summ,
                                                  taus={str(t): (None if v is None else dict(size=v['size'],
                                                        relres=v['relres'], retry_cap=v['retry_cap'])) for t, v in rec.items()})
        return rec

    def arm_data(self, kind, r, hrp):
        if kind == 'none':
            return dict(Phi=self.Phi[:, :r], LPhi=self.LPhi[:, :r], DPhi=self.DPhi[:, :r]), {}
        if kind == 'ecsw':
            rec = self.ecsw_rec[r][hrp] if not isinstance(hrp, tuple) else hrp[1]
            dat, info = BD.ecsw_data(self.Phi[:, :r], rec['nodes'], rec['weights'], self.d, self.L)
            return dict(dat, Phi=self.Phi[:, :r]), info
        return self.qdeim(r, hrp)

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

    def run(self, kind, r, data, solver, option, level, u0, nu, mutation=None, certify=True):
        """One timed query + (untimed) certification. Returns a record with lattice fields and Q1 evidence."""
        q = self.query(kind, r, solver, mutation)
        sch = RM.schedule(self.d, option[0], option[1], level)
        u0 = jnp.asarray(u0)
        t0 = time.perf_counter()
        out = q(u0, jnp.float64(nu), data, sch)
        sync(out['fields'])
        t = time.perf_counter() - t0
        rec = dict(seconds=t, W=np.asarray(out['W']), fields_lat=self.lattice(out['fields']))
        if certify:
            gam = self.certifier(kind)(out['C'], jnp.float64(nu), data, sch)
            rec['cert'] = RM.certify(out, gam, sch, level)
            rec['gamma'] = np.asarray(gam)[:int(sch['steps'])]
        rec['C'] = np.asarray(out['C'])[:int(sch['steps']) + 1]
        return rec

    def lattice(self, fields):
        idx = H.lattice_nodes(self.d, self.L)
        v = np.asarray(fields)[:, idx]
        return H.to_lattice(v, self.d)


def q1_config(records, tight, tighter, n0):
    """Per-configuration Q1 (A2.7) on a cohort: records/tight dicts case -> run record; tighter on the subset."""
    allv = all(r['cert']['valid'] for r in records.values())
    ns = sum(r['cert']['nonstat'] for r in records.values())
    st = sum(r['cert']['steps'] for r in records.values())
    anchors_ok = all(tight[c]['cert']['anchor_ok'] for c in records) and all(v['cert']['anchor_ok'] for v in tighter.values())
    import score as SC
    dist = {c: SC.lattice_distance(records[c]['fields_lat'], tight[c]['fields_lat'], n0[c]) for c in records}
    dtt = {c: SC.lattice_distance(tight[c]['fields_lat'], tighter[c]['fields_lat'], n0[c]) for c in tighter}
    ok = bool(allv and st > 0 and ns <= Q1_NONSTAT * st and anchors_ok and all(v <= Q1_DIST for v in dist.values())
              and all(v <= Q1_ANCHOR for v in dtt.values()))
    return dict(q1=ok, all_valid=allv, nonstat=ns, steps=st, anchors_ok=anchors_ok, max_dist=max(dist.values()),
                max_anchor=max(dtt.values()) if dtt else None, dist=dist, anchor=dtt)


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
