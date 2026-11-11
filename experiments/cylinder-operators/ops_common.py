"""Shared scoring for all operator arms (PROTOCOL.md "Metric"): map native predictions to the ROM's
scoring quadrature Q, phase-sensitive field errors, gradient-free force read-out, cycle statistics."""
import json
import math
from pathlib import Path

import numpy as np
import torch

import cycles as cyc

TARGETS = {'e_u_rms': 1e-2, 'e_u_max': 3e-2, 'e_p_rms': 2e-2, 'St_rel': 1e-2, 'CD_max_rel': 2e-2, 'CL_max_abs': 2e-2}
CENTER, RADIUS, DELTA = (0.2, 0.2), 0.05, 0.1
NX, NY = 1024, 192


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def csr(data, indices, indptr, shape, device):
    return torch.sparse_csr_tensor(torch.as_tensor(indptr.astype(np.int64)), torch.as_tensor(indices.astype(np.int64)),
                                   torch.as_tensor(data, dtype=torch.float64), size=shape).to(device)


def ddt(y, dt):
    """Time derivative on a uniform grid: 4th-order central inside, 2nd-order one-sided at the two ends."""
    y = np.asarray(y, float); d = np.gradient(y, dt, axis=0, edge_order=2)
    d[2:-2] = (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * dt)
    return d


def stats(t, cd, cl):
    recs = cyc.cycles(t, cd, cl)
    if not recs:
        return None
    return {k: float(np.mean([r[k] for r in recs])) for k in ('CD_max', 'CL_max', 'CL_min', 'St', 'period')} | {'cycles': len(recs)}


def robust_stats(t, cd, cl, min_dist):
    """Cycle statistics with a minimum distance between lift minima (keeps the lower of two close minima)."""
    mins = []
    for i, tm in cyc.lift_minima(t, cl):
        if mins and tm - mins[-1][1] < min_dist:
            if cl[i] < cl[mins[-1][0]]:
                mins[-1] = (i, tm)
        else:
            mins.append((i, tm))
    if len(mins) < 2:
        return None
    recs = []
    for (i0, t0), (i1, t1) in zip(mins[:-1], mins[1:]):
        w = slice(i0, i1 + 1)
        recs.append({'St': 0.1 / (t1 - t0), 'CD_max': cyc.extremum(t, cd, w, 1), 'CL_max': cyc.extremum(t, cl, w, 1)})
    return {k: float(np.mean([r[k] for r in recs])) for k in ('St', 'CD_max', 'CL_max')} | {'cycles': len(recs), 'St_per_cycle': [float(r['St']) for r in recs]}


class Geometry:
    def __init__(self, data_dir, device):
        z = np.load(Path(data_dir) / 'geom.npz')
        self.z = z; self.device = device
        self.xq = z['Q_points']; self.wq = z['Q_weights']; self.nQ = len(self.wq)
        self.vol = float(self.wq.sum())
        self.w = torch.as_tensor(self.wq, dtype=torch.float64, device=device)
        self.fluid = z['grid_fluid']; self.inlet_u = z['inlet_u']
        self._maps = {}

    def node_xy(self, k):
        return self.z[f'n{k}_xy']

    def node_weights(self, k):
        return self.z[f'n{k}_weights']

    def node_map(self, k):
        if k not in self._maps:
            z = self.z
            self._maps[k] = csr(z[f'n{k}_A_data'], z[f'n{k}_A_indices'], z[f'n{k}_A_indptr'], (self.nQ, len(z[f'n{k}_xy'])), self.device)
        A = self._maps[k]
        return lambda f: A @ f.to(torch.float64)            # (nodes, 3) -> (nQ, 3)

    def grid_map(self):
        if 'g' not in self._maps:
            z = self.z
            self._maps['g'] = csr(z['G_data'], z['G_indices'], z['G_indptr'], (self.nQ, (NX + 2) * (NY + 2)), self.device)
            self._inlet = torch.as_tensor(self.inlet_u, dtype=torch.float64, device=self.device)
            self._fluid = torch.as_tensor(self.fluid, device=self.device)
        G = self._maps['g']

        def f(g):                                            # (3, NY, NX) -> (nQ, 3)
            g = g.to(torch.float64).clone()
            g[:2] = g[:2] * self._fluid                     # no-slip inside the masked cylinder
            o = torch.zeros((3, NY + 2, NX + 2), dtype=torch.float64, device=g.device)
            o[:, 1:-1, 1:-1] = g
            o[:, 1:-1, -1] = g[:, :, -1]
            o[2, 1:-1, 0] = g[2, :, 0]
            o[0, 1:-1, 0] = self._inlet
            o[2, 0, :] = o[2, 1, :]; o[2, -1, :] = o[2, -2, :]
            return G @ o.reshape(3, -1).T
        return f


class Scorer:
    """Scores one case. ref_dir holds <case>_Q.npy, _t.npy, _hist.npy (data dir, or the sealed dir for held-out)."""

    def __init__(self, geom, ref_dir, case, Re):
        self.g = geom; dev = geom.device
        self.case = case; self.nu = 0.1 / Re
        self.ref = np.load(Path(ref_dir) / f'{case}_Q.npy', mmap_mode='r')
        self.t = np.load(Path(ref_dir) / f'{case}_t.npy')
        self.hist = np.load(Path(ref_dir) / f'{case}_hist.npy')
        x = torch.as_tensor(geom.xq, dtype=torch.float64, device=dev)
        d = x - torch.tensor(CENTER, dtype=torch.float64, device=dev)
        r = d.norm(dim=1); rh = d / r[:, None]
        s = ((r - RADIUS) / DELTA).clamp(0, 1)
        eta = 1 - (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
        e1 = -(30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / DELTA
        e2 = -(60 * s - 180 * s ** 2 + 120 * s ** 3) / DELTA ** 2
        w = geom.w
        self.w_eta = w * eta
        self.w_geta = w[:, None] * e1[:, None] * rh                               # w * grad eta
        self.w_lap = w * (e2 + e1 / r)
        eye = torch.eye(2, dtype=torch.float64, device=dev)
        rr = rh[:, :, None] * rh[:, None, :]
        self.w_hess = w[:, None, None] * (e2[:, None, None] * rr + (e1 / r)[:, None, None] * (eye - rr))
        h = self.hist
        self.ref_stats = stats(h[1:, 0], h[1:, 1], h[1:, 2])

    def ref_Q(self, i):
        return torch.as_tensor(np.asarray(self.ref[i]), dtype=torch.float64, device=self.g.device)

    def functionals(self, f):
        """f (nQ, 3) -> momentum integral m (2,) and the steady part of the force (2,)."""
        u, p = f[:, :2], f[:, 2]
        m = (self.w_eta[:, None] * u).sum(0)
        conv = (u * (u * self.w_geta).sum(1, keepdim=True)).sum(0)
        pres = (p[:, None] * self.w_geta).sum(0)
        visc = self.nu * ((u * self.w_lap[:, None]).sum(0) + torch.einsum('qj,qij->i', u, self.w_hess))
        return m.cpu().numpy(), (conv + pres + visc).cpu().numpy()

    def forces(self, M, S):
        dt = float(np.mean(np.diff(self.t)))
        F = -ddt(M, dt) + S
        return 20 * F[:, 0], 20 * F[:, 1]

    def reference_readout(self):
        M, S = [], []
        for i in range(len(self.t)):
            m, s = self.functionals(self.ref_Q(i)); M.append(m); S.append(s)
        cd, cl = self.forces(np.array(M), np.array(S))
        st = stats(self.t, cd, cl)
        return {'cycles': st, 'reference_cycles': self.ref_stats, 'errors': self.cycle_errors(st),
                'CD_maxabs_rel_vs_history': float(np.max(np.abs(cd[3:-3] - np.interp(self.t, self.hist[:, 0], self.hist[:, 1])[3:-3])) / np.max(np.abs(self.hist[:, 1]))),
                'CL_maxabs_vs_history': float(np.max(np.abs(cl[3:-3] - np.interp(self.t, self.hist[:, 0], self.hist[:, 2])[3:-3])))}, cd, cl

    def cycle_errors(self, st):
        rs = self.ref_stats
        if not st or not rs:
            return None
        return {'St_rel': abs(st['St'] - rs['St']) / rs['St'], 'CD_max_rel': abs(st['CD_max'] - rs['CD_max']) / rs['CD_max'],
                'CL_max_abs': abs(st['CL_max'] - rs['CL_max'])}

    def score(self, preds, to_Q):
        """preds: iterable of native predictions for the saves (in order); to_Q maps one to (nQ, 3)."""
        e_u, e_p, M, S = [], [], [], []
        w = self.g.w
        for i, pr in enumerate(preds):
            f = to_Q(pr); r = self.ref_Q(i)
            if not bool(torch.isfinite(f).all()):
                e_u.append(float('nan')); e_p.append(float('nan')); M.append([np.nan] * 2); S.append([np.nan] * 2)
                continue
            un = torch.sqrt((w * (r[:, :2] ** 2).sum(1)).sum())
            e_u.append(float(torch.sqrt((w * ((f[:, :2] - r[:, :2]) ** 2).sum(1)).sum()) / un))
            e_p.append(float(torch.sqrt((w * (f[:, 2] - r[:, 2]) ** 2).sum()) / math.sqrt(self.g.vol)))
            m, s = self.functionals(f); M.append(m); S.append(s)
        if len(e_u) != len(self.t):
            raise ValueError(f'expected {len(self.t)} saves, got {len(e_u)}')
        e_u, e_p = np.array(e_u), np.array(e_p)
        rec = {'case': self.case, 'saves': len(e_u), 'finite': bool(np.isfinite(e_u).all())}
        series = {'t': self.t, 'e_u': e_u, 'e_p': e_p}
        if not rec['finite']:
            rec['status'] = 'failed_nonfinite'; rec['first_nonfinite_save'] = int(np.argmax(~np.isfinite(e_u)))
            return rec, series
        cd, cl = self.forces(np.array(M), np.array(S))
        st = stats(self.t, cd, cl)
        ce = self.cycle_errors(st) or {}
        period = self.ref_stats['period']; ts = self.t
        last = ts >= ts[-1] - period
        rec.update(status='completed', e_u={'rms': rms(e_u), 'max': float(e_u.max()), 'last_period_mean': float(e_u[last].mean())},
                   e_p={'rms': rms(e_p), 'max': float(e_p.max()), 'last_period_mean': float(e_p[last].mean())},
                   cycles=st, reference_cycles=self.ref_stats, cycle_errors=ce)
        # lift phase lag at the end of the window (support rule of rom_r4.py)
        ds = ts[1] - ts[0]; kmax = int(round(0.5 * period / ds))
        I = np.where((ts >= ts[-1] - 1.5 * period) & (ts <= ts[-1] - 0.5 * period))[0]
        cl_ref = np.interp(ts, self.hist[:, 0], self.hist[:, 2])
        best = min(((k, rms(cl[I] - cl_ref[I + k])) for k in range(-kmax, kmax + 1)), key=lambda z: z[1])
        rec['phase'] = {'lag_saves': int(best[0]), 'lag_seconds': float(best[0] * ds), 'lag_fraction_of_period': float(best[0] * ds / period),
                        'lift_rms_at_lag': best[1], 'phase_accurate': abs(best[0]) <= 2}
        got = {'e_u_rms': rec['e_u']['rms'], 'e_u_max': rec['e_u']['max'], 'e_p_rms': rec['e_p']['rms'], **ce}
        rec['targets'] = {k: {'value': got.get(k), 'target': v, 'met': got.get(k) is not None and got[k] <= v} for k, v in TARGETS.items()}
        rec['targets_met'] = int(sum(v['met'] for v in rec['targets'].values()))
        series.update(CD=cd, CL=cl, CD_ref=np.interp(ts, self.hist[:, 0], self.hist[:, 1]), CL_ref=cl_ref)
        # supplementary (post hoc, PROTOCOL amendment 04:58): cycle statistics after a centred 9-save moving average,
        # prediction and reference history filtered identically; does not replace the fixed read-out above
        ker = np.ones(9) / 9; sm = lambda z: np.convolve(z, ker, mode='valid')
        h = self.hist[1:]; cdr = np.interp(ts, h[:, 0], h[:, 1]); clr = np.interp(ts, h[:, 0], h[:, 2])
        st_s, rs_s = stats(ts[4:-4], sm(cd), sm(cl)), stats(ts[4:-4], sm(cdr), sm(clr))
        sup = {'cycles': st_s, 'reference_cycles': rs_s, 'lift_error_rms': rms(cl[3:-3] - clr[3:-3]), 'lift_error_max': float(np.max(np.abs(cl[3:-3] - clr[3:-3]))),
               'drag_error_max_rel': float(np.max(np.abs(cd[3:-3] - cdr[3:-3])) / np.max(np.abs(cdr)))}
        if st_s and rs_s:
            sup['cycle_errors'] = {'St_rel': abs(st_s['St'] - rs_s['St']) / rs_s['St'], 'CD_max_rel': abs(st_s['CD_max'] - rs_s['CD_max']) / rs_s['CD_max'],
                                   'CL_max_abs': abs(st_s['CL_max'] - rs_s['CL_max'])}
            sup['targets_met_with_smoothed_cycles'] = int(sum(rec['targets'][k]['met'] for k in ('e_u_rms', 'e_u_max', 'e_p_rms'))
                                                          + sum(sup['cycle_errors'][k] <= TARGETS[k] for k in ('St_rel', 'CD_max_rel', 'CL_max_abs')))
        rec['supplementary_smoothed9'] = sup
        # diagnostic (post hoc, chosen after seeing that one double minimum survives the smoothing): same smoothed
        # signals, lift minima closer than half a reference period to a lower minimum are discarded
        rec['diagnostic_min_distance'] = {'cycles': robust_stats(ts[4:-4], sm(cd), sm(cl), 0.5 * period)}
        rr = robust_stats(ts[4:-4], sm(cdr), sm(clr), 0.5 * period); st_r = rec['diagnostic_min_distance']['cycles']
        rec['diagnostic_min_distance']['reference_cycles'] = rr
        if st_r and rr:
            rec['diagnostic_min_distance']['cycle_errors'] = {
                'St_rel': abs(st_r['St'] - rr['St']) / rr['St'], 'CD_max_rel': abs(st_r['CD_max'] - rr['CD_max']) / rr['CD_max'],
                'CL_max_abs': abs(st_r['CL_max'] - rr['CL_max']), 'St_rel_worst_cycle': float(np.max(np.abs(np.array(st_r['St_per_cycle']) - rr['St'])) / rr['St'])}
        return rec, series


def calibrate(data_dir, case, out):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = Geometry(data_dir, dev)
    cases = json.loads((Path(data_dir) / 'cases.json').read_text())
    sc = Scorer(g, data_dir, case, cases[case]['Re'])
    rec, cd, cl = sc.reference_readout()
    # self-consistency: reference node/grid data mapped to Q and scored (floors incl. force read-out)
    floors = {}
    for name, arr, mp in (('n2', f'{case}_n2.npy', g.node_map(2)), ('n3', f'{case}_n3.npy', g.node_map(3)), ('grid', f'{case}_grid.npy', g.grid_map())):
        a = np.load(Path(data_dir) / arr, mmap_mode='r')
        r, _ = sc.score((torch.as_tensor(np.asarray(a[i]), device=dev) for i in range(len(a))), mp)
        floors[name] = r
    rec['representation_floors'] = floors
    Path(out).write_text(json.dumps(rec, indent=2, default=float) + '\n')
    print('readout', json.dumps(rec['errors']), rec['CD_maxabs_rel_vs_history'], rec['CL_maxabs_vs_history'])
    for k, r in floors.items():
        print(k, r['e_u'], r['e_p'], r['cycle_errors'], r['phase'])


if __name__ == '__main__':
    import sys
    calibrate(sys.argv[1], sys.argv[2], sys.argv[3])
