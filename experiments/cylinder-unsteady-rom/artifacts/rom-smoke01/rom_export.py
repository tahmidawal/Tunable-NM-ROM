"""G5 R2 export (cluster, NGSolve + NumPy): common quadrature, bases, projections.

DESIGN-G5 §5.2/§5.4. Inputs: reference run directories (transient.py output) listed in a
manifest with split labels, plus the frozen run-09 bank. Outputs (all float64, hashed):
  quadrature.npz      Q points/weights (order-10 mapped volume rule), inlet/outlet rules
  basis.npz           lift, POD velocity modes, supremizers, pressure basis, chi, bank(s)
                      evaluated on Q (values, physical gradients) and on the boundary rules
  pod.npz             singular values, FE coefficient modes (velocity/pressure), lift
  proj_<case>.npz     per saved time of a trajectory: inner products of the reference with
                      every basis function on Q, reference norms, times
Held-out trajectories are only processed when --allow-heldout is passed (R5).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))

RANKS = [2, 4, 8, 12, 16, 20, 24]
QORDER = 10


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshots(run_dir):
    run_dir = Path(run_dir)
    st = json.loads((run_dir / 'stages.json').read_text())
    for rec in st['snapshots']:
        p = run_dir / rec['file']
        if sha(p) != rec['sha256']:
            raise ValueError(f'snapshot hash mismatch {p}')
        z = np.load(p)
        for t, x in zip(z['t'], z['x']):
            yield float(t), x


def recorded(run_dir, stage_name):
    """Snapshots of one named stage only (the recorded window)."""
    st = json.loads((Path(run_dir) / 'stages.json').read_text())
    rec = [s for s in st['stages'] if s['name'] == stage_name]
    if len(rec) != 1 or rec[0]['status'] != 'completed':
        raise ValueError(f'stage {stage_name} missing or incomplete in {run_dir}')
    t0, t1 = rec[0]['t_start'], rec[0]['t_end']
    for t, x in snapshots(run_dir):
        if t0 - 1e-9 <= t <= t1 + 1e-9:
            yield t, x


class FE:
    def __init__(self, ng, mesh_path, mesh_sha, k=3, bonus=6):
        if sha(mesh_path) != mesh_sha:
            raise ValueError('mesh hash')
        self.ng = ng
        self.mesh = mesh = ng.Mesh(str(mesh_path))
        self.V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        self.Q = ng.H1(mesh, order=k - 1)
        self.X = self.V * self.Q
        self.nv = self.V.ndof
        dx = ng.dx(bonus_intorder=bonus)
        u, v = self.V.TnT(); p, q = self.Q.TnT()
        self.Mv = ng.BilinearForm(self.V); self.Mv += u * v * dx; self.Mv.Assemble()
        self.Mp = ng.BilinearForm(self.Q); self.Mp += p * q * dx; self.Mp.Assemble()
        self.gx = ng.GridFunction(self.X); self.gv = ng.GridFunction(self.V); self.gp = ng.GridFunction(self.Q)
        self.freev = np.array([bool(b) for b in self.V.FreeDofs()])
        self.dx = dx

    def mv(self, a, b=None, mat=None):
        mat = mat or self.Mv
        va = mat.mat.CreateColVector(); vb = mat.mat.CreateColVector()
        va.FV().NumPy()[:] = a
        vb.data = mat.mat * va
        return vb.FV().NumPy().copy() if b is None else float(b @ vb.FV().NumPy())

    def rule(self):
        """Order-10 mapped rule: element loop for weights, vectorized map for evaluation."""
        ng, mesh = self.ng, self.mesh
        ir = ng.IntegrationRule(ng.TRIG, QORDER)
        pts, w = [], []
        for el in mesh.Elements(ng.VOL):
            tr = mesh.GetTrafo(el)
            for ip in ir:
                mip = tr(ip)
                wt = float(ip.weight * mip.measure)
                if not wt > 0:
                    raise ArithmeticError('nonpositive weight')
                pts.append(tuple(mip.point)); w.append(wt)
        self.mapped = mesh.MapToAllElements(ir, ng.VOL)
        xy = np.asarray(ng.CF((ng.x, ng.y))(self.mapped))
        pts = np.asarray(pts)
        if xy.shape != pts.shape or np.max(np.abs(xy - pts)) > 1e-12:
            raise AssertionError('MapToAllElements order differs from the element loop')
        bnd = {}
        for label in ('inlet', 'outlet'):
            bp, bw = [], []
            for el in mesh.Elements(ng.BND):
                if el.mat != label:
                    continue
                tr = mesh.GetTrafo(el)
                for ip in ng.IntegrationRule(ng.SEGM, QORDER):
                    mip = tr(ip); bp.append(tuple(mip.point)); bw.append(float(ip.weight * mip.measure))
            bnd[label] = (np.asarray(bp), np.asarray(bw))
        self.bnd = bnd
        self.bmapped = {lab: [mesh(*p) for p in bnd[lab][0]] for lab in bnd}
        return pts, np.asarray(w)

    def eval_v(self, coeffs):
        """Velocity FE vector(s) -> values (N,2) and gradients (N,2,2) on Q."""
        ng = self.ng
        self.gv.vec.FV().NumPy()[:] = coeffs
        val = np.asarray(self.gv(self.mapped)).reshape(-1, 2)
        grd = np.asarray(ng.Grad(self.gv)(self.mapped)).reshape(-1, 2, 2)
        return val, grd

    def eval_p(self, coeffs):
        self.gp.vec.FV().NumPy()[:] = coeffs
        return np.asarray(self.gp(self.mapped)).reshape(-1)

    def eval_bnd(self, coeffs, kind):
        ng = self.ng
        g = self.gv if kind == 'v' else self.gp
        g.vec.FV().NumPy()[:] = coeffs
        out = {}
        for lab, mps in self.bmapped.items():
            if kind == 'v':
                out[lab] = (np.array([g(m) for m in mps]).reshape(-1, 2),
                            np.array([ng.Grad(g)(m) for m in mps]).reshape(-1, 2, 2))
            else:
                out[lab] = np.array([g(m) for m in mps]).reshape(-1)
        return out


def pod(fe, snaps_v, lift):
    """L2 method of snapshots on velocity fluctuations; f64 with re-orthogonalization."""
    Y = snaps_v - lift[None, :]
    MY = np.stack([fe.mv(y) for y in Y])
    K = Y @ MY.T
    lam, W = np.linalg.eigh(K)
    order = np.argsort(lam)[::-1]
    lam, W = lam[order], W[:, order]
    r = max(RANKS)
    sig = np.sqrt(np.maximum(lam, 0))
    modes = (W[:, :r].T @ Y) / sig[:r, None]
    for _ in range(2):  # mass-metric CGS2
        for i in range(r):
            for j in range(i):
                modes[i] -= fe.mv(modes[j], modes[i]) * modes[j]
            modes[i] /= np.sqrt(fe.mv(modes[i], modes[i]))
    G = np.array([[fe.mv(a, b) for b in modes] for a in modes])
    return modes, sig, float(np.max(np.abs(G - np.eye(r))))


def pressure_basis(fe, snaps_p, npress):
    fe.gp.Set(fe.ng.CF(1.0)); one = fe.gp.vec.FV().NumPy().copy()
    vol = fe.mv(one, one, fe.Mp)
    one /= np.sqrt(vol)
    P = snaps_p - np.outer(np.array([fe.mv(one, s, fe.Mp) for s in snaps_p]), one)
    MP = np.stack([fe.mv(s, mat=fe.Mp) for s in P])
    lam, W = np.linalg.eigh(P @ MP.T)
    order = np.argsort(lam)[::-1]; lam, W = lam[order], W[:, order]
    k = npress - 1
    modes = [one] + list((W[:, :k].T @ P) / np.sqrt(np.maximum(lam[:k], 1e-300))[:, None])
    for _ in range(2):
        for i in range(len(modes)):
            for j in range(i):
                modes[i] = modes[i] - fe.mv(modes[j], modes[i], fe.Mp) * modes[j]
            modes[i] = modes[i] / np.sqrt(fe.mv(modes[i], modes[i], fe.Mp))
    return np.array(modes), np.sqrt(np.maximum(lam, 0))


def supremizers(fe, pmodes):
    ng = fe.ng
    u, v = fe.V.TnT()
    a = ng.BilinearForm(fe.V); a += ng.InnerProduct(ng.Grad(u), ng.Grad(v)) * fe.dx; a.Assemble()
    inv = a.mat.Inverse(fe.V.FreeDofs(), inverse='sparsecholesky')
    out = []
    for c in pmodes:
        fe.gp.vec.FV().NumPy()[:] = c
        f = ng.LinearForm(fe.V); f += fe.gp * ng.div(v) * fe.dx; f.Assemble()
        s = ng.GridFunction(fe.V); s.vec.data = inv * f.vec
        out.append(s.vec.FV().NumPy().copy())
    return np.array(out)


def chi(fe):
    out = []
    for i in range(2):
        g = fe.ng.GridFunction(fe.V)
        g.Set(fe.ng.CF(tuple(1.0 if j == i else 0.0 for j in range(2))), definedon=fe.mesh.Boundaries('obstacle'))
        out.append(g.vec.FV().NumPy().copy())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='JSON: mesh, cases[{name,split,run_dir,stage,Re}]')
    ap.add_argument('--bank', required=True)
    ap.add_argument('--bank-sha256', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-heldout', action='store_true')
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    import ngsolve as ng
    ng.SetNumThreads(args.threads)
    man = json.loads(Path(args.manifest).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    log = {'started': time.time(), 'manifest_sha256': sha(args.manifest)}
    with ng.TaskManager():
        fe = FE(ng, man['mesh_path'], man['mesh_sha256'])
        pts, w = fe.rule()
        np.savez(out / 'quadrature.npz', points=pts, weights=w,
                 inlet_points=fe.bnd['inlet'][0], inlet_weights=fe.bnd['inlet'][1],
                 outlet_points=fe.bnd['outlet'][0], outlet_weights=fe.bnd['outlet'][1])
        train = [c for c in man['cases'] if c['split'] == 'train']
        sv, sp, nu_train = [], [], []
        for c in train:
            for t, x in recorded(c['run_dir'], c['stage']):
                sv.append(x[:fe.nv]); sp.append(x[fe.nv:])
        sv, sp = np.array(sv), np.array(sp)
        lift = sv.mean(axis=0)
        modes, sig, orth = pod(fe, sv, lift)
        pmodes, psig = pressure_basis(fe, sp, max(RANKS))
        sups = supremizers(fe, pmodes)
        chis = chi(fe)
        np.savez(out / 'pod.npz', lift=lift, modes=modes, singular_values=sig, orthogonality_error=orth,
                 pressure_modes=pmodes, pressure_singular_values=psig, supremizers=sups, chi=chis,
                 n_train_snapshots=len(sv))
        del sv, sp
        # Velocity functions on Q: lift, POD, supremizers, chi.
        vfun = [lift] + list(modes) + list(sups) + list(chis)
        names = ['lift'] + [f'pod{i}' for i in range(len(modes))] + [f'sup{i}' for i in range(len(sups))] + ['chi_x', 'chi_y']
        V = np.empty((len(pts), 2, len(vfun))); G = np.empty((len(pts), 2, 2, len(vfun)))
        Bv = {lab: [np.empty((len(fe.bnd[lab][1]), 2, len(vfun))), np.empty((len(fe.bnd[lab][1]), 2, 2, len(vfun)))] for lab in fe.bnd}
        for j, cf in enumerate(vfun):
            V[:, :, j], G[:, :, :, j] = fe.eval_v(cf)
            b = fe.eval_bnd(cf, 'v')
            for lab in b:
                Bv[lab][0][:, :, j], Bv[lab][1][:, :, :, j] = b[lab]
        P = np.stack([fe.eval_p(c) for c in pmodes], axis=1)
        Bp = {lab: np.stack([fe.eval_bnd(c, 'p')[lab] for c in pmodes], axis=1) for lab in fe.bnd}
        probes = []
        for c in pmodes:
            fe.gp.vec.FV().NumPy()[:] = c
            probes.append([float(fe.gp(fe.mesh(*xy))) for xy in man['probe_points']])
        probes = np.array(probes)
        from accuracy_v2.residual_head_bank import ResidualHeadBank
        if sha(args.bank) != args.bank_sha256:
            raise ValueError('bank hash')
        bank = ResidualHeadBank.load(args.bank)
        bv, bg = [], []
        for i in range(0, len(pts), 8192):
            a, b = bank.evaluate(pts[i:i + 8192]); bv.append(a); bg.append(b)
        bv, bg = np.concatenate(bv), np.concatenate(bg)
        bbnd = {lab: bank.evaluate(fe.bnd[lab][0]) for lab in fe.bnd}
        np.savez(out / 'basis.npz', names=np.array(names), V=V, G=G, P=P, probes=probes,
                 inlet_V=Bv['inlet'][0], inlet_G=Bv['inlet'][1], outlet_V=Bv['outlet'][0], outlet_G=Bv['outlet'][1],
                 inlet_P=Bp['inlet'], outlet_P=Bp['outlet'],
                 frozen_bank_V=bv, frozen_bank_G=bg,
                 frozen_bank_inlet_V=bbnd['inlet'][0], frozen_bank_inlet_G=bbnd['inlet'][1],
                 frozen_bank_outlet_V=bbnd['outlet'][0], frozen_bank_outlet_G=bbnd['outlet'][1])
        # Projections of non-training trajectories (dev always; held-out only with the flag).
        allb_v = np.concatenate([V, bv], axis=2)
        for c in man['cases']:
            if c['split'] == 'train' or (c['split'] == 'heldout' and not args.allow_heldout):
                continue
            rows = {'t': [], 'uv': [], 'uu': [], 'pp': [], 'pv': [], 'pref': []}
            for t, x in recorded(c['run_dir'], c['stage']):
                u, gu = fe.eval_v(x[:fe.nv]); p = fe.eval_p(x[fe.nv:])
                rows['t'].append(t)
                rows['uv'].append(np.einsum('q,qc,qcj->j', w, u, allb_v))
                rows['uu'].append(float(np.einsum('q,qc,qc->', w, u, u)))
                rows['pp'].append(float(np.einsum('q,q,q->', w, p, p)))
                rows['pv'].append(np.einsum('q,q,qk->k', w, p, P))
                rows['pref'].append(float(w @ p))
            np.savez(out / f"proj_{c['name']}.npz", **{k: np.asarray(v) for k, v in rows.items()},
                     columns=np.array(names + [f'bank{i}' for i in range(bv.shape[2])]))
    log['finished'] = time.time()
    log['files'] = {p.name: sha(p) for p in sorted(out.glob('*.npz'))}
    (out / 'EXPORT.json').write_text(json.dumps(log, indent=2) + '\n')


if __name__ == '__main__':
    main()
