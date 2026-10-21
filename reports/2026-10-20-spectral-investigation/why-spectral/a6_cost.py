"""A6. Cost per step, work per step and the cost model (CPU; reads measured H200 timings from the lanes).

Measured inputs (all H200, f64, JAX_DEFAULT_MATMUL_PRECISION=highest, jax_backend=gpu, as recorded by the lanes):
  * 2D spectral FOM: jcp-fom2d stage-0 warm costs of every PS (N, dt) at the 256^2 / 1024^2 / 4096^2 full-field
    contracts (checks/c2<L>-result.json `stage0`), and lattice-contract screening costs (checks/match-final.json
    pareto.PS.points); 3D spectral FOM: jcp-speed3d f1 warm medians per (N, dt) (lattice contract).
    For each N, cost = alpha + n_steps * tau (least squares over dt), so tau is the per-step cost and alpha the
    per-query overhead (coefficient set-up, IC transfer, output synthesis).
  * 2D ROM: jcp-p1-scaling h_cost_n (t_init, t_evolve over 50 steps at dt0 = 0.005, t_decode) for R128/R256/R384 at
    256^2 / 1024^2 / 4096^2; component kernels (nominal M = 4r, own rule); jcp-fom2d stage-2 full queries at 4096^2.
  * 3D ROM: jcp-speed3d p1 H31 fused query and components for its four selected ROMs (mesh-free path).
Work counts are operation-count models (stated below), not measurements.
Writes analysis/a6_cost.json.
"""
import json
import math

import numpy as np

import common as C


def lsq_alpha_tau(pairs):
    n = np.array([p[0] for p in pairs], float)
    t = np.array([p[1] for p in pairs], float)
    A = np.c_[np.ones_like(n), n]
    (a, tau), *_ = np.linalg.lstsq(A, t, rcond=None)
    res = t - A @ np.array([a, tau])
    return float(a), float(tau), float(np.max(np.abs(res) / t))


def w_ps(N, d):
    """flops per ETDRK4 step of pscore.nonlinear: 4 stages x T transforms, each (M-1)^(d-1) real FFTs of length 2M
    (cost model 2.5 L log2 L per real FFT of length L), M = 3N/2; T = 7 (2D) or 12 (3D) as in pscore.nonlinear."""
    M = 3 * N // 2
    T = 7 if d == 2 else 12
    return 4 * T * (M - 1) ** (d - 1) * 2.5 * (2 * M) * math.log2(2 * M)


def bytes_ps(N, d):
    M = 3 * N // 2
    return 8 * (M - 1) ** d          # one padded real field; the working set is a few of these


RULE_M = {'fib4181': 4181, 'gauss128': 128 ** 2, 'gauss192': 192 ** 2, 'gl24': 24 ** 3, 'lat32768': 32768,
          'lat65536': 65536}


def w_rom(r, M, m, nj):
    """flops per implicit step, dominant term: nj Jacobian assemblies Psi^T[diag(Dc)B + diag(Bc)D] of 2 m M r each."""
    return nj * 2.0 * m * M * r


def bytes_rom(r, M, m):
    """tables read per Jacobian: Psi (m x M) and the two bank tables (m x r), f64."""
    return 8.0 * (m * M + 2 * m * r)


def main():
    out = {}
    # ---------------- 2D PS, full-field stage 0 (per mesh) and lattice screening ----------------
    ps2 = {}
    for L in (256, 1024, 4096):
        s0 = C.J(C.FOM / f'checks/c2{L}-result.json')['stage0']
        byN = {}
        for k, v in s0.items():
            if not k.startswith('ps|'):
                continue
            _, Ns, dt = k.split('|')
            byN.setdefault(int(Ns[1:]), []).append((round(0.25 / float(dt)), v['cost']))
        ps2[L] = {N: dict(zip(('alpha', 'tau', 'max_rel_resid'), lsq_alpha_tau(p))) | dict(points=sorted(p))
                  for N, p in sorted(byN.items())}
    mf = C.J(C.FOM / 'checks/match-final.json')
    adm = mf['admissibility']
    byN = {}
    for k, v in adm.items():
        if k.startswith('ps|'):
            _, Ns, dt = k.split('|')
            byN.setdefault(int(Ns[1:]), []).append((round(0.25 / float(dt)), v['cost'], v['admissible']))
    lat = {}
    for N, p in sorted(byN.items()):
        a, tau, rr = lsq_alpha_tau([(n, c) for n, c, _ in p])
        dts = [0.25 / n for n, _, ok in p if ok]
        lat[N] = dict(alpha=a, tau=tau, max_rel_resid=rr, dt_max_admissible=max(dts) if dts else None,
                      inadmissible_dts=sorted(0.25 / n for n, _, ok in p if not ok), points=sorted((n, c) for n, c, _ in p))
    out['ps2_full'] = ps2
    out['ps2_lattice'] = lat
    # ---------------- 3D PS (f1, lattice) ----------------
    f1 = C.J(C.S3D / 'runs/f1/archive/output/result.json')['fom']['ps']
    byN = {}
    for k, v in f1.items():
        _, Ns, dt = k.split('|')
        byN.setdefault(int(Ns[1:]), []).append((round(0.25 / float(dt)), v['warm']['median']))
    out['ps3_lattice'] = {N: dict(zip(('alpha', 'tau', 'max_rel_resid'), lsq_alpha_tau(p))) | dict(points=sorted(p))
                          for N, p in sorted(byN.items())}
    # ---------------- 2D ROM (P1) ----------------
    a = C.J(C.P1 / 'checks/analysis.json')
    hc = a['timing']['h_cost_n']
    comp = [c for c in a['timing']['components'] if c['subject'] == '1024|R384|gauss128|frozen|CNR'][0]
    it_mean = float(np.mean(comp['iterations']))
    rules = {128: ('fib4181', 4181), 256: ('gauss128', 16384), 384: ('gauss128', 16384), 512: ('gauss192', 36864)}
    rom2 = {}
    for key, v in hc.items():
        r, sch, pol = key.split('|')
        r = int(r[1:])
        if pol != 'frozen':
            continue
        ds = v['descriptive_seconds']
        M, m = 4 * r, rules[r][1]
        nj = 1 + it_mean
        rom2[key] = dict(r=r, M=M, m=m, rule=rules[r][0], scheme=sch, steps=50,
                         t_init=ds['t_init'], t_evolve=ds['t_evolve'], t_decode=ds['t_decode'],
                         tau=ds['t_evolve']['1024'] / 50, W_step=w_rom(r, M, m, nj), bytes_per_jac=bytes_rom(r, M, m),
                         nj=nj)
    out['rom2'] = rom2
    out['rom2_it_mean'] = it_mean
    out['rom2_components_R384'] = {k: comp['components'][k] for k in ('hist_cn', 'pred', 'init', 'iter')}
    nom = a['kernels_2d']['descriptive']['nominal']
    out['rom2_kernels_nominal'] = {k: nom[k] for k in ('N', 'J', 'JTJ', 'chol', 'res', 'evalJ')}
    out['rom2_kernel_J_flops'] = {r: 2.0 * rules[r][1] * 4 * r * r for r in (128, 256, 384, 512)}
    # fom2d stage-2 full-field queries at 4096^2 (absolute seconds)
    cells = C.J(C.FOM / 'checks/c2-analysis.json')['cells']
    q4 = {}
    for k, v in cells.items():
        if v['family'] == 'union' and v['criterion'] == 'joint' and '|H|' in k:
            L, rest = k.split('|', 1)
            lab = rest.split('@')[0].replace('ckpt:18f0266ae6f0|', '')
            q4.setdefault(L, {})[lab] = dict(rom_s=v['seconds_rom'], fom_s=v['seconds_fom'], fom=v['selected'], S=v['S'])
    out['c2_queries'] = q4
    # ---------------- 3D ROM (speed3d p1 H31) ----------------
    ex = C.J(C.S3D / 'runs/EXPORT.json')
    keys = dict(W_C_W768=('M2_512|gl24|qn|CN|0.016666667|k4', 512, 2052, 24 ** 3, 15),
                W_C_old=('M2_512|gl24|fs|BDF2|0.0125|K1', 512, 2052, 24 ** 3, 20),
                C_old=(None, None, None, None, None), C_W1024=(None, None, None, None, None))
    rom3 = {}
    for lab, v in ex['p1']['H31'].items():
        cs = v['components_seconds']
        spec = keys.get(lab)
        d = dict(fused=v['fused_seconds'], project=cs['project'], evolve=cs['evolve'], decode=cs['decode'],
                 contraction=cs['contraction'], dispatch_floor=cs['dispatch_floor'])
        if spec and spec[0]:
            d.update(config=spec[0], r=spec[1], M=spec[2], m=spec[3], steps=spec[4], tau=cs['evolve'] / spec[4],
                     W_contraction=2.0 * spec[3] * spec[2] * spec[1])
        rom3[lab] = d
    out['rom3'] = rom3
    out['rom3_m1_L384'] = {k: v for k, v in ex['m1']['rom_per_mesh']['384'].items()}
    # ---------------- fits: per-step time = tau0 + W / P ----------------
    def fit(pairs):
        W = np.array([p[0] for p in pairs]); t = np.array([p[1] for p in pairs])
        A = np.c_[np.ones_like(W), W]
        (t0, invP), *_ = np.linalg.lstsq(A / t[:, None], np.ones_like(t), rcond=None)   # relative least squares
        return dict(tau0=float(t0), P=float(1 / invP) if invP > 0 else None,
                    max_rel_resid=float(np.max(np.abs(t - A @ np.array([t0, invP])) / t)))
    pts_ps2 = [(w_ps(N, 2), v['tau']) for N, v in ps2[256].items()]
    pts_ps3 = [(w_ps(N, 3), v['tau']) for N, v in out['ps3_lattice'].items()]
    pts_rom2 = [(v['W_step'], v['tau']) for v in rom2.values()]
    def fit_asym(pairs, n_small):
        # latency floor = median per-step time of the n_small smallest-work points; throughput from the largest point
        ps_ = sorted(pairs)
        t0 = float(np.median([p[1] for p in ps_[:n_small]]))
        W, t = ps_[-1]
        return dict(tau0=t0, P=float(W / (t - t0)))
    out['fit_ps2_asym'] = fit_asym(pts_ps2, 4)
    out['fit_ps3_asym'] = fit_asym(pts_ps3, 1)
    out['fit_ps2'] = fit(pts_ps2) | dict(points=pts_ps2)
    out['fit_ps3'] = fit(pts_ps3) | dict(points=pts_ps3)
    out['fit_rom2'] = fit(pts_rom2) | dict(points=pts_rom2)
    out['W_ps'] = {f'{d}d': {N: w_ps(N, d) for N in (16, 24, 32, 48, 64, 96, 128, 256)} for d in (2, 3)}
    out['bytes_ps'] = {f'{d}d': {N: bytes_ps(N, d) for N in (16, 24, 48, 64, 128, 256)} for d in (2, 3)}
    (C.OUT / 'a6_cost.json').write_text(json.dumps(out, indent=0, default=str))
    for k in ('fit_ps2', 'fit_ps3', 'fit_rom2'):
        print(k, {a_: out[k][a_] for a_ in ('tau0', 'P', 'max_rel_resid')})
    for N in (16, 24, 48, 96, 256):
        print('PS2', N, 'tau %.3f ms alpha(256) %.2f alpha(4096) %.2f  lattice tau %.3f alpha %.2f dtmax %s' % (
            1e3 * ps2[256][N]['tau'], 1e3 * ps2[256][N]['alpha'], 1e3 * ps2[4096][N]['alpha'], 1e3 * lat[N]['tau'],
            1e3 * lat[N]['alpha'], lat[N]['dt_max_admissible']))
    for N, v in out['ps3_lattice'].items():
        print('PS3', N, 'tau %.3f ms alpha %.2f ms resid %.3f' % (1e3 * v['tau'], 1e3 * v['alpha'], v['max_rel_resid']))
    for k, v in rom2.items():
        print(k, 'tau %.3f ms W %.3g flop  rate %.1f TF/s  bytes/J %.3g' % (1e3 * v['tau'], v['W_step'], v['W_step'] / v['tau'] / 1e12, v['bytes_per_jac']))
    for k, v in rom3.items():
        print('3D', k, {a_: (round(1e3 * b, 3) if isinstance(b, float) else b) for a_, b in v.items()})


if __name__ == '__main__':
    main()
