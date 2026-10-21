"""A7. Cost model, its check against the measured matched speed-ups, and the predicted crossover (CPU).

Model (mesh-free / lattice contract unless stated; every constant is read from A2, A2b, A4 and A6):
  spectral FOM   T_PS(N)  = n_PS * (tau0_PS + W_PS(N) / P_PS) + alpha_PS
  ROM            T_ROM(r) = n_ROM * (tau0_R + W_R(r) / P_R) + alpha_R
  W_PS(N) = 4 stages x T transforms x (M-1)^(d-1) FFTs of length 2M (2.5 L log2 L each), M = 3N/2
  W_R(r)  = n_J x 2 m M r with M = 4r, m = kappa r (n_J Jacobian assemblies per step)
tau0, P fitted per method on the measured H200 per-step times (A6); n_PS from the measured stability limit (A2b and
jcp-fom2d admissibility); N from the measured per-case spectral accuracy (A2); r = K / rho with K = (N-1)^d sine modes and
rho the compression of the learned basis over sines (A3, A5).
Writes analysis/a7_model.json.
"""
import json
import math

import numpy as np

import common as C
from a6_cost import w_ps

T_END = 0.25


def main():
    c6 = C.J(C.OUT / 'a6_cost.json')
    out = {}
    fp2, fp3, fr = c6['fit_ps2'], c6['fit_ps3'], c6['fit_rom2']
    lat2 = {int(k): v for k, v in c6['ps2_lattice'].items()}
    lat3 = {int(k): v for k, v in c6['ps3_lattice'].items()}
    alpha_ps2 = float(np.median([lat2[N]['alpha'] for N in lat2 if N <= 128]))
    alpha_ps3 = float(np.median([lat3[N]['alpha'] for N in lat3 if N <= 32]))
    rom2 = c6['rom2']
    alpha_r2 = rom2['R384|CNR|frozen']['t_init']['256'] + rom2['R384|CNR|frozen']['t_decode']['256']
    r3 = c6['rom3']['W_C_W768']
    alpha_r3 = r3['project'] + r3['decode']
    nj = c6['rom2_it_mean'] + 1
    kappa2 = 16384 / 384
    out['constants'] = dict(tau0_ps2=fp2['tau0'], P_ps2=fp2['P'], tau0_ps3=fp3['tau0'], P_ps3=fp3['P'],
                            tau0_r=fr['tau0'], P_r=fr['P'], alpha_ps2=alpha_ps2, alpha_ps3=alpha_ps3,
                            alpha_r2=alpha_r2, alpha_r3=alpha_r3, nj=nj, kappa2=kappa2,
                            tau_r3_meas=r3['tau'], r3=r3['r'], M3=r3['M'], m3=r3['m'], steps_r3=r3['steps'])

    VAR = {'fit': (fp2, fp3), 'asym': (c6['fit_ps2_asym'], c6['fit_ps3_asym'])}
    out['constants'].update(asym_ps2=VAR['asym'][0], asym_ps3=VAR['asym'][1])

    def t_ps(N, d, n, var='fit'):
        # per-query overhead interpolated from the measured lattice alphas, held at the largest measured value beyond
        f2, f3 = VAR[var]
        if d == 2:
            a_ = np.interp(N, sorted(lat2), [lat2[k]['alpha'] for k in sorted(lat2)])
            return n * (f2['tau0'] + w_ps(N, 2) / f2['P']) + a_
        a_ = np.interp(N, sorted(lat3), [lat3[k]['alpha'] for k in sorted(lat3)])
        return n * (f3['tau0'] + w_ps(N, 3) / f3['P']) + a_

    def t_rom(r, d, n):
        if d == 2:
            return n * (fr['tau0'] + nj * 2 * kappa2 * r * 4 * r * r / fr['P']) + alpha_r2
        # 3D: anchored on the measured W_C_W768 step (QN, Jacobian reuse), cubic in r at fixed m/r, M/r
        return n * (fr['tau0'] + (r3['tau'] - fr['tau0']) * (r / r3['r']) ** 3) + alpha_r3

    # ---------- model check against measured per-query times ----------
    chk = {}
    for N in (16, 24, 48, 96, 128, 256):
        n = round(T_END / 0.0125)
        chk[f'PS2 N{N} n{n}'] = dict(model=t_ps(N, 2, n), model_asym=t_ps(N, 2, n, 'asym'),
                                     meas=[c for nn, c in lat2[N]['points'] if nn == n][0])
    for N in (16, 32, 64):
        n = round(T_END / 0.01)
        chk[f'PS3 N{N} n{n}'] = dict(model=t_ps(N, 3, n), model_asym=t_ps(N, 3, n, 'asym'),
                                     meas=[c for nn, c in lat3[N]['points'] if nn == n][0])
    for key in ('R128|BE|frozen', 'R256|BE|frozen', 'R384|CNR|frozen'):
        v = rom2[key]
        chk[f'ROM2 {key} lattice'] = dict(model=t_rom(v['r'], 2, 50), model_own_kappa=50 * (fr['tau0'] + v['W_step'] / fr['P']) + alpha_r2,
                                          meas=v['t_init']['256'] + v['t_evolve']['1024'] + v['t_decode']['256'])
    chk['ROM3 W_C_W768'] = dict(model=t_rom(512, 3, r3['steps']), meas=r3['fused'])
    for v in chk.values():
        v['ratio'] = v['model'] / v['meas']
        if 'model_asym' in v:
            v['ratio_asym'] = v['model_asym'] / v['meas']
    out['check'] = chk
    # predicted vs measured matched speed-up, lattice contract
    a4 = C.J(C.OUT / 'a4_match.json')
    pred = {}
    for cfg, lab, n_rom in (('1024|R384|LSPG|CNR|1', 'R384|CNR|frozen', 50), ('1024|R128|LSPG|BE|1', 'R128|BE|frozen', 50),
                            ('1024|R256|LSPG|BE|1', 'R256|BE|frozen', 50)):
        Nj = a4['rom2d'][cfg]['N_joint_charged']
        v = rom2[lab]
        meas_rom = v['t_init']['256'] + v['t_evolve']['1024'] + v['t_decode']['256']
        ps_meas = [c for nn, c in lat2[Nj]['points'] if nn == 20] if Nj in lat2 else []
        pred[cfg] = dict(N_joint=Nj, S_model=t_ps(Nj, 2, 20) / t_rom(v['r'], 2, n_rom),
                         S_meas_lattice=(ps_meas[0] / meas_rom) if ps_meas else None,
                         S_with_measured_rom=t_ps(Nj, 2, 20) / meas_rom)
    out['pred_S'] = pred
    # ---------- crossover: ROM with zero-cost unknowns (latency floor) ----------
    cross = {}
    for d, nps_list, Ns in ((2, (20, 40, 80), [2 ** k for k in range(4, 14)]), (3, (10, 25, 50), [8 * k for k in range(2, 65)])):
        for n_ps in nps_list:
            for n_rom in (50, 20, 10):
                floor = t_rom(1e-9, d, n_rom)
                for Sbar in (1.0, 2.0):
                    row = dict(rom_floor_s=floor)
                    for var in ('fit', 'asym'):
                        Nc = None
                        grid = np.arange(8, 4097, 4) if d == 2 else np.arange(8, 513, 2)
                        for N in grid:
                            if t_ps(int(N), d, n_ps, var) >= Sbar * floor:
                                Nc = int(N)
                                break
                        row[f'N_c_{var}'] = Nc
                        row[f'below_grid_{var}'] = bool(Nc == 8)
                    cross[f'd{d}|nps{n_ps}|nrom{n_rom}|S{Sbar:g}'] = row
    out['crossover_floor'] = cross
    # ---------- required compression rho*(N) for S = 1, 2 ----------
    rho = {}
    G2 = (24, 48, 96, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048)
    G3 = (12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256)
    for d, n_ps, n_rom, Ns in ((2, 20, 50, G2), (2, 20, 20, G2), (3, 10, 15, G3), (3, 25, 15, G3)):
        for N, var in [(N, v) for N in Ns for v in ('fit', 'asym')]:
            K = (N - 1) ** d
            T = t_ps(N, d, n_ps, var)
            row = dict(K=K, T_ps=T, var=var)
            for Sbar in (1.0, 2.0):
                lo, hi = 1e-6, float(K)          # largest r with t_rom(r) <= T / Sbar
                if t_rom(lo, d, n_rom) > T / Sbar:
                    row[f'rho_S{Sbar:g}'] = None
                    row[f'r_max_S{Sbar:g}'] = 0.0
                    continue
                for _ in range(200):
                    mid = math.sqrt(lo * hi)
                    lo, hi = (mid, hi) if t_rom(mid, d, n_rom) <= T / Sbar else (lo, mid)
                row[f'r_max_S{Sbar:g}'] = lo
                row[f'rho_S{Sbar:g}'] = K / lo
            rho[f'd{d}|nps{n_ps}|nrom{n_rom}|N{N}|{var}'] = row
    out['rho_needed'] = rho
    # ---------- where our configurations sit ----------
    sit = {}
    for cfg in ('1024|R128|LSPG|BE|1', '1024|R384|LSPG|CNR|1', '1024|R512|LSPG|CNR|1', '1024|R384|GAL|CN|1'):
        Nj = a4['rom2d'][cfg]['N_joint_charged']
        r = int(cfg.split('|')[1][1:])
        sit[cfg] = dict(N_joint=Nj, K=(Nj - 1) ** 2, r=r, rho=(Nj - 1) ** 2 / r)
    for lab, v in a4['rom3d'].items():
        r = 512 if lab.startswith('W_') or lab == 'C_old' else 1024
        Nj3 = v['N_joint_charged']
        sit['3D ' + lab] = dict(N_joint=Nj3, K=(Nj3 - 1) ** 3, r=r, rho=(Nj3 - 1) ** 3 / r)
    out['ours'] = sit
    # ---------- crossover in steepness s (2D measured relations; 3D in-range relation, weak) ----------
    def firstN(es, t):
        for N in sorted(es, key=int):
            if es[N] is not None and es[N] <= t:
                return int(N)
        return None
    def powfit(S, Nn):
        p = np.polyfit(np.log(S), np.log(Nn), 1)
        return float(p[0]), float(math.exp(p[1]))
    # accuracy at the step the model charges: E0 2D at dt 0.0125 (20 steps), regimes at their dt (A2, at or below the
    # stability step), E0 3D at dt 0.025 (10 steps)
    rows2 = list(C.J(C.OUT / 'a2_stab.json')['rows'].values())     # 2D: accuracy at the charged, executable step
    rows3 = [r for k, r in C.J(C.OUT / 'a2_e03d_025.json')['rows'].items()]
    st = C.J(C.OUT / 'a2b_stab.json')
    prods = []
    for r in st['rows'].values():
        m = min(max([dt for dt in st['DTS'] if r['res'][f'{N}|{dt}']['stable']] or [0]) for N in st['NS'])
        if 0 < m < max(st['DTS']):
            prods.append(m * r['s'])
    c_variants = dict(c_min=float(min(prods)), c_med=float(np.median(prods)))
    scross = dict(c_dt=c_variants['c_min'], c_variants=c_variants)
    for d, rows_, dt0, n0 in ((2, rows2, 0.0125, 20), (3, rows3, 0.025, 10)):
        for eps in (1e-2, 1e-3):
            allp = [(r['s'], firstN(r['e'], eps)) for r in rows_]
            pts = [(a_, b_) for a_, b_ in allp if b_]
            cens = [a_ for a_, b_ in allp if not b_]
            al, cc = powfit([p_[0] for p_ in pts], [p_[1] for p_ in pts])
            env = max(b_ / (cc * a_ ** al) for a_, b_ in pts)        # worst-case envelope factor over resolved cases
            row = dict(alpha=al, c=cc, envelope=env, n_cases=len(pts), n_censored=len(cens),
                       s_censored=sorted(cens), s_max_data=max(p_[0] for p_ in pts))
            scen = {'fixed50': lambda n: 50, 'fixed20': lambda n: 20, 'equal': lambda n: n,
                    'ratio2.5': lambda n: 2.5 * n}
            for nmod, fac in (('typ', 1.0), ('env', env)):
                for cname, cval in c_variants.items():
                    for sname, nrom_of in scen.items():
                        for rho_ in (None, 1000.0, 100.0, 5.0):
                            sc = None
                            for s_ in np.geomspace(5, 1e6, 600):
                                N = max(8, int(math.ceil(fac * cc * s_ ** al)))
                                n_ps = max(n0, 5 * math.ceil(0.05 * s_ / cval))     # executable: whole steps per output interval
                                r_ = 1e-9 if rho_ is None else (N - 1) ** d / rho_
                                if t_ps(N, d, n_ps) >= 2.0 * t_rom(r_, d, nrom_of(n_ps)):
                                    sc = dict(s=float(s_), N=N, n_ps=n_ps, n_rom=float(nrom_of(n_ps)),
                                              r=None if rho_ is None else r_)
                                    break
                            row[f'{nmod}|{cname}|{sname}|rho{rho_ if rho_ else "inf"}'] = sc
            scross[f'd{d}|eps{eps:g}'] = row
    # ---------- sensitivity of the zero-cost-unknown crossover N to the model's weakest constants ----------
    sens = {}
    base = dict(fp2=dict(fp2), fp3=dict(fp3), fr=dict(fr))
    def crossN(d, n_ps, n_rom, tau0_r, Pmul, alpha_grow):
        floor = n_rom * tau0_r + (alpha_r2 if d == 2 else alpha_r3)
        lat = lat2 if d == 2 else lat3
        Nmax = max(lat)
        f = fp2 if d == 2 else fp3
        for N in (np.arange(8, 4097, 4) if d == 2 else np.arange(8, 513, 2)):
            a_ = np.interp(N, sorted(lat), [lat[k]['alpha'] for k in sorted(lat)])
            if alpha_grow and N > Nmax:
                a_ = lat[Nmax]['alpha'] * ((N - 1) / (Nmax - 1)) ** d
            T = n_ps * (f['tau0'] + w_ps(int(N), d) / (Pmul * f['P'])) + a_
            if T >= 2.0 * floor:
                return int(N)
        return None
    for d, n_ps, n_rom in ((2, 20, 50), (2, 20, 20), (3, 10, 50), (3, 10, 20)):
        k = f'd{d}|nps{n_ps}|nrom{n_rom}'
        sens[k] = dict(base=crossN(d, n_ps, n_rom, fr['tau0'], 1.0, False),
                       tau0_rom_half=crossN(d, n_ps, n_rom, 0.5 * fr['tau0'], 1.0, False),
                       ps_throughput_x2=crossN(d, n_ps, n_rom, fr['tau0'], 2.0, False),
                       ps_alpha_grows=crossN(d, n_ps, n_rom, fr['tau0'], 1.0, True))
    out['sensitivity'] = sens
    out['s_crossover'] = scross
    (C.OUT / 'a7_model.json').write_text(json.dumps(out, indent=0))
    for k, v in chk.items():
        print('check', k, '%.2f ms model / %.2f ms meas = %.2f (asym %s)' % (1e3 * v['model'], 1e3 * v['meas'], v['ratio'], v.get('ratio_asym')))
    for k, v in pred.items():
        print('pred', k, v)
    for k, v in cross.items():
        print('cross', k, v)
    for k, v in rho.items():
        print('rho', k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items()})
    print(sit)
    print(json.dumps(sens, indent=0))
    for k, v in scross.items():
        if not k.startswith('d'):
            print(k, v); continue
        print(k, 'alpha %.2f c %.2f env %.2f n %d cens %d smax %.0f' % (v['alpha'], v['c'], v['envelope'], v['n_cases'], v['n_censored'], v['s_max_data']))
        for kk, vv in v.items():
            if '|' in kk:
                print('   ', kk, None if vv is None else {a: (round(b, 1) if isinstance(b, float) else b) for a, b in vv.items()})


if __name__ == '__main__':
    main()
