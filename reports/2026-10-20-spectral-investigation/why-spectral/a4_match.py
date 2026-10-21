"""A4. Which spectral N reaches our ROM's accuracy, per case and jointly (CPU; reads A1/A2 outputs and lane JSONs).

ROM errors (all conditional E0 estimates, as published):
  * 2D: jcp-qualify FRONT-2D v1 (as frozen into jcp-fom2d `inputs/FRONT-2D.json`), every numerically qualified
    configuration (front members + other qualified) at 1024^2 and 4096^2, per case on the 38 validation cases;
  * 3D: jcp-speed3d m1, the four selected ROMs at L = 384 (median and worst over val64; per-case values are archived on
    the cluster and not read here).
PS errors: A2 (this folder; CPU, dt 0.0025 in 2D and 0.01 in 3D, i.e. spatial error only; gates reproduce the H200 in-job
scores to <= 1e-12 relative).
Outputs per ROM configuration: N_joint = smallest N whose PS median AND worst tie-or-beat the ROM's (the campaign's
joint match), and per case N*_case = smallest N with e_PS(case) <= e_ROM(case); plus fits of N against steepness.
Writes analysis/a4_match.json.
"""
import json

import numpy as np

import common as C


def first_N(es, target):
    for N in sorted(es, key=int):
        v = es[N]
        if v is not None and v <= target:
            return int(N)
    return None


def main():
    ps2 = C.J(C.OUT / 'a2_e02d.json')['rows']
    ps3 = C.J(C.OUT / 'a2_e03d.json')['rows']
    ps2c = C.J(C.OUT / 'a2_e02d_0125.json')['rows']       # at the charged step (20 steps)
    ps3c = C.J(C.OUT / 'a2_e03d_025.json')['rows']        # at the charged step (10 steps)

    def summ(rows, Ns, pref=None):
        o = {}
        for N in Ns:
            v = np.array([np.inf if r['e'][str(N)] is None else r['e'][str(N)] for k, r in rows.items()
                          if pref is None or k.startswith(pref)], float)
            o[N] = dict(median=float(np.median(v)), worst=float(v.max()))
        return o

    def joint(sm, Ns, med, worst):
        return next((N for N in Ns if sm[N]['median'] <= med and sm[N]['worst'] <= worst), None)
    Ns2 = sorted(int(n) for n in next(iter(ps2.values()))['e'])
    Ns3 = sorted(int(n) for n in next(iter(ps3.values()))['e'])
    out = dict(rom2d={}, rom3d={}, ps2_summary={}, ps3_summary={})
    for N in Ns2:
        v = np.array([r['e'][str(N)] for r in ps2.values()], float)
        out['ps2_summary'][N] = dict(median=float(np.median(v)), worst=float(v.max()), modes=(N - 1) ** 2)
    v3 = {k: r for k, r in ps3.items() if k.startswith('val64')}
    for N in Ns3:
        v = np.array([r['e'][str(N)] for r in v3.values()], float)
        out['ps3_summary'][N] = dict(median=float(np.median(v)), worst=float(v.max()), modes=(N - 1) ** 3)
    sm2c = summ(ps2c, Ns2)
    sm3c = summ(ps3c, Ns3, 'val64')
    out['ps2c_summary'] = sm2c
    out['ps3c_summary'] = sm3c
    front = C.J(C.FOM / 'inputs/FRONT-2D.json')
    for mesh in ('1024', '4096'):
        for m in front['meshes'][mesh]['members'] + front['meshes'][mesh]['other_qualified']:
            pc = m['errors']['per_case']
            per = {}
            for case, d in pc.items():
                per[case] = first_N(ps2[case]['e'], d['e'])
            med, worst = m['errors']['median'], m['errors']['worst']
            Nj = None
            for N in Ns2:
                s_ = out['ps2_summary'][N]
                if s_['median'] <= med and s_['worst'] <= worst:
                    Nj = N
                    break
            nstar = np.array([per[c] for c in per], float)
            out['rom2d'][m['config']] = dict(mesh=int(mesh), median=med, worst=worst, N_joint=Nj,
                                             N_joint_charged=joint(sm2c, Ns2, med, worst),
                                             N_case=per, N_case_median=float(np.median(nstar)),
                                             N_case_max=float(np.max(nstar)), t_fused_s=m['timing'].get('t_fused_s'))
    ex = C.J(C.S3D / 'runs/EXPORT.json')['m1']['rom_per_mesh']['384']
    for lab, d in ex.items():
        Nj = None
        for N in Ns3:
            s_ = out['ps3_summary'][N]
            if s_['median'] <= d['median'] and s_['worst'] <= d['worst']:
                Nj = N
                break
        out['rom3d'][lab] = dict(median=d['median'], worst=d['worst'], N_joint=Nj,
                                 N_joint_charged=joint(sm3c, Ns3, d['median'], d['worst']))
    # steepness fits: per case N*_case for two reference configurations, and A1's projection N_proj(eps)
    rows = []
    for case, r in ps2.items():
        rows.append(dict(case=case, s=r['s'], nu=r['nu']))
    out['per_case_s'] = rows
    (C.OUT / 'a4_match.json').write_text(json.dumps(out, indent=0))
    for k, v in out['rom2d'].items():
        if v['mesh'] == 1024:
            print(k, f"{100*v['median']:.3f}% {100*v['worst']:.2f}%", 'Njoint', v['N_joint'], 'charged', v['N_joint_charged'], 'Ncase med/max', v['N_case_median'], v['N_case_max'])
    for k, v in out['rom3d'].items():
        print('3D', k, f"{100*v['median']:.3f}% {100*v['worst']:.2f}%", 'Njoint', v['N_joint'], 'charged', v['N_joint_charged'])


if __name__ == '__main__':
    main()
