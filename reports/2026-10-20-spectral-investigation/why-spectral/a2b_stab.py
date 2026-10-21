"""A2b. Largest stable ETDRK4 step of the spectral FOM on steep fronts (CPU, JAX x64).

jcp-fom2d measured, on the 38 in-range cases, that dt = 0.0125 is admissible for every N from 32 to 256 and dt = 0.025
is not. This checks whether that holds on steeper fronts (jcp-regimes N0, C1, C2; s up to ~840), which the cost model
needs when it extrapolates: same solver as A2 (E0 pscore.make_ps), N in {64, 128, 256}, dt in {0.0125, 0.00625,
0.003125}; a run is 'stable' if finite with running max |u| <= 1.5 x max|u0|; its error against the regime reference is
recorded (lattice metric). Writes analysis/a2b_stab.json.
"""
import json

import numpy as np

import common as C
import a2_ps as A

NS = (64, 128, 256)
DTS = (0.0125, 0.00625, 0.003125)


def main():
    out = dict(NS=NS, DTS=DTS, rows={})
    for g in ('N0', 'C1', 'C2'):
        for case in C.regime_cases(g):
            ref, meta = C.load_regime(g, case)
            n0 = meta['judge']['n0']
            a0 = float(meta['desc']['a'])
            row = dict(s=float(meta['desc']['a']) / float(meta['desc']['nu']), res={})
            for N in NS:
                run_, S = A.solver(N, 2)
                b0 = A.ic2(meta['desc'], N)
                for dt in DTS:
                    r = run_(np.asarray(b0), float(meta['desc']['nu']), dt, S)
                    ok = bool(r['finite'] and r['umax'] <= 1.5 * a0)
                    e = A.score_reg(r['lat'], ref, n0) if r['finite'] else None
                    row['res'][f'{N}|{dt}'] = dict(stable=ok, e=e, umax=r['umax'])
            out['rows'][f'{g}|{case}'] = row
            print(g, case, round(row['s']), {k: (v['stable'], None if v['e'] is None else f"{v['e']:.1e}") for k, v in row['res'].items()}, flush=True)
    (C.OUT / 'a2b_stab.json').write_text(json.dumps(out, indent=0))


if __name__ == '__main__':
    main()
