"""Write the pre-registered panel / certification configs (DESIGN.md sections 4-8, R1) for one model width R."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = False


def arms(R, exact):
    top = R
    span_ladder = [r for r in (512, 384, 256, 192, 128, 96, 64, 32) if r <= R]
    a = [dict(kind='span', rule='tensor', Rp=r) for r in span_ladder]
    a.append(dict(kind='span', rule='tensor', Rp=top, gtol=1e-6))
    a += [dict(kind='span', rule='lat16', Rp=r) for r in sorted({top, 256, 128, 64}, reverse=True)]
    a += [dict(kind='head', rule='tensor', Rp=r, K=32) for r in sorted({top, 256, 128}, reverse=True)]
    a.append(dict(kind='head', rule='tensor', Rp=top, K=64))
    a.append(dict(kind='head', rule='lat16', Rp=top, K=32))
    if exact:
        a.append(dict(kind='span', rule='exact', Rp=top))
    if FS:   # amendment A2: fixed-sweep fast path, two backward-Euler steps
        a += [dict(kind='span', rule='tensor', Rp=r, solver='fs1', dt=dt) for dt in (0.005, 0.01)
              for r in span_ladder if r >= 64]
    return a


def fom_grid():
    g = [[dt, nt, lt] for dt in (0.005, 0.01) for nt in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4) for lt in (0.5, 0.1)]
    return g + [[0.025, 1e-2, 0.5], [0.025, 1e-3, 0.5]]


def base(R, n):
    return dict(mesh=n, lattice=16, audit_lattice=16, ladder=[r for r in (32, 64, 96, 128, 192, 256, 384, 512) if r <= R],
                gtol=1e-3, step_budget=50, trust_fraction=0.05, rho_bar=0.116)


def main(R, tag=''):
    out = HERE / 'configs'
    for n in (33, 65, 129):
        c = base(R, n)
        c.update(mode='panel', arms=arms(R, exact=n in (33, 65)), cohort_seed=923101, cohort_count=16,
                 ref_ntol=1e-10, ref_ltol=1e-11, save_reference_steps=n in (33, 65), fom_grid=fom_grid(), reps=3,
                 audit_arms=[f'span_R{R}_tensor', 'span_R128_tensor', f'head32_R{R}_tensor', 'fom_dt0.005_nt0.001_lt0.5'],
                 parity=dict(arm=f'head32_R{R}_lat16', cases=4) if n in (33, 65) else dict(arm=None))
        (out / f'val_R{R}{tag}_n{n}.json').write_text(json.dumps(c, indent=1) + '\n')
        c = base(R, n)
        c.update(mode='certify', arms=arms(R, exact=False) + [dict(kind='span', rule='lat16', Rp=128, bad=True)],
                 cert_draws=[[923201, 8], [923202, 8], [923203, 8], [923204, 8], [923205, 8], [923206, 16]])
        (out / f'cert_R{R}{tag}_n{n}.json').write_text(json.dumps(c, indent=1) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'A2':
        FS = True
        main(int(sys.argv[1]), tag='A2')
    else:
        main(int(sys.argv[1]))
