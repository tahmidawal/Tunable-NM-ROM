"""Write the validation / certification configs of DESIGN.md §§5-8 for the chosen model.

    python make_configs3.py --model <model dir> [--head <tag>]   (head arms only if DESIGN §4 admits a head)
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESHES = (65, 129, 257)
LADDER = (512, 384, 256, 192, 128, 96, 64)
FOM_GRID = [[dt, nt, lt] for dt in (0.005, 0.01) for nt in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0001) for lt in (0.5, 0.1)] + \
           [[0.025, 0.01, 0.5], [0.025, 0.001, 0.5]]
CERT_DRAWS = [[923811 + i, 8] for i in range(5)] + [[923816, 16]]


def arms(head):
    out = [dict(kind='span', Rp=r, solver='fsc', dt=dt, **({'t32': True} if t32 else {}))
           for r in LADDER for dt in (0.005, 0.01) for t32 in (False, True)]
    if head:
        K = int(head.split('_')[0][1:])
        out += [dict(kind='head', head=head, K=K, Rp=r, solver='fsh', dt=dt) for r in (512, 256) for dt in (0.005, 0.01)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--head')
    a = ap.parse_args()
    md = HERE.parent.parent.parent / a.model if not Path(a.model).is_absolute() else Path(a.model)
    files = ['bank.pkl'] + ([f'head_{a.head}.pkl'] if a.head else [])
    sha = {f: hashlib.sha256((md / f).read_bytes()).hexdigest() for f in files}
    base = dict(audit_lattice=16, gtol=1e-3, trust_fraction=0.05, rho_bar=0.116, ref_ntol=1e-10, ref_ltol=1e-11,
                fom_grid=FOM_GRID, reps=3, arms=arms(a.head), expected_model_sha256=sha,
                model=str(a.model))
    assert len(FOM_GRID) == 26
    for n in MESHES:
        audit = ['span_R512_fsc_dt0.005', 'span_R192_fsc_dt0.01_t32', 'fom_dt0.005_nt0.001_lt0.5'] if n < 257 else \
                ['span_R512_fsc_dt0.005']
        (HERE / 'configs' / f'val_n{n}.json').write_text(json.dumps(dict(
            base, mesh=n, mode='panel', cohort_seed=923801, cohort_count=64, audit_arms=audit), indent=1) + '\n')
        (HERE / 'configs' / f'cert_n{n}.json').write_text(json.dumps(dict(
            base, mesh=n, mode='certify', cert_draws=CERT_DRAWS), indent=1) + '\n')
    print('arms', len(base['arms']), 'fom', len(FOM_GRID), sha)


if __name__ == '__main__':
    main()
