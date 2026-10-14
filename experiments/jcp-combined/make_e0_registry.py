"""DESIGN A13: write the E0 registry entries `2d_e0` and `3d_e0` into refs.json from the E0 store (read-only), pinned to
the store version. Per case: eta = eps_ref of R_ST (the primary reference; conditional Richardson estimate, not a bound);
per case-kind: outcome and eligibility (K0 flags: a non-eligible case-kind is reported, never ranked).

    python make_e0_registry.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
sys.path.insert(0, str(E0))
import refstore as RS  # noqa: E402

COH2 = {'dev6': 6, 'val32': 32}
COH3 = ('val64', 64, 923801)


def kinds(dim, coh, c, ks):
    out = {}
    for k in ks:
        st = RS.reference_status(dim, coh, c, k)
        out[k] = dict(outcome=st['outcome'], eligible=bool(st['eligible']), eps_ref=st['eps_ref'],
                      crosscheck=st['crosscheck'])
    return out


def main():
    reg = json.loads((HERE / 'refs.json').read_text())
    old2, old3 = reg['2d_first_order'], reg['3d_first_order']
    ver = RS.store_version()
    ck2 = {f'{coh}|{c}': kinds(2, coh, c, ('ST', 'S')) for coh, n in COH2.items() for c in range(n)}
    reg['2d_e0'] = dict(
        kind='2d_e0', loader_dir=str(E0), store_version=ver,
        note=('E0 second-order references (jcp-references, store ' + ver[:12] + '): R_ST = central differences + '
              'Crank-Nicolson (Rannacher start), order-verified, primary; R_S = central + backward Euler at dt0 = 0.005, '
              'matched to BE arms at dt0 ONLY. "TX" is retired: it is aliased to R_ST so the pre-registered TX-based '
              'machinery evaluates the primary E0 reference. eta = eps_ref(R_ST) per case: a conditional Richardson '
              'estimate (asymptotic regime), not a rigorous bound.'),
        tags={'ST': 'ST', 'S': 'S'}, derived={'TX': {'ST': 1.0}},
        bank_sha256=old2['bank_sha256'], rotation_sha256=old2['rotation_sha256'], times=old2['times'], nodes=old2['nodes'],
        normaliser='||R(t0)||_2 of the E0 reference at the nodes (= ||u0||_2 of the physical initial field to round-off, checked per case)',
        eta={k: v['ST']['eps_ref'] for k, v in ck2.items()}, case_kinds=ck2,
        not_rankable=sorted(f'{k}|{kk}' for k, v in ck2.items() for kk, x in v.items() if not x['eligible']))
    ck3 = {f'{COH3[0]}|{c}': kinds(3, COH3[0], c, ('ST',)) for c in range(COH3[1])}
    reg['3d_e0'] = dict(
        kind='3d_e0', loader_dir=str(E0), store_version=ver, cohort=COH3[0], cohort_seed=COH3[2],
        note=('E0 second-order 3D reference R_ST (central + CN, 128/256/512 ladder), cohort val64 = seed 923801. The 3D '
              'space-only R_S (BE at dt 0.005) does not match the 3D ROM step 0.01 and is not used. eta = eps_ref(R_ST) per '
              'case (conditional Richardson estimate).'),
        bank_sha256=old3['bank_sha256'], times=old3['times'], nodes=old3['nodes'],
        normaliser='||R(t0)||_2 of the E0 reference at the nodes (= ||u0||_2 of the physical initial field to round-off, checked per case)',
        eta={str(c): ck3[f'{COH3[0]}|{c}']['ST']['eps_ref'] for c in range(COH3[1])}, case_kinds=ck3,
        not_rankable=sorted(f'{k}|ST' for k, v in ck3.items() if not v['ST']['eligible']))
    (HERE / 'refs.json').write_text(json.dumps(reg, indent=1) + '\n')
    print('2d_e0 not rankable:', reg['2d_e0']['not_rankable'])
    print('3d_e0 not rankable:', reg['3d_e0']['not_rankable'])


if __name__ == '__main__':
    main()
