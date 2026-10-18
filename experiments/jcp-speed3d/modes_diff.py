"""jcp-speed3d (COMMON-BRIEF §4 / PLAN V4-1): the m1 route-4 ROM uses the deployment policy (per-mesh test-mode list:
the 4R' lowest discrete-Laplacian modes of mesh n, completed to the end of the eigen-shell). This script records, per R'
used in m1 and per mesh n, M(n) and the symmetric difference between the mode-triple set at n and the frozen 3D list
taken at n = 65. Pure NumPy on CPU (no JAX); writes checks/modes-diff.json.

    python modes_diff.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('common_b3d', HERE / 'vendor/quad3d/vendor/burgers3d-span/common.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def mode_set(n, Rp):
    order, lam = C.mode_order(n)
    M = int(C.complete_M(n, 4 * Rp, order, lam))
    k, _ = C.modes(n, M)
    return M, {tuple(map(int, x)) for x in k}


def main():
    cfg = json.loads((HERE / 'configs' / 'm1.json').read_text())
    Rps = sorted({s['Rp'] for s in cfg['settings'] if any(r['setting'] == s['name'] for r in cfg['mesh']['roms'])})
    out = dict(policy='deployment policy (mode set varies with mesh)', frozen_reference_n=65, by_Rp={})
    for Rp in Rps:
        M0, S0 = mode_set(65, Rp)
        d = {}
        for L in cfg['mesh']['Ls']:
            M, S_ = mode_set(L + 1, Rp)
            d[str(L)] = dict(n=L + 1, M=M, only_here=len(S_ - S0), only_in_frozen=len(S0 - S_), symmetric_difference=len(S_ ^ S0))
        out['by_Rp'][str(Rp)] = dict(M_frozen=M0, per_L=d)
    (HERE / 'checks' / 'modes-diff.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
