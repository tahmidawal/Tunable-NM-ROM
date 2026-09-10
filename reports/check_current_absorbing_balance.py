"""Check the absorbing-wave conserved moment from already captured full fields.

Uses independent NumPy area and boundary sums. No solver or model is executed.
The moment is area_integral(v) + c * boundary_integral(u); its errors are signed
absolute quantities, not relative field errors or a proof of a causal mechanism.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/device04/cluster/out/pilot'
DEST = ROOT/'reports/2026-09-10-absorbing-wave-balance-check.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    result = json.loads((BASE/'result.json').read_text())
    n = max(result['config']['meshes'])
    weights = np.ones(n+1)/n
    weights[[0, -1]] *= .5
    mass = weights[:, None]*weights[None, :]
    sources = {str((BASE/'result.json').relative_to(ROOT)): sha(BASE/'result.json'),
               str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))}

    def load(name):
        path = BASE/name
        actual = sha(path)
        assert actual == result['output_sha256'][name]
        sources[str(path.relative_to(ROOT))] = actual
        return np.load(path)

    def moment(u, v, c):
        interior = np.sum(mass*v, axis=(-2, -1))
        boundary = (u[:, 0, :]+u[:, -1, :])@weights + (u[:, :, 0]+u[:, :, -1])@weights
        return interior+c*boundary

    rows = []
    for ci in result['config']['validation_indices']:
        with load(f'reference_absorbing_{n}_{ci}.npz') as fields:
            speed = float(fields['parameters'][5])
            truth = moment(fields['u'], fields['v'], speed)
        assert float(np.max(abs(truth-truth[0]))) < 1e-10
        for method in (result['config']['control_method'], result['config']['primary_method']):
            with load(f'absorbing_{n}_{ci}_{method}.npz') as fields:
                value = moment(fields['u'], fields['v'], speed)
            rows.append(dict(case=ci, method=method, speed=speed,
                reference_drift=float(np.max(abs(truth-truth[0]))),
                initial_moment_error=float(value[0]-truth[0]),
                max_drift_from_own_initial=float(np.max(abs(value-value[0]))),
                final_moment_error=float(value[-1]-truth[-1]),
                reference_moment=truth.tolist(), rom_moment=value.tolist()))
    payload = dict(provenance=result['provenance'], intervals=n, source_sha256=sources,
        observations=(np.arange(len(rows[0]['reference_moment']))*result['config']['observation_dt']).tolist(),
        method='Independent trapezoid area integral of velocity plus wave speed times trapezoid boundary integral of displacement; both corner contributions retained.',
        interpretation='Signed absolute moment errors. Confirms a missing conservation property in these saved ROM fields; no constrained correction, ablation or causal fraction of the physical error is measured.',
        rows=rows, numerical_solver_executed=False)
    DEST.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows], indent=2))


if __name__ == '__main__':
    main()
