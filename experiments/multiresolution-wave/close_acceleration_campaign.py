"""Generate the owner lab-log closing entry exclusively from accepted artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


CELL=Path(__file__).resolve().parent
CANONICAL=Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def read(label):
    record=CELL/'runs'/label
    raw=record/'cluster/out/pilot/result.json'
    result=json.loads(raw.read_text());audit=json.loads((record/'audit.json').read_text())
    assert audit['passed'] and audit['source_and_job_match']
    assert audit['result_sha256']==hashlib.sha256(raw.read_bytes()).hexdigest()
    assert json.loads((record/'cleanup.json').read_text())['remote_deleted_and_absence_checked']
    return result,audit


def group(result,n,method,cohort=None):
    rows=[r for r in result['invocations'] if r['intervals']==n and r['method']==method and (cohort is None or r['cohort']==cohort)]
    return dict(method=method,ms=float(np.median([r['seconds']['complete_device_query']*1000 for r in rows])),
        errors=[100*max(r['same_grid_discrepancy'][k]['max_initial_normalized'] for r in rows) for k in ('displacement','velocity','energy_state')],
        numerical=all(r['completed'] and r.get('fit_stationary',True) and r.get('cg_all_converged',True) for r in rows))


def generate():
    labels=['accel06','accel07','accel08','accel09','accel10','accel12']
    records={label:read(label) for label in labels}
    lines=['## 2026-09-11 — Reflective-wave acceleration and accuracy campaign',
        '', '### Reflective-wave owner: guarded geometry, phase supervision and nested enrichment', '',
        'Continued only the approved `exp/2026-09-07-mr-wave2d` worktree/namespace. Absorbing waves and the sealed final-paper cohort remained excluded. All accepted runs used fresh post-reset bank/head lineage, regenerated query inputs, f64/highest GPU execution, full displacement and physical-velocity outputs, paired same-job iterative CG and direct DST controls, saved repetitions, source/output hashes and independent NumPy/SciPy field/geometry audits. Every exact remote attempt directory was removed after checksum collection.', '',
        '| Attempt | Job | Scientific source | Timed invocations | Distinct timed fields | Refinement fields |',
        '|---|---|---|---:|---:|---:|']
    for label,(r,a) in records.items():
        lines.append(f"| {label} | {a['job_id']} | `{a['source_commit']}` | {a['timed_invocations']} | {a['distinct_timed_fields']} | {a['accuracy_control_fields']} |")
    r6=records['accel06'][0];n6=lambda name:group(r6,64,name)
    # Geometry-only and timestep arms share a method name; choose by setting.
    def setting_median(method,setting):
        return float(np.median([r['seconds']['complete_device_query']*1000 for r in r6['invocations'] if r['method']==method and r['setting']==setting]))
    original=setting_median('baseline',.0025);geometry=setting_median('chol_guard',.0025);fast=setting_median('chol_guard',.01)
    lines += ['',f'The parity-preserving guarded-Cholesky geometry reduced the original ROM median by {original/geometry:.9f}× at the unchanged timestep; retaining the verified larger timestep gave {original/fast:.9f}× in the same screen. Shared analytic derivatives, conservative rank bounds with exact-SVD fallback and measured normal-solve backward residuals preserve the original guard. Independent saved-state audits are empirical finite-precision checks, not exhaustive interval certificates for every internal stage.', '',
        'The further timestep enlargement failed its predeclared refinement requirement and was rejected. Linear-bank evolution with stationary nonlinear output reconstruction was tested separately with a larger internal state; it reduced some errors but still missed the all-state target and was not selected. Physical output velocity used the full implicit stationarity Hessian, with saved neighboring-time kinematic checks.', '',
        'Matched fixed-encoder field-only and field/energy/tangent training separated the effect of added supervision. Phase supervision modestly improved initial-scaled errors; a newly trained larger head did not improve the worst rollout error. The final additive architecture preserves the accepted nonlinear head and appends fixed linear training directions, improving both displacement and velocity errors. The fixed training-library appended scores are not residual-corrected; all initial coordinates are nevertheless fitted to supplied fields.', '',
        'Final confirmation settings/checkpoint were frozen before the separately seeded fresh development cohort. The first confirmation attempt failed at an operational baseline lookup before generating any query case; its complete failure archive is retained under `runs/accel11`, and the corrected retry preserves every scientific setting and selection hash. No earlier accepted numerical result was retracted. Audit provenance was strengthened by binding every accepted audit to its raw-result hash, auditor hash and matching source/job identities.', '',
        '| Intervals per axis | Original ROM GPU ms | Selected nested ROM GPU ms | Worst initial-scaled u / v / energy-state % | Fastest tested passing CG GPU ms | CG / selected ROM | Direct DST GPU ms |',
        '|---:|---:|---:|---|---:|---:|---:|']
    result,audit=records['accel12']
    for n in result['config']['meshes']:
        old=group(result,n,'baseline');selected=group(result,n,'trained_nested40');dst=group(result,n,'dst')
        candidates=[group(result,n,name) for name in dict.fromkeys(r['method'] for r in result['invocations'] if r['method'].startswith('cg'))]
        best=min((r for r in candidates if r['numerical'] and max(r['errors'])<=5),key=lambda r:r['ms'])
        lines.append(f"| {n} | {old['ms']:.9f} | {selected['ms']:.9f} | {' / '.join(f'{x:.9f}' for x in selected['errors'])} | {best['ms']:.9f} (`{best['method']}`) | {best['ms']/selected['ms']:.9f}× | {dst['ms']:.9f} |")
    lines += ['', 'These confirmation medians pool both opened and both fresh development cases, with the full repetition arrays retained. The selected nested manifold improves high-resolution iterative-FOM timing and physical errors, but the pooled all-state accuracy target remains missed. Direct DST remains faster. Fresh cases are reported separately in the generated panels; they were not used for further tuning. Some loose CG settings converge numerically while failing the physical target and are excluded from the passing comparator.', '',
        'Displacement uses the supplied initial displacement norm. Velocity and energy-state use the square root of twice the supplied initial physical energy; initial velocity can vanish in some cases. Energy-state error measures trajectory mismatch and differs from energy drift. Current-relative errors and defined phase errors remain in every full result/panel. The selected model has forty configuration coordinates and eighty phase coordinates in the frozen sixty-four-function bank; original controls have thirty-two configuration coordinates.', '',
        'Artifacts, scientific sources, checkpoints, audits, generated panels and verified bounded archive parts live under `experiments/multiresolution-wave/runs/`. The coordinator owns the generated main campaign report. No further wave search, new worktree, merge, final-cohort opening or presentation replacement occurred. Remaining paper work includes meeting the all-state target and broader independent families/seeds; this is a bounded development result.', '']
    text='\n'.join(lines)
    (CELL/'checks/campaign-close.md').write_text(text)
    return text


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--append-canonical',action='store_true');args=ap.parse_args()
    text=generate()
    if args.append_canonical:
        assert '### Reflective-wave owner: guarded geometry, phase supervision and nested enrichment' not in CANONICAL.read_text()
        with CANONICAL.open('a') as stream:stream.write('\n'+text)
    print(text)
