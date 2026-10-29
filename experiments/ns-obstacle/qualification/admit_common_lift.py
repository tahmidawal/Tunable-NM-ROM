"""Publish a portable lift admission from genuine full FD1 admission and raw evidence."""
from pathlib import Path
import argparse
import json
import math
import os
import numpy as np
import admit_finite_channel as reference


def validate_lift(folder, review, mesh_sha256, layout_sha256):
    if (review.get('schema') != 'ns-obstacle-FD1-shared-independent-review-v1'
            or review.get('passed') is not True):
        raise ValueError('Independent shared-control review required')
    checks = [r for r in review['checks']
              if r['name'] == 'common_immutable_mesh_layout_and_Stokes_scaling']
    if len(checks) != 1 or checks[0].get('passed') is not True:
        raise ValueError('Reviewed immutable lift/scaling control required')
    evidence = checks[0]['evidence_sha256']
    names = ('control.json', 'lift_scaling_arrays.npz',
             'common_stokes_lift.json', 'common_stokes_lift.npz')
    for name in names:
        matches = [value for key, value in evidence.items() if Path(key).name == name]
        if len(matches) != 1 or reference.sha(folder/name) != matches[0]:
            raise ValueError('Lift evidence differs: ' + name)
    meta = json.loads((folder/'common_stokes_lift.json').read_text())
    if (meta.get('schema') != 'ns-obstacle-common-stokes-v1'
            or meta['mesh_sha256'] != mesh_sha256
            or meta['layout_sha256'] != layout_sha256
            or meta['U'] != 0.2 or meta['D'] != 0.1
            or meta['outlet'] != 'gradient'
            or meta['pressure_gauge'] != 'natural outlet, unshifted'
            or meta['linear_tolerance'] != 1e-10
            or not math.isfinite(meta['base_nu']) or meta['base_nu'] <= 0
            or not math.isfinite(meta['linear_control']['value'])
            or not 0 <= meta['linear_control']['value'] <= meta['linear_tolerance']
            or meta['coefficient_sha256'] != reference.sha(folder/'common_stokes_lift.npz')):
        raise ValueError('Lift mesh/layout/physics/linear control mismatch')
    expected = {'velocity_coefficients': meta['velocity_ndof'],
                'pressure_coefficients_per_nu': meta['pressure_ndof']}
    with np.load(folder/'common_stokes_lift.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise ValueError('Unexpected lift coefficient arrays')
        for name, size in expected.items():
            value = arrays[name]
            if value.shape != (size,) or value.dtype != np.dtype('float64') or not np.isfinite(value).all():
                raise ValueError('Invalid lift coefficient shape/dtype/values')
    return meta


def build(args):
    candidate, contract, independent = [json.loads(p.read_text()) for p in
                                       (args.candidate, args.check_contract, args.reference_review)]
    bindings = dict(candidate_sha256=reference.sha(args.candidate),
                    check_contract_sha256=reference.sha(args.check_contract),
                    publisher_sha256=reference.sha(reference.__file__))
    scope, adopted = reference.validate(candidate, contract, independent, bindings)
    truth = json.loads(args.reference_admission.read_text())
    if (truth.get('schema') != 'ns-obstacle-FD1-reference-admission-v1'
            or truth.get('gate') != reference.GATE or truth.get('passed') is not True
            or not reference.exact_scope(truth.get('qualification_scope'), scope)
            or truth.get('population') != reference.POPULATION
            or truth.get('admitted_references') != adopted
            or truth.get('independent_review_sha256') != reference.sha(args.reference_review)
            or any(truth.get(k) != v for k, v in bindings.items())
            or truth.get('original_domain_gate') != 'failed'):
        raise ValueError('Actual published full-population admission does not match reviewed evidence')
    shared = json.loads(args.shared_review.read_text())
    if shared.get('check_contract_sha256') != bindings['check_contract_sha256']:
        raise ValueError('Shared review uses a different numerical contract')
    mesh = {r['mesh_sha256'] for r in adopted}
    layout = {r['layout_sha256'] for r in adopted}
    if len(mesh) != 1 or len(layout) != 1:
        raise ValueError('Adopted population does not share the lift space')
    folder = args.lift_directory.resolve()
    meta = validate_lift(folder, shared, next(iter(mesh)), next(iter(layout)))
    return dict(schema='ns-obstacle-common-stokes-admission-v1', gate=reference.GATE,
                passed=True, qualification_scope=scope,
                reference_admission_sha256=reference.sha(args.reference_admission),
                review_sha256=reference.sha(args.shared_review),
                directory=os.path.relpath(folder, args.output.resolve().parent),
                manifest_sha256=reference.sha(folder/'common_stokes_lift.json'),
                coefficient_sha256=meta['coefficient_sha256'],
                mesh_sha256=meta['mesh_sha256'], layout_sha256=meta['layout_sha256'],
                publisher_sha256=reference.sha(__file__),
                original_domain_gate='failed', training_released=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-admission', 'candidate', 'check-contract', 'reference-review',
                 'shared-review', 'lift-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    record = build(args)
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
