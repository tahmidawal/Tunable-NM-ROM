"""2D steady bank/POD integration, with data admission kept separate from algebra.

Production callers use load_preparation after a source-bound release. Array-level
constructors also support small synthetic tests; they confer no physical admission.
No NGSolve or JAX is imported by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import gc
import json
from pathlib import Path
import time
import zipfile
import shutil

import numpy as np

from bank import FrozenBank
from geometry import CHANNEL
from interpolation import CoefficientPCHIP
from metrics import field_errors, representation_admission
from quadrature import output_moments, fit_positive, convection_candidate
from solver import solve_steady, jacobian_check
from spaces import weighted_gram, projection_floor, inf_sup
from weak import WeakRule, force_moments


PREFIXES = (2, 4, 8)
TRAIN_RE = 15 + 1.25*np.arange(9)
DEV_RE = (16.875, 23.125)
VALIDATION_RE = (16., 18., 22., 24.)
QUADRATURE_SUPPORTS = (320, 640, 1280, 2560)
OUTPUTS = ('full_velocity_pressure_fields', 'fixed_probes', 'integrated_forces')


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_digest():
    root = Path(__file__).parent
    names = ('pilot2d.py', 'MODEL-CONTRACT.json', 'bank.py', 'geometry.py', 'interpolation.py', 'metrics.py',
             'quadrature.py', 'solver.py', 'spaces.py', 'weak.py', 'timing.py', 'train_bank.py',
             'pilot2d_blocked.py', 'point_store.py', 'quadrature_blocked.py', 'training_evidence.py', 'fom_bridge2d.py', 'frozen_rom2d.py', 'qualification.py', 'FD1-SCOPE-CONTRACT.json', 'capacity_context2d.py')
    return hashlib.sha256(json.dumps({n: sha_file(root/n) for n in names}, sort_keys=True).encode()).hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        digest.update(name.encode()); digest.update(str((a.shape, str(a.dtype))).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _read_arrays(path, max_bytes):
    with zipfile.ZipFile(path) as archive:
        if sum(a.file_size for a in archive.infolist()) > max_bytes:
            raise MemoryError('prepared archive exceeds reviewed uncompressed size cap')
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    for name, a in result.items():
        if name == 'labels':
            if a.dtype.kind not in 'US': raise TypeError('boundary labels must be strings')
        elif name == 'orientation_factors':
            if a.dtype.kind not in 'iu' or not np.isin(a, (-1, 1)).all():
                raise TypeError('orientation factors must be signed unit integers')
        elif a.dtype != np.dtype('float64') or not np.isfinite(a).all():
            raise TypeError(f'prepared {name} must already be finite float64')
    return result


def read_preparation_metadata(directory, release_path):
    """Read production arrays only after an explicit exact-source preparation release."""
    from fe_adapter import CAPS
    folder = Path(directory); release = json.loads(Path(release_path).read_text())
    if (release.get('schema') != 'ns-obstacle-rom-preparation-release-v1' or
            release.get('authorized_stage') != 'G3-2d-steady' or
            release.get('source_sha256') != source_digest()):
        raise RuntimeError('reviewed exact-source 2D preparation release required')
    meta_path = folder/'manifest.json'
    if sha_file(meta_path) != release['adapter_manifest_sha256']:
        raise ValueError('released adapter artifact differs')
    meta = json.loads(meta_path.read_text())
    from qualification import require_qualification
    qualification = require_qualification(meta.get('reference_qualification'))
    if require_qualification(release.get('reference_qualification')) != qualification:
        raise ValueError('released preparation qualification scope differs')
    if (meta.get('split') != 'training' or meta.get('Re') != TRAIN_RE.tolist() or
            meta.get('units', {}).get('already_dimensionless') is not True):
        raise ValueError('exact training-only dimensionless adapter export required')
    if meta['admission_evidence']['reference_admission_sha256'] != release['reference_admission_sha256']:
        raise ValueError('reference admission differs from released preparation')
    for name, digest in meta['coefficient_files'].items():
        if sha_file(folder/name) != digest: raise ValueError(f'prepared artifact hash mismatch: {name}')
    return meta, release


def load_preparation(directory, release_path):
    from fe_adapter import CAPS
    folder = Path(directory)
    meta, release = read_preparation_metadata(directory, release_path)
    from capacity_context2d import consumer_caps
    effective_caps=consumer_caps(meta,release_path,release,CAPS)
    arrays = {name: _read_arrays(folder/(name+'.npz'), CAPS['archive_bytes']) for name in ('boundary','training')}
    if meta.get('volume_storage') == 'point-block-v1':
        from point_store import PointVolumeStore
        arrays['volume'] = PointVolumeStore(folder/'volume', meta['coefficient_files']['volume/manifest.json'])
    else:
        arrays['volume'] = _read_arrays(folder/'volume.npz', CAPS['archive_bytes'])
    if len(arrays['volume']['points_physical']) > effective_caps['volume_points']:
        raise MemoryError('prepared full-rule point cap')
    return arrays, meta, release


def train_from_preparation(directory, release_path, output_path):
    """One gated fixed-recipe fit; no development/validation reference is read."""
    from train_bank import train
    arrays, meta, release = load_preparation(directory, release_path)
    data = arrays['training']
    from qualification import require_qualification
    admission = dict(require_qualification(meta['reference_qualification']), **{'split': 'training',
        'Re': TRAIN_RE.tolist(), 'review_record': release['reference_admission_sha256'],
        'reference_manifest_sha256': meta['manifest_sha256'],
        'array_sha256': meta['training_array_sha256'],
        'source_sha256':release['source_sha256'],'adapter_manifest_sha256':release['adapter_manifest_sha256']})
    # The scoped passed value above is a translation of the already checked, genuine producer
    # admission, never a way to run against smoke or synthetic reference data.
    target = Path(output_path)
    if target.exists(): raise FileExistsError('bank output already exists')
    bank = train(data['points_physical'], data['weights_dimensionless'],
                 data['velocity_corrections'], CHANNEL, admission,
                 evidence_directory=target.with_name(target.name+'.training'))
    bank.save(target)
    return bank


def prepare_probe_arrays(adapter):
    """Exact frozen pole coordinates, evaluated by the admitted FE adapter; no nudge."""
    points = np.array([[.15, .2], [.25, .2]], dtype=np.float64)
    arrays = {'points_physical': points, 'points_dimensionless': points/adapter.D}
    for name, coefficients in adapter.coefficients.items():
        vector = name in ('lift_u', 'pod_velocity', 'supremizers', 'momentum_tests')
        values, gradients = adapter.evaluate(coefficients, vector=vector, points=points)
        arrays[name] = values
        arrays[name+'_gradient_dimensionless'] = gradients*adapter.D
    return arrays


@dataclass
class Model2D:
    kind: str
    prefix: int
    pressure_count: int
    fields: dict
    rule: WeakRule
    operator: object
    force: dict
    stability: dict
    training_coefficients: np.ndarray
    interpolator: CoefficientPCHIP | None
    source: dict

    @property
    def name(self):
        return f'{self.kind}_r{self.prefix}'

    def reconstruct(self, state, reynolds, location='volume', gradients=False):
        f = self.fields[location]; nu_hat = 1./float(reynolds)
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.operator.nv+self.operator.np,):
            raise ValueError('returned reduced coefficient shape mismatch')
        u = f['lift'] + f['velocity']@state[:self.operator.nv]
        p = nu_hat*f['pressure_lift'] + f['pressure']@state[self.operator.nv:]
        result = {'velocity': u, 'pressure': p}
        if gradients:
            result['velocity_gradient'] = f['lift_gradient'] + f['velocity_gradient']@state[:self.operator.nv]
        return result

    def force_output(self, state, reynolds):
        nu_hat = 1./float(reynolds); nv = self.operator.nv
        return (nu_hat*(self.force['lifting_per_nu'] + self.force['velocity_per_nu']@state[:nv])
                + self.force['pressure']@state[nv:])

    def query(self, reynolds, output_contract, operator=None, interpolate=False):
        if not 15 <= reynolds <= 25 or output_contract not in OUTPUTS:
            raise ValueError('query outside frozen family/output contract')
        op = self.operator if operator is None else operator
        start = time.perf_counter()
        if interpolate:
            if self.interpolator is None: raise ValueError('PCHIP is only the POD comparator')
            state = self.interpolator.evaluate(reynolds)
            solved = {'status': 'training_coefficient_interpolation', 'history': [],
                      'residual_evaluations': 0, 'jacobian_evaluations': 0}
        else:
            # Shared Stokes field is the physical initial state for both model families.
            initial = np.zeros(op.nv+op.np)
            initialization_done = time.perf_counter()
            solved = solve_steady(op, 1./reynolds, initial=initial)
            state = solved['state']
        solved_at = time.perf_counter()
        if output_contract == 'integrated_forces':
            outputs = {'force_coefficient': self.force_output(state, reynolds)}
        else:
            location = 'volume' if output_contract == 'full_velocity_pressure_fields' else 'probes'
            outputs = self.reconstruct(state, reynolds, location)
        done = time.perf_counter()
        init_seconds = (solved_at-start) if interpolate else (initialization_done-start)
        return {'output_contract': output_contract, 'outputs': outputs,
            'evidence': {'state': state.copy(), 'reynolds': float(reynolds), 'model': self.name,
                         'operator_formulation': op.formulation, 'history': solved['history']},
            'work': {k: solved[k] for k in ('status', 'residual_evaluations', 'jacobian_evaluations')},
            'component_seconds': {'initialization': init_seconds,
                'coefficient_solve': 0. if interpolate else solved_at-initialization_done,
                'output_reconstruction': done-solved_at, 'total_query': done-start},
            'physical_admission': False}


def build_model(kind, prefix, arrays, metadata, probes, bank=None):
    """Same FE pressure/supremizers/lift/tests for bank and POD; no physical pass flag."""
    if kind not in ('bank', 'POD') or prefix not in PREFIXES:
        raise ValueError('unregistered method or prefix')
    if hasattr(arrays['volume'], 'iter_blocks'):
        from pilot2d_blocked import build_block_model
        return build_block_model(kind, prefix, arrays, metadata, probes, bank)
    start = time.perf_counter(); volume = arrays['volume']; training = arrays['training']
    pressure_count = min(prefix, volume['pressure'].shape[-1])
    if kind == 'bank' and (bank is None or bank.geometry != CHANNEL or bank.rank < prefix):
        raise ValueError('frozen bank for exact declared geometry and prefix required')
    if kind == 'POD' and volume['pod_velocity'].shape[-1] < prefix:
        raise ValueError('prefix unavailable under the frozen POD rank policy')
    fields = {}
    for location, source in [('volume', volume), ('boundary', arrays['boundary']), ('probes', probes)]:
        if kind == 'bank':
            base, derivative = bank.evaluate(source['points_physical'], prefix)
            derivative = derivative*metadata['scales']['D']
        else:
            base = source['pod_velocity'][..., :prefix]
            derivative = source['pod_velocity_gradient_dimensionless'][..., :prefix]
        fields[location] = {'lift': source['lift_u'][..., 0],
            'lift_gradient': source['lift_u_gradient_dimensionless'][..., 0],
            'pressure_lift': source['lift_p_per_nu'][..., 0],
            'velocity': np.concatenate((base, source['supremizers'][..., :pressure_count]), axis=-1),
            'velocity_gradient': np.concatenate((derivative, source['supremizers_gradient_dimensionless'][..., :pressure_count]), axis=-1),
            'pressure': source['pressure'][..., :pressure_count]}
    f = fields['volume']; weights = volume['weights_dimensionless']
    rule = WeakRule(weights, f['lift'], f['lift_gradient'], f['pressure_lift'],
        f['velocity'], f['velocity_gradient'], f['pressure'], volume['momentum_tests'],
        volume['momentum_tests_gradient_dimensionless'], volume['continuity_tests'])
    operator = rule.tensor()
    divergence = np.einsum('n,np,niiv->pv', weights, f['pressure'], f['velocity_gradient'])
    h1 = weighted_gram(f['velocity'], weights) + weighted_gram(f['velocity_gradient'], weights)
    stability = inf_sup(divergence, h1, weighted_gram(f['pressure'], weights),
                        metadata['reference_inf_sup'][str(pressure_count)]['beta'])
    boundary = arrays['boundary']; obstacle = boundary['labels'] == 'obstacle'
    bf = fields['boundary']
    force = force_moments(boundary['weights_dimensionless'][obstacle],
        -boundary['normals_fluid'][obstacle], bf['velocity_gradient'][obstacle],
        bf['pressure'][obstacle], bf['lift_gradient'][obstacle], bf['pressure_lift'][obstacle], 1.)
    velocity_coefficients, _ = projection_floor(f['velocity'], training['velocity_corrections'], weights)
    pressure_coefficients, _ = projection_floor(f['pressure'], training['pressure_corrections'], weights)
    coefficients = np.c_[velocity_coefficients, pressure_coefficients]
    interpolator = None
    if kind == 'POD':
        # The zero-PDE baseline interpolates actual POD coefficients. Supremizers are
        # pressure-coupling enrichments for PDE solves, not extra interpolation modes.
        pod_coefficients, _ = projection_floor(f['velocity'][..., :prefix],
                                               training['velocity_corrections'], weights)
        interpolation_coefficients = np.c_[pod_coefficients, np.zeros((9, pressure_count)), pressure_coefficients]
        interpolator = CoefficientPCHIP(TRAIN_RE, interpolation_coefficients, {'split': 'training',
            'supremizer_coefficients': 'identically zero; interpolation uses POD modes only',
            'adapter_manifest_sha256': metadata['manifest_sha256']})
    return Model2D(kind, prefix, pressure_count, fields, rule, operator, force, stability,
        coefficients, interpolator, {'source_sha256': source_digest(),
        'adapter_manifest_sha256': metadata['manifest_sha256'], 'offline_build_seconds': time.perf_counter()-start,
        'pressure_gauge_shift': False, 'native_data_admission': metadata.get('admission_evidence')})


def galerkin_operator(model):
    """Mandatory square POD control; test space equals the actual enriched trial space."""
    if model.kind != 'POD': raise ValueError('registered mixed Galerkin comparator is POD')
    if hasattr(model.rule, 'galerkin'): return model.rule.galerkin()
    return replace(model.rule, tests=model.rule.velocity,
        test_gradient=model.rule.velocity_gradient, scalar_tests=model.rule.pressure).tensor(formulation='mixed_Galerkin')


def projection_diagnostic(model, reference, boundary, weights):
    f = model.fields['volume']; re = reference['Re']; nu_hat = 1./re
    uc = np.asarray(reference['velocity'])-f['lift']; pc = np.asarray(reference['pressure'])-nu_hat*f['pressure_lift']
    a, ur = projection_floor(f['velocity'], uc[None], weights)
    c, pr = projection_floor(f['pressure'], pc[None], weights)
    state = np.r_[a[0], c[0]]
    row = score_state(model, state, reference, boundary, weights)
    row.update(kind='representation_only_not_PDE_solution', projection_velocity=ur,
               projection_pressure=pr, achieved_state=state.tolist())
    return row


def score_state(model, state, reference, boundary, weights):
    """All scores derive from this invocation's returned state; no pressure shift."""
    re = reference['Re']; full = model.reconstruct(state, re, gradients=True)
    b = model.reconstruct(state, re, 'boundary')
    return score_field_arrays(model.name,full,b,model.force_output(state,re),reference,boundary,weights)


def score_field_arrays(name,full,b,force,reference,boundary,weights):
    """Identical comparison metrics for actual ROM/FOM invocation diagnostic fields."""
    re=reference['Re'];force=np.asarray(force)
    errors = field_errors(full['velocity'], full['pressure'], reference['velocity'], reference['pressure'], weights)
    expected_gradient = np.asarray(reference['velocity_gradient'])
    if expected_gradient.shape != full['velocity_gradient'].shape or not np.isfinite(expected_gradient).all():
        raise ValueError('dimensionless reference velocity gradient required')
    gradient_error = full['velocity_gradient']-expected_gradient
    gradient_numerator = float(np.sqrt(np.einsum('n,nij,nij->', weights, gradient_error, gradient_error)))
    gradient_reference_norm = float(np.sqrt(np.einsum('n,nij,nij->', weights, expected_gradient, expected_gradient)))
    gradient_volume_scale = float(np.sqrt(weights.sum()))
    gradient_denominator = max(gradient_reference_norm, 1e-12*gradient_volume_scale)
    div = np.trace(full['velocity_gradient'], axis1=1, axis2=2)
    divergence = float(np.sqrt(weights@(div*div)/weights.sum()))
    labels = boundary['labels']; bw = boundary['weights_dimensionless']
    flux = bw*np.einsum('ni,ni->n', b['velocity'], boundary['normals_fluid'])
    qin = -float(flux[labels == 'inlet'].sum()); qout = float(flux[labels == 'outlet'].sum())
    mass = abs(qout-qin)/max(abs(qin), 1e-12)
    essential = labels != 'outlet'; target = np.zeros_like(b['velocity'])
    y = boundary['points_physical'][:, 1]; inlet = labels == 'inlet'
    target[inlet, 0] = 6*y[inlet]*(CHANNEL.height-y[inlet])/CHANNEL.height**2
    trace = float(np.max(np.linalg.norm(b['velocity'][essential]-target[essential], axis=1)))
    expected_force = np.asarray(reference['force_coefficient'])
    drag = float(abs(force[0]-expected_force[0])/max(abs(expected_force[0]), 1e-12))
    lift = float(abs(force[1]-expected_force[1]))
    checks = {'velocity': errors['velocity_relative'] <= .01, 'pressure': errors['pressure_scaled'] <= .02,
        'drag': drag <= .02, 'transverse': lift <= .002, 'divergence': divergence <= .005,
        'mass': mass <= .001, 'boundary_trace': trace <= 1e-6}
    return dict(errors, Re=float(re), model=name, force_coefficient=force.tolist(),
        drag_relative=drag, transverse_absolute=lift, divergence=divergence,
        inlet_flux=qin, outlet_flux=qout, mass=mass, boundary_trace=trace,
        velocity_gradient_L2_error=gradient_numerator,
        velocity_gradient_reference_L2_norm=gradient_reference_norm,
        velocity_gradient_relative_denominator=gradient_denominator,
        velocity_gradient_relative=gradient_numerator/gradient_denominator,
        velocity_gradient_scaled=gradient_numerator/gradient_volume_scale,
        velocity_gradient_definition='weighted L2 d(u/U)/d(x/D); scaled by sqrt(dimensionless volume); relative floor 1e-12 sqrt(volume); diagnostic only',
        reference_force_definition=reference['force_definition'],
        target_checks=checks, accuracy_targets_passed=all(checks.values()), physical_admission=False)


def fit_candidates(model, points_dimensionless, supports=QUADRATURE_SUPPORTS, max_bytes=2**30):
    """Fit once from reconstructed training states only; retain every failed attempt."""
    if tuple(supports) != QUADRATURE_SUPPORTS:
        raise ValueError('only the frozen ordered support sequence is allowed')
    blocked = hasattr(model.rule, 'iter_rules')
    if blocked:
        from quadrature_blocked import MemoryMomentBlocks,fit_positive_blocks
        moments = MemoryMomentBlocks(model.rule, model.training_coefficients[:, :model.operator.nv], points_dimensionless)
    else:
        moments = output_moments(model.rule, model.training_coefficients[:, :model.operator.nv],
                                 points_dimensionless, max_bytes=max_bytes)
    results = []
    for support in supports:
        try:
            fit = (fit_positive_blocks if blocked else fit_positive)(moments, model.rule.weights, support)
        except Exception as exc:
            fit = {'passed': False, 'exception': repr(exc)}
        row = {'support_cap': support, 'fit': fit, 'operator': None, 'physical_admission': False}
        if fit['passed']:
            row['operator'] = (model.rule.fitted_convection(model.operator, fit) if blocked else
                               convection_candidate(model.operator, model.rule, fit))
        results.append(row)
    return results


def quadrature_state_check(model, candidate, full_state, candidate_state, reynolds):
    gaps = [model.operator.quadrature_gap(candidate, z, 1./reynolds) for z in (full_state, candidate_state)]
    full = model.reconstruct(full_state, reynolds); reduced = model.reconstruct(candidate_state, reynolds)
    fields = field_errors(reduced['velocity'], reduced['pressure'], full['velocity'], full['pressure'], model.rule.weights)
    a = model.force_output(full_state, reynolds); b = model.force_output(candidate_state, reynolds)
    drag = float(abs(a[0]-b[0])/max(abs(a[0]), 1e-12)); lift = float(abs(a[1]-b[1]))
    passed = (all(g['operator_relative'] <= 1e-4 and g['jacobian_relative'] <= 1e-4 for g in gaps)
              and fields['velocity_relative'] <= 2e-4 and fields['pressure_scaled'] <= 5e-4
              and drag <= 1e-3 and lift <= 1e-4)
    return {'passed': passed, 'operator_at_both_reached_states': gaps,
        'reached_field_change': fields, 'drag_relative_change': drag, 'lift_absolute_change': lift,
        'PDE_solution_admission': False}


def check_population(references, split):
    expected = DEV_RE if split == 'development' else VALIDATION_RE if split == 'validation' else None
    if expected is None or tuple(float(r['Re']) for r in references) != expected:
        raise ValueError('exact declared development or validation population required')
    if any(r.get('split') != split or r.get('reference_admitted') is not True or
           r.get('force_definition') != 'physical_symmetric' for r in references):
        raise ValueError('qualified same-domain references with explicit physical force definition required')


def development_floors(models, references, boundary):
    check_population(references, 'development')
    rows = {m.name: [projection_diagnostic(m, r, boundary, m.rule.weights) for r in references] for m in models}
    maximum = next((m for m in models if m.kind == 'bank' and m.prefix == 8), None)
    if maximum is None: raise ValueError('maximum-prefix bank floor must be recorded before bank solves')
    return {'rows': rows, 'bank_online_gate': representation_admission(rows[maximum.name]),
            'validation_seen': False, 'physical_admission': False}


def evaluate_case(model, reference, boundary, bank_gate=None, fitted=None):
    if model.kind == 'bank' and (bank_gate is None or bank_gate.get('passed') is not True):
        raise RuntimeError('maximum-bank development floor gate closed')
    methods = {'exact_tensor_weak_LS': (model.operator, False)}
    rows = {}
    if model.kind == 'POD':
        methods['coefficient_PCHIP'] = (model.operator, True)
        try:
            methods['mixed_Galerkin'] = (galerkin_operator(model), False)
        except Exception as exc:
            rows['mixed_Galerkin'] = {'failure': repr(exc), 'eligible_accuracy': False}
    if fitted is not None: methods['fitted_convection_weak_LS'] = (fitted, False)
    for name, (operator, interpolation) in methods.items():
        if not interpolation and not model.stability['passed']:
            rows[name] = {'status': 'inf_sup_rejected', 'eligible_accuracy': False,
                          'physical_admission': False}
            continue
        try:
            invocation = model.query(reference['Re'], 'full_velocity_pressure_fields', operator, interpolation)
            state = invocation['evidence']['state']
            scores = score_state(model, state, reference, boundary, model.rule.weights)
            direct = np.r_[model.rule.direct(state, 1./reference['Re'])]
            tensor = model.operator.residual(state, 1./reference['Re'], whiten=False)
            terms = model.operator.terms(state, 1./reference['Re'])
            denominator = max(sum(float(np.linalg.norm(t)) for t in terms.values()), 1e-12)
            algebra_gap = float(np.linalg.norm(direct-tensor)/denominator)
            derivative = jacobian_check(operator, state, 1./reference['Re'], np.ones(len(state)))
            converged = interpolation or invocation['work']['status'] in ('stationary', 'root_converged')
            rows[name] = {'invocation': invocation, 'scores': scores,
                'full_direct_tensor_relative_gap': algebra_gap,
                'full_direct_tensor_denominator': denominator, 'jacobian_check': derivative,
                'eligible_accuracy': bool(converged and scores['accuracy_targets_passed']
                                          and (interpolation or (derivative['passed'] and algebra_gap <= 1e-10))),
                'physical_admission': False}
        except Exception as exc:
            rows[name] = {'failure': repr(exc), 'eligible_accuracy': False, 'physical_admission': False}
    if fitted is not None:
        full_row = rows['exact_tensor_weak_LS']; fitted_row = rows['fitted_convection_weak_LS']
        statuses = {name: row.get('invocation', {}).get('work', {}).get('status', 'missing_invocation')
                    for name, row in [('full', full_row), ('fitted', fitted_row)]}
        both_converged = all(status in ('stationary', 'root_converged') for status in statuses.values())
        diagnostic = {'passed': False, 'solver_statuses': statuses,
                      'both_solves_converged': both_converged}
        if model.stability['passed'] and all('invocation' in row for row in (full_row, fitted_row)):
            diagnostic.update(quadrature_state_check(model, fitted,
                full_row['invocation']['evidence']['state'],
                fitted_row['invocation']['evidence']['state'], reference['Re']))
            diagnostic['numerical_checks_passed'] = diagnostic['passed']
            diagnostic['passed'] = bool(diagnostic['passed'] and both_converged)
        fitted_row['quadrature_reached_state_check'] = diagnostic
        fitted_row['eligible_accuracy'] &= diagnostic['passed']
    return {'model': model.name, 'Re': reference['Re'], 'stability': model.stability, 'methods': rows}


def save_evidence(directory, name, record):
    """Store actual arrays separately; failures/nonfinite scalars remain explicit."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/(name+'.json'); array_path = directory/(name+'.npz')
    if path.exists() or array_path.exists(): raise FileExistsError('evidence already exists')
    arrays = {}; seen = {}; contents = {}; reference_bytes = []
    def byte_view(array):return memoryview(array).cast('B') if array.size else b''
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'biufUS': raise TypeError('unsupported evidence array dtype')
            if id(value) not in seen:
                contiguous=value if value.flags.c_contiguous else np.ascontiguousarray(value)
                digest=hashlib.sha256()
                digest.update(str((contiguous.dtype.str,contiguous.shape)).encode())
                digest.update(byte_view(contiguous))
                address=digest.hexdigest()
                if address in contents:
                    key=contents[address]
                    if (arrays[key].dtype!=contiguous.dtype or arrays[key].shape!=contiguous.shape or
                        byte_view(arrays[key])!=byte_view(contiguous)):
                        raise ValueError('evidence content hash collision')
                else:
                    key=f'a{len(arrays):06d}';contents[address]=key;arrays[key]=contiguous.copy()
                seen[id(value)]=(key,address)
            key,address = seen[id(value)]
            reference_bytes.append(value.nbytes)
            return {'array': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
                    'content_sha256':address}
        if isinstance(value, np.generic): return encode(value.item())
        if isinstance(value, float) and not np.isfinite(value): return {'nonfinite_scalar': str(value)}
        if isinstance(value, dict): return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [encode(v) for v in value]
        return value
    encoded = encode(record)
    unique_bytes=sum(a.nbytes for a in arrays.values())
    # Lossless ZIP can slightly expand incompressible inputs; include header/JSON
    # overhead before writing, rather than relying on hoped-for compression.
    projected_bytes=(unique_bytes+(unique_bytes+99)//100+max(2**20,len(arrays)*2**16)
                     +2*len(json.dumps(encoded,allow_nan=False)))
    reserve_bytes=10*2**30
    if shutil.disk_usage(directory).free<reserve_bytes+projected_bytes:
        raise OSError('evidence write would violate ten-GiB retained disk reserve')
    np.savez_compressed(array_path, **arrays)
    wrapped = {'record': encoded, 'array_file': array_path.name,
               'array_sha256': sha_file(array_path), 'physical_admission': False,
               'array_retention':{'policy':'lossless dtype/shape/bytes content deduplication with byte-equality check',
                   'unique_arrays':len(arrays),'array_references':len(reference_bytes),
                   'unique_expanded_bytes':sum(a.nbytes for a in arrays.values()),
                   'logical_expanded_reference_bytes':sum(reference_bytes),
                   'disk_reserve_bytes':reserve_bytes,'projected_output_bytes':projected_bytes}}
    path.write_text(json.dumps(wrapped, indent=2, allow_nan=False)+'\n')
    return {'record_file': path.name, 'record_sha256': sha_file(path),
            'array_file': array_path.name, 'array_sha256': sha_file(array_path)}


def load_reference_population(path, expected_sha256, split, volume_rule_hash):
    """Open only the declared admitted split on the identical common volume rule."""
    path = Path(path)
    if sha_file(path) != expected_sha256: raise ValueError('released reference population differs')
    record = json.loads(path.read_text())
    from qualification import require_qualification
    qualification = require_qualification(record)
    if (record.get('schema') != 'ns-obstacle-pilot-reference-population-v1'
        or record.get('passed') is not True or record.get('split') != split
        or record.get('volume_rule_sha256') != volume_rule_hash
        or record.get('velocity_gradient_units') != 'd(u/U)/d(x/D)'):
        raise ValueError('qualified common-rule reference population required')
    cases = []
    for row in record['cases']:
        source = path.parent/row['array_file']
        if sha_file(source) != row['array_sha256']: raise ValueError('reference arrays differ')
        arrays = _read_arrays(source, 2**30)
        if set(arrays) != {'velocity', 'pressure', 'velocity_gradient'}: raise ValueError('reference field contract')
        cases.append(dict(row, **arrays, reference_qualification=qualification))
    check_population(cases, split)
    return cases


def model_fingerprint(model):
    arrays = {'training_coefficients': model.training_coefficients}
    arrays.update({'operator_'+name: value for name, value in vars(model.operator).items()
                   if isinstance(value, np.ndarray)})
    arrays.update({'force_'+name: value for name, value in model.force.items()
                   if isinstance(value, np.ndarray)})
    return array_digest(arrays)


def run_accuracy_pilot(arrays, metadata, probes, bank, development, validation_loader,
                       output_directory, fit_quadrature=True, development_only=False, development_identity=None):
    """Actual model construction/solves with development-only quadrature selection.

    validation_loader is called only after all choices are frozen. Construction and
    solver failures are retained per arm; failed bank floors do not remove POD controls.
    Timing/FOM comparison is a separately reviewed invocation of paired_queries below.
    """
    pilot_started = time.perf_counter()
    check_population(development, 'development')
    out = Path(output_directory); out.mkdir(parents=True, exist_ok=False)
    models = {}; failures = {}; floors = {}; records = {}; fitted = {}; choices = {}
    rebuilds = []
    # Fixed ordering contains no performance or validation-data selection.
    for kind in ('bank', 'POD'):
        for prefix in PREFIXES:
            name = f'{kind}_r{prefix}'
            try:
                model = build_model(kind, prefix, arrays, metadata, probes, bank)
                floors[name] = [projection_diagnostic(model, r, arrays['boundary'], model.rule.weights)
                                for r in development]
                models[name] = {'kind': kind, 'prefix': prefix, 'fingerprint': model_fingerprint(model)}
                records[name+'_model'] = save_evidence(out, name+'_model', {
                    'source': model.source, 'stability': model.stability,
                    'unknown_counts': {'velocity': model.operator.nv, 'pressure': model.operator.np},
                    'training_coefficients': model.training_coefficients,
                    'development_projection': floors[name]})
            except Exception as exc:
                failures[name] = {'phase': 'model_construction_or_projection', 'exception': repr(exc)}
            finally:
                model = None
                gc.collect()
    bank_gate = (representation_admission(floors['bank_r8']) if 'bank_r8' in floors else
                 {'passed': False, 'reason': 'maximum-prefix bank construction/projection failed'})
    save_evidence(out, 'development_floors', {'rows': floors, 'bank_online_gate': bank_gate,
                                           'failures': failures, 'validation_seen': False})
    for name, specification in models.items():
        if specification['kind'] == 'bank' and not bank_gate['passed']:
            choices[name] = {'selected_support': None, 'reason': 'bank_development_floor_stop'}
            continue
        try:
            model = build_model(specification['kind'], specification['prefix'], arrays, metadata, probes, bank)
            if model_fingerprint(model) != specification['fingerprint']:
                raise ValueError('offline model rebuild changed frozen operators/coefficient projections')
            rebuilds.append({'model': name, 'phase': 'development',
                             'offline_build_seconds': model.source['offline_build_seconds']})
        except Exception as exc:
            failures[name+'_development_rebuild'] = {'exception': repr(exc)}
            choices[name] = {'selected_support': None, 'reason': 'development_rebuild_failed'}
            model = None
            gc.collect()
            continue
        attempted = []; selected = None
        if fit_quadrature and model.stability['passed']:
            # All fitting inputs are reconstructed training outputs. Development
            # reached states decide eligibility; validation never changes a rule.
            try:
                candidates = fit_candidates(model, arrays['volume']['points_dimensionless'])
                for index, candidate in enumerate(candidates):
                    diagnostics = []
                    if candidate['operator'] is not None:
                        for reference in development:
                            try:
                                diagnostics.append(evaluate_case(model, reference, arrays['boundary'],
                                    bank_gate, candidate['operator']))
                            except Exception as exc:
                                diagnostics.append({'failure': repr(exc)})
                        acceptable = all('methods' in d and d['methods']['fitted_convection_weak_LS'].get(
                            'quadrature_reached_state_check', {}).get('passed') is True
                            and all(d['methods'][method].get('invocation', {}).get('work', {}).get('status')
                                    in ('stationary', 'root_converged')
                                    for method in ('exact_tensor_weak_LS', 'fitted_convection_weak_LS'))
                            for d in diagnostics)
                        if selected is None and acceptable: selected = candidate
                    attempted.append({'support_cap': candidate['support_cap'], 'fit': candidate['fit'],
                                      'development_checks': diagnostics})
                save_evidence(out, name+'_quadrature_candidates', attempted)
            except Exception as exc:
                save_evidence(out, name+'_quadrature_failure', {'exception': repr(exc),
                    'unsampled_control_retained': True})
        fitted[name] = selected['operator'] if selected is not None else None
        choices[name] = {'selected_support': selected['support_cap'] if selected is not None else None,
                         'selection_population': 'development', 'validation_seen': False}
        for reference in development:
            try:
                record = evaluate_case(model, reference, arrays['boundary'], bank_gate, fitted[name])
            except Exception as exc: record = {'failure': repr(exc), 'physical_admission': False}
            key = name+'_development_'+str(reference['Re'])
            records[key] = save_evidence(out, key, record)
        model = None
        gc.collect()
    save_evidence(out, 'frozen_development_choices', {'choices': choices, 'bank_gate': bank_gate,
                                                    'validation_seen': False})
    from weak import WeakOperator
    from frozen_rom2d import input_identity
    frozen = {'development_identity': development_identity, 'input_identity': input_identity(metadata, probes, bank), 'schema': 'ns-obstacle-frozen-rom-development-v1',
              'source_sha256': source_digest(), 'reference_qualification': metadata.get('reference_qualification'),
              'models': models, 'choices': choices, 'bank_gate': bank_gate, 'validation_seen': False,
              'fitted_operators': {name: None if operator is None else {
                  key: getattr(operator, key) for key in WeakOperator.__dataclass_fields__}
                  for name, operator in fitted.items()}}
    frozen_receipt = save_evidence(out, 'frozen_models', frozen)
    if development_only:
        summary = {'source_sha256': source_digest(), 'reference_qualification': metadata.get('reference_qualification'),
                   'records': records, 'construction_failures': failures, 'bank_development_gate': bank_gate,
                   'frozen_choices': choices, 'frozen_models': frozen_receipt, 'model_fingerprints': models,
                   'validation_seen': False, 'timing_comparison_performed': False, 'physical_admission': False,
                   'offline_rebuilds': rebuilds, 'development_elapsed_seconds': time.perf_counter()-pilot_started}
        save_evidence(out, 'development_summary', summary)
        return summary
    validation = validation_loader()
    check_population(validation, 'validation')
    scores = []
    for name, specification in models.items():
        if specification['kind'] == 'bank' and not bank_gate['passed']: continue
        if name not in fitted: continue
        try:
            model = build_model(specification['kind'], specification['prefix'], arrays, metadata, probes, bank)
            if model_fingerprint(model) != specification['fingerprint']:
                raise ValueError('validation rebuild changed frozen operators/coefficient projections')
            rebuilds.append({'model': name, 'phase': 'validation',
                             'offline_build_seconds': model.source['offline_build_seconds']})
        except Exception as exc:
            failures[name+'_validation_rebuild'] = {'exception': repr(exc)}
            model = None
            gc.collect()
            continue
        for reference in validation:
            try:
                record = evaluate_case(model, reference, arrays['boundary'], bank_gate, fitted.get(name))
                for method, row in record['methods'].items():
                    scores.append({'model': name, 'method': method, 'Re': reference['Re'],
                                   'eligible_accuracy': row.get('eligible_accuracy', False),
                                   'scores': row.get('scores')})
            except Exception as exc: record = {'failure': repr(exc), 'physical_admission': False}
            key = name+'_validation_'+str(reference['Re'])
            records[key] = save_evidence(out, key, record)
        model = None
        gc.collect()
    aggregates = {}
    for key in sorted({(r['model'], r['method']) for r in scores}):
        rows = [r for r in scores if (r['model'], r['method']) == key]
        values = [r['scores'] for r in rows if r['scores'] is not None]
        aggregates['/'.join(key)] = {'returned_cases': len(rows), 'expected_cases': len(VALIDATION_RE),
            'failed_or_ineligible_cases': len(VALIDATION_RE)-len(rows)+sum(not r['eligible_accuracy'] for r in rows),
            'all_cases_qualified': len(rows) == len(VALIDATION_RE) and all(r['eligible_accuracy'] for r in rows),
            'metrics': {metric: {'median': float(np.median([v[metric] for v in values])),
                                  'worst': float(max(v[metric] for v in values))}
                        for metric in ('velocity_relative', 'pressure_scaled', 'drag_relative',
                                       'transverse_absolute', 'divergence', 'mass', 'velocity_gradient_scaled',
                                       'velocity_gradient_relative', 'boundary_trace') if values}}
    summary = {'reference_qualification': metadata.get('reference_qualification'),
               'source_sha256': source_digest(), 'records': records, 'construction_failures': failures,
               'bank_development_gate': bank_gate, 'frozen_choices': choices, 'validation': aggregates,
               'no_validation_refit': True, 'physical_admission': False,
               'timing_comparison_performed': False, 'model_memory_policy': 'one full field model at a time',
               'offline_rebuilds': rebuilds, 'model_fingerprints': models,
               'accuracy_pilot_elapsed_seconds': time.perf_counter()-pilot_started}
    save_evidence(out, 'summary', summary)
    return summary


def paired_queries(calls, contract, cores, score_invocation, output_directory):
    """Caller supplies reviewed FOM/ROM callables on one allocation and output contract.

    Every score uses the retained full record from that very timed invocation. The
    independent scoring callback may reconstruct diagnostic fields outside query timing.
    This function never supplies a fabricated FOM or infers matched accuracy from cost.
    """
    from timing import paired_cpu
    if 'FOM' not in calls: raise ValueError('strong reviewed FOM callable is mandatory')
    timing = paired_cpu(calls, contract, cores)
    scored = []
    for row in timing['actual_invocation_outputs']:
        if row['invocation'].get('query_failed'):
            score={'eligible_accuracy':False,'accuracy_targets_passed':False,
                   'query_failed':True,'exception':row['invocation']['exception'],
                   'state_or_output_available':bool('state' in row['invocation'].get('evidence',{})
                                                   or row['invocation'].get('outputs',{}))}
        else:
            try:
                score=score_invocation(row['method'], row['invocation'])
            except Exception as exc:
                score={'eligible_accuracy':False,'accuracy_targets_passed':False,
                       'diagnostic_failed':True,'exception':repr(exc),
                       'actual_invocation_retained':True}
        scored.append({'method': row['method'], 'repetition': row['repetition'],'score':score})
    record = {'timing': timing, 'same_invocation_accuracy': scored, 'physical_admission': False}
    save_evidence(output_directory, 'paired_'+contract, record)
    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--bank', required=True, type=Path)
    parser.add_argument('--probes', required=True, type=Path)
    parser.add_argument('--development', required=True, type=Path)
    parser.add_argument('--validation', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    arrays, metadata, release = load_preparation(args.prepared, args.release)
    for path, key in [(args.bank, 'bank_sha256'), (args.probes, 'probes_sha256')]:
        if sha_file(path) != release[key]: raise ValueError('released '+key+' differs')
    bank = FrozenBank.load(args.bank)
    if (bank.metadata.get('trained') is not True or
        bank.metadata.get('training_array_sha256') != metadata['training_array_sha256']):
        raise ValueError('frozen bank from this exact admitted training population required')
    probes = _read_arrays(args.probes, 2**30)
    rule_hash = array_digest({k: arrays['volume'][k] for k in ('points_dimensionless', 'weights_dimensionless')})
    development = load_reference_population(args.development, release['development_manifest_sha256'], 'development', rule_hash)
    validation_loader = lambda: load_reference_population(args.validation,
        release['validation_manifest_sha256'], 'validation', rule_hash)
    run_accuracy_pilot(arrays, metadata, probes, bank, development, validation_loader, args.out)


if __name__ == '__main__': main()
