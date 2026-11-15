"""Independently rescore the fixed 2D field and 3D coefficient witnesses for a Burgers mesh."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
KINDS = ('ST', 'S')
TOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scores(field, truth, n0):
    require(field.shape == truth.shape and field.shape[0] == 6, 'field/reference lattice mismatch')
    require(field.dtype == truth.dtype == np.float64 and np.isfinite(n0) and n0 > 0, 'incorrect scoring precision or norm')
    delta = (field[1:] - truth[1:]).reshape(5, -1)
    errors = np.linalg.norm(delta, axis=1) / n0
    require(np.isfinite(errors).all(), 'nonfinite offline scores')
    return np.r_[max(errors), errors]


def compare(values, stored):
    reference = np.array([stored['e'], *stored['e_t']], dtype=np.float64)
    require(reference.shape == (6,) and np.isfinite(reference).all(), 'invalid stored score')
    return float(np.max(np.abs(values-reference)) / max(values[0], 1e-300))


def index(rows, name, names, cases):
    mapping = {(r[name], r['case']): r for r in rows}
    require(len(mapping) == len(rows) and set(mapping) == {(n,c) for n in names for c in cases},
            f'incomplete or duplicate {name} rows')
    return mapping


def corrupted_reference(dim, cohort, case, opened, refstore):
    rel = f'{dim}d/{cohort}/ST/c{case:03d}.npz'
    source = opened.store if opened is not None else refstore._root()
    with tempfile.TemporaryDirectory() as temp:
        new_path = Path(temp)/rel
        new_path.parent.mkdir(parents=True)
        data = bytearray((source/rel).read_bytes())
        data[len(data)//2] ^= 0xff
        new_path.write_bytes(data)
        old = os.environ.get('JCPREF_STORE')
        try:
            if opened is not None:
                test = copy.copy(opened)
                test.store = Path(temp)
                test.bytes(rel)
            else:
                os.environ['JCPREF_STORE'] = temp
                # The vendored loader checks metadata first, then its sibling NPZ.
                new_path.with_suffix('.json').write_bytes((source/rel).with_suffix('.json').read_bytes())
                refstore.load(dim, cohort, case, 'ST')
        except (ValueError, refstore.ReferenceIntegrityError) as exc:
            reason = str(exc).lower()
            return any(s in reason for s in ('mismatch','sha256','checksum'))
        finally:
            if old is None:
                os.environ.pop('JCPREF_STORE', None)
            else:
                os.environ['JCPREF_STORE'] = old
    return False


def run(base, config, output):
    from panel import check_upstream
    check_upstream(base)
    cfg = json.loads(config.read_bytes())
    res_path = output/'result.json'
    result_bytes = res_path.read_bytes()
    rep = json.loads(result_bytes)
    require((output/'COMPLETE').is_file() and rep['complete'] and rep['config'] == cfg, 'incomplete or wrong panel')
    dim, cohort = cfg['dim'], cfg['cohort']
    sys.path.insert(0, str(base/'vendor/tree/experiments/jcp-references'))
    import refstore
    opened = None
    if cohort in ('sealed128', 'sealed64'):
        from published import Opened
        opened = Opened(HERE, config)
        store = opened.refstore(refstore)
    else:
        require(cohort in ('dev6','dev4'), 'unregistered witness population')
        store = refstore
    require(rep['store_version'] == cfg['expected_store_version'] == store.store_version(), 'store mismatch')
    names = [a['name'] for a in cfg['arms']] + (['tensor'] if dim == 3 else [])
    full_solvers = [f['name'] for f in cfg['foms']]
    cases = cfg['cases']
    require(cases == rep['cases'] and len(cases) >= 2, 'incorrect cases')
    rows = index(rep['rows'], 'arm', names, cases)
    foms = index(rep['fom_rows'], 'fom', full_solvers, cases)
    checks = {'G_audit': {}, 'fault_injections': {}}
    report = dict(job_id=rep['job_id'], dim=dim, mesh=rep['mesh'], cohort=cohort, checks=checks,
                  result_sha256=hashlib.sha256(result_bytes).hexdigest())
    fields = output/'fields'
    bank = None
    if dim == 3:
        sys.path.insert(0, str(base/'vendor/tree/experiments/quadrature-burgers3d'))
        import npcore
        model = base/cfg['model']
        require(hashlib.sha256((model/'bank.pkl').read_bytes()).hexdigest() == cfg['expected_model_sha256'],
                'incorrect 3D bank')
        params, rotation = npcore.load_bank(model)
        bank = npcore.bank_np(params, rotation, npcore.lattice65_coords())[:, :cfg['Rp']]
    witness_cases = cfg['audit_cases'] if dim == 2 else cases
    require(set(witness_cases) <= set(cases) and len(witness_cases)>0, 'no witness cases')
    first = witness_cases[0]
    second = next(c for c in cases if c != first)
    counts = {'bank':0, 'full_solver':0}
    worst = 0.0
    perturbed = swapped = False
    with np.load(fields/'coefficients.npz', allow_pickle=False) as coeffs:
        require(set(coeffs.files) == {f'{n}__c{c}' for n in names for c in cases}, 'incomplete coefficient set')
        for case in witness_cases:
            refs = {k:store.load(dim,cohort,case,k) for k in KINDS}
            own = {k:store.reference_status(dim,cohort,case,k) for k in KINDS}
            for name in names + (full_solvers if dim == 2 else []):
                is_bank = name in names
                label = 'bank' if is_bank else 'full_solver'
                record = (rows if is_bank else foms)[name,case]
                if bank is None:
                    stem = 'f257' if is_bank else 'fom'
                    with np.load(fields/f'{stem}_{name}_c{case:03d}.npz', allow_pickle=False) as z:
                        field = z['f257']
                        if is_bank:
                            require(np.array_equal(z['coef'],coeffs[f'{name}__c{case}']), '2D coefficient mismatch')
                            require(hashlib.sha256(field.tobytes()).hexdigest()==record['f257_sha256'], '2D field hash mismatch')
                else:
                    coef = coeffs[f'{name}__c{case}']
                    require(coef.dtype == np.float64 and coef.shape == (6, cfg['Rp']), 'incorrect 3D coefficients')
                    field = coef @ bank.T
                for k in KINDS:
                    truth, meta = refs[k]
                    values = scores(field,truth,meta['judge']['n0'])
                    require(record[k]['eligible'] == own[k]['eligible'], 'incorrect status')
                    worst = max(worst, compare(values,record[k]))
                    counts[label] += 1
                    if case == first and name == cfg['conv_arm'] and k == 'ST':
                        bad = copy.copy(record[k])
                        bad['e'] *= 1 + 1e-6
                        perturbed = compare(values,bad) > TOL
                        other, extra = store.load(dim,cohort,second,k)
                        wrong = scores(field,other,extra['judge']['n0'])
                        swapped = abs(values[0] - wrong[0]) > 1e-3
    expected = len(witness_cases)*len(KINDS)
    require(counts['bank']==expected*len(names) and counts['full_solver']==(expected*len(full_solvers) if dim==2 else 0),
            'missing independent field witnesses')
    checks['G_audit'] = dict(passed=worst<=TOL, compared=sum(counts.values()), max_rel=worst,
                             relative_tolerance=TOL, case_kind_counts=counts, physical_witness_cases=len(witness_cases),
                             FOM_coverage='fixed 2D saved witnesses' if dim==2 else 'no 3D full-solver field saved',
                             times_scored_per_field=5)
    faults = dict(swapped_case=swapped, perturbed_record=perturbed,
                  perturbed_reference_byte=corrupted_reference(dim,cohort,first,opened,refstore))
    checks['fault_injections'] = {**{k:bool(v) for k,v in faults.items()}, 'passed':all(faults.values())}
    report['passed'] = all(check['passed'] for check in checks.values())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','config','output','receipt'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    result = run(a.base.resolve(),a.config.resolve(),a.output.resolve())
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    a.receipt.write_text(data)
    print(data, flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
