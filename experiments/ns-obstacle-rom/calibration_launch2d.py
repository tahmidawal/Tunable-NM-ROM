"""Bounded common-target screening wrapper; never freezes a matched comparator."""
from pathlib import Path
import argparse, json, os
from fe_adapter import sha_file
import fom_calibrate2d as calibration

JOB = '/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/fom-calibration02'
IDS = ['p3g3h2', 'p4g4h1']


def verify(plan_path, release_path, inventory_path):
    plan_path, release_path, inventory_path = map(Path, (plan_path, release_path, inventory_path))
    p, r, qualification, truth, population, resolved = calibration.verify(plan_path, release_path)
    expected = {'allocation_seconds':2061, 'launcher_hard_wall_seconds':2061,
                'outer_timeout_seconds':2046, 'outer_kill_grace_seconds':10,
                'finalization_reserve_seconds':180, 'job_directory':JOB,
                'output_directory':JOB+'/results/calibration',
                'launcher_sha256':sha_file(__file__),
                'runtime_inventory_sha256':sha_file(inventory_path)}
    if any(type(r.get(k)) is not type(v) or r[k] != v for k,v in expected.items()):
        raise ValueError('exact setup-inclusive restricted screening release required')
    if [c['id'] for c in p['candidates']] != IDS or p.get('selection_frozen') is not False or p.get('matched_accuracy') is not False:
        raise ValueError('only the declared two-candidate common-target screen is allowed')
    inventory = json.loads(inventory_path.read_text()); seen = set()
    for row in inventory['files']:
        name = row['destination']; path = plan_path.parent/name
        if name in seen or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe or duplicate runtime path')
        seen.add(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']:
            raise ValueError('runtime input differs: '+name)
    # These are the actual population consumer's indirect field files, not just its JSON.
    meta = json.loads(population.read_text())
    if [row['Re'] for row in meta['cases']] != [16.875,23.125]:
        raise ValueError('complete development-only population required')
    for row in meta['cases']:
        path = population.parent/row['array_file']
        if path.is_symlink() or not path.resolve().is_relative_to(population.parent.resolve()) or sha_file(path) != row['array_sha256']:
            raise ValueError('development field identity differs')
    for candidate, files, cfg, layout in resolved:
        if candidate['field_rule_mode'] != 'certified_requested_points':
            raise ValueError('coarse candidates must certify every requested point')
        if any(cfg.get(k)!=v for k,v in {'nonlinear_relative_tolerance':1e-10,
                'linear_relative_tolerance':1e-10,'newton_max':30,
                'assembly_bonus_order':6,'diagnostic_order':12}.items()):
            raise ValueError('fixed solver and quadrature policy differs')
        # Existing eligibility loader checks every quartet array and configuration binding.
        from fom_comparator2d import load_native_modules
        modules = load_native_modules()
        modules['candidate_lift'].CandidateStokesLift.from_eligibility(files['eligibility'],
            sha_file(files['mesh']), layout['sha256'], cfg['U'], cfg['D'], cfg,
            expected_truth_admission_sha256=sha_file(truth))
    return p, r


def complete(ledger, output):
    expected = {(c,re,contract) for c in IDS for re in (16.875,23.125) for contract in calibration.OUTPUTS}
    rows = ledger.get('blocks',[])
    good = not ledger.get('construction_failures') and len(rows)==len(expected)
    good = good and {(r['candidate'],r['Re'],r['contract']) for r in rows}==expected
    for row in rows:
        path = Path(row['evidence']['path'])
        if sha_file(path) != row['evidence']['sha256']:raise ValueError('scoring evidence changed')
        record = json.loads(path.read_text())['record']
        scores = record['same_invocation_scores']
        good = good and len(row['elapsed_seconds'])==7 and len(scores)==7
        good = good and not any(s.get('diagnostics_unavailable_budget') for s in scores)
        good = good and record.get('readout_control') is not None
        good = good and 'TimeoutError' not in str((record.get('readout_control') or {}).get('exception',''))
    receipt = {'complete_planned_workload':bool(good), 'selection_frozen':False,
        'matched_accuracy':False, 'validation_used':False,
        'tested_candidates':IDS, 'untested_candidates':['p3g3h1','p3g3h3','p4g4h2','p4g4h3'],
        'scope':'Common-target screening among two tested mesh/order configurations at fixed nonlinear tolerance 1e-10; stopping-tolerance optimization untested.',
        'partial_timeout_is_not_accepted_calibration':True}
    (Path(output)/'WORKLOAD-COMPLETENESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not good:raise RuntimeError('incomplete declared screening workload; raw failures retained')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan','release','inventory'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true'); args=parser.parse_args()
    p,r=verify(args.plan,args.release,args.inventory)
    if args.verify_only:
        print(json.dumps({'verified':True,'native_solver_executed':False,'candidates':IDS}));return
    if Path.cwd().resolve()!=Path(JOB) or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('released unique cluster allocation required')
    ledger=calibration.run(args.plan,args.release,r['output_directory'])
    complete(ledger,r['output_directory'])


if __name__=='__main__':main()
