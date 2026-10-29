"""Package genuine training inputs without copying their bulk bytes locally.

The output is a review candidate, never an execution release. Every destination
preserves the reviewed training manifest's relative paths under the ROM root.
"""
from pathlib import Path
import argparse, ast, json, shutil
import prepare_packet2d
from prepare2d import preparation_digest
from fe_adapter import read_training_manifest, sha_file

ROOT = Path(__file__).resolve().parent
JOB = '/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/fine-preparation01'


def source_closure():
    pending = ['prepare2d.py', 'preparation_launch2d.py', 'primary_volume.py']
    found = {'MODEL-CONTRACT.json', 'FD1-SCOPE-CONTRACT.json'}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [item.name.replace('.', '/')+'.py' for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                candidates = [node.module.replace('.', '/')+'.py']
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith('.py'):
                candidates = [node.value]
            for candidate in candidates:
                if not Path(candidate).is_absolute() and '..' not in Path(candidate).parts and (ROOT/candidate).is_file():
                    pending.append(candidate)
    return sorted(found)


def build(a):
    manifest = a.training_manifest.resolve()
    # No synthetic/admission placeholder path exists in this builder.
    admitted = read_training_manifest(manifest)
    manifest_rel = str(manifest.relative_to(ROOT))
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    packet = prepare_packet2d.build(manifest, a.measurement, a.factor_reuse, a.native_review,
        a.reconstruction_control, out/'capacity', a.other_active_reservation_bytes)
    amend_path = out/'capacity/CAPACITY-AMENDMENT-CANDIDATE.json'
    amend = json.loads(amend_path.read_text())
    (out/'capacity/evidence').mkdir()
    for key in ('measurement', 'factor_reuse', 'native_control_review'):
        original = Path(amend[key]['path'])
        target = out/'capacity/evidence'/(key+'.json')
        shutil.copyfile(original, target)
        assert sha_file(target) == amend[key]['sha256']
        amend[key]['path'] = 'evidence/'+target.name
    amend_path.write_text(json.dumps(amend, indent=2)+'\n')
    packet['capacity_amendment_sha256'] = sha_file(amend_path)
    packet['training_manifest']['path'] = manifest_rel
    (out/'capacity/PREPARATION-PACKET.json').write_text(json.dumps(packet, indent=2)+'\n')

    template = ROOT/'staging/training-materialize01'
    for name in ('dependencies.py', 'DEPENDENCIES.json'):
        shutil.copyfile(template/name, out/name)
    (out/'REPAIR.json').write_text(json.dumps({'fresh_attempt':'fine-preparation01',
        'purpose':'released actual fine basis preparation and complete exact primary replay; no training'})+'\n')
    outer = (template/'native.sh').read_text().replace('ctol_nsobsrom_materialize', 'ctol_nsobsrom_prepare')
    outer = outer.replace('00:07:30', '00:22:30').replace('435s', '1335s')
    body = (template/'native-body.sh').read_text().replace('training-materialize01', 'fine-preparation01')
    body = body.replace("results/BATCH-RESULT.json", "results/preparation/result.json")
    body = body.replace("'materialization_completed':r.get('passed',False)",
        "'preparation_completed':r.get('status')=='execution_completed_pending_independent_review'")
    body = body.replace('18*2**30', '42*2**30').replace('eight-GiB workspace/archive envelope', '32-GiB workspace/archive envelope')
    body = body.replace('JOB_PHASE=gpu_preflight',
        '"$PY" preparation_launch2d.py --release PREPARATION-RELEASE.json --inventory INPUT-INVENTORY.json --verify-only\nJOB_PHASE=gpu_preflight')
    body = body.replace('"$PY" materialize_batch2d.py --plan BATCH-PLAN.json --release BATCH-RELEASE.json --out results',
        '"$PY" preparation_launch2d.py --release PREPARATION-RELEASE.json --inventory INPUT-INVENTORY.json')
    (out/'native.sh').write_text(outer)
    (out/'native-body.sh').write_text(body)

    paths = {manifest}
    for case in admitted['cases']:
        paths.update(Path(case['resolved_directory'])/name for name in case['files'])
    lift_admission = (manifest.parent/admitted['common_stokes_lift']['admission_path']).resolve()
    lift_folder = Path(admitted['resolved_common_stokes_lift']['archive']).parent
    paths.update([lift_admission, lift_folder/'common_stokes_lift.json', lift_folder/'common_stokes_lift.npz'])
    rows = []
    for path in sorted(paths):
        rows.append({'source':str(path.relative_to(ROOT)), 'destination':str(path.relative_to(ROOT)),
                     'sha256':sha_file(path), 'bytes':path.stat().st_size, 'role':'scientific_input'})
    for name in source_closure():
        path = ROOT/name
        rows.append({'source':name, 'destination':name, 'sha256':sha_file(path),
                     'bytes':path.stat().st_size, 'role':'runtime_source'})
    for path in sorted(out.rglob('*')):
        if path.is_file():
            rows.append({'source':str(path.relative_to(ROOT)), 'destination':str(path.relative_to(out)),
                         'sha256':sha_file(path), 'bytes':path.stat().st_size, 'role':'launch_metadata'})
    assert len({r['destination'] for r in rows}) == len(rows)
    inventory = {'schema':'ns-obstacle-fine-preparation-input-inventory-v1',
        'preparation_source_sha256':preparation_digest(), 'builder_sha256':sha_file(__file__),
        'training_manifest':manifest_rel, 'files':rows, 'bulk_inputs_copied_locally':False,
        'reference_admission_sha256':admitted['reference_admission_sha256']}
    (out/'INPUT-INVENTORY.json').write_text(json.dumps(inventory, indent=2)+'\n')
    plan = {'schema':'ns-obstacle-fine-preparation-launch-candidate-v1', 'execution_released':False,
        'job_directory':JOB, 'allocation_seconds':1350, 'outer_timeout_seconds':1335,
        'outer_kill_grace_seconds':10, 'outer_total_ceiling_seconds':1345,
        'slurm_rounded_limit_seconds':1380, 'launcher_sha256':sha_file(ROOT/'preparation_launch2d.py'),
        'input_inventory_sha256':sha_file(out/'INPUT-INVENTORY.json'),
        'preparation_source_sha256':preparation_digest(), 'training_manifest_sha256':sha_file(manifest),
        'reference_admission_sha256':admitted['reference_admission_sha256'],
        'staged_input_bytes':sum(r['bytes'] for r in rows), 'native_environment_bound_bytes':2**30,
        'remote_workspace_plus_full_archive_envelope_bytes':32*2**30,
        'full_original_and_restored_fields_retained':True, 'local_full_collection_authorized':False,
        'collection_requires_measured_completed_output_and_independent_replay_review':True,
        'bank_training_released':False,
        'required_release_fields':['capacity_amendment','capacity_review','launcher_sha256',
            'input_inventory_sha256','single_execution','allocation_seconds','outer_timeout_seconds',
            'outer_kill_grace_seconds','job_directory'],
        'required_before_staging':['actual training handoff review','exact capacity review',
            'fresh local input/cold-recovery/native staging guards','global remote namespace ledger',
            'single exact coordinator release']}
    (out/'LAUNCH-CANDIDATE.json').write_text(json.dumps(plan, indent=2)+'\n')
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('training-manifest','measurement','factor-reuse','native-review','reconstruction-control','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--other-active-reservation-bytes', type=int, required=True)
    print(json.dumps(build(p.parse_args()), indent=2))
