"""Frozen four-case validation reconstruction; no model fitting or selection."""
from pathlib import Path
import argparse,json,os
import handoff2d
from fe_adapter import sha_file

JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/validation-materialize01'


def validate_release_contract(release):
    fixed={'authorized_stage':'validation_population_materialization','split':'validation',
           'job_directory':JOB,'output_directory':JOB+'/results/population',
           'allocation_seconds':180,'outer_timeout_seconds':165,'outer_kill_grace_seconds':5,
           'debit_from_existing_validation_reservation_seconds':180,
           'remaining_validation_reservation_seconds':3120,
           'all_development_decisions_frozen':True,
           'future_validation_driven_selection_changes_forbidden':True}
    if any(type(release.get(k)) is not type(v) or release[k]!=v for k,v in fixed.items()):
        raise ValueError('exact frozen validation-only paths, flags and cumulative budget required')
    reserve=release.get('other_active_disk_reservation_bytes')
    if type(reserve) is not int or reserve<0:
        raise ValueError('explicit nonnegative concurrent remote storage reservation required')


def verify(plan_path,release_path,inventory_path):
    release=json.loads(Path(release_path).read_text());validate_release_contract(release)
    if release.get('launcher_sha256')!=sha_file(__file__) or release.get('input_inventory_sha256')!=sha_file(inventory_path):
        raise ValueError('exact reviewed launcher and inventory required')
    # Freeze/review/source/admission guards run BEFORE hashing any validation payload.
    plan,_,truth,_=handoff2d.verify(plan_path,release_path)
    if plan['split']!='validation' or [r['Re'] for r in plan['cases']]!=[16,18,22,24]:
        raise ValueError('complete four-case registered validation population required')
    root=Path.cwd().resolve()
    if root!=Path(JOB):raise ValueError('released unique validation directory required')
    inventory=json.loads(Path(inventory_path).read_text());seen=set()
    for row in inventory['files']:
        relative=Path(row['destination']);path=root/relative
        if (relative.is_absolute() or '..' in relative.parts or str(relative) in seen or
            path.is_symlink() or not path.resolve().is_relative_to(root) or
            path.stat().st_size!=row['bytes'] or sha_file(path)!=row['sha256']):
            raise ValueError('exact validation runtime/input differs')
        seen.add(str(relative))
    for row in plan['cases']:
        adopted=handoff2d.adopted_case(truth,row['Re'],row['directory'])
        if adopted['split']!='validation' or adopted['storage_mode']!='coefficient_primary':
            raise ValueError('actual admitted validation primary required')
    return release


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','release','inventory'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    release=verify(a.plan,a.release,a.inventory)
    if not a.verify_only:
        with Path('VALIDATION-HANDOFF-CLAIM.json').open('x') as f:
            json.dump({'release_sha256':sha_file(a.release),'job_id':os.environ.get('SLURM_JOB_ID')},f)
        handoff2d.run(a.plan,a.release,Path(release['output_directory']))
