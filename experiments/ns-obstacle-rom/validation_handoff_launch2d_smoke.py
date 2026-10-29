"""Metadata-only guard checks; no validation payload or native solver is read."""
from pathlib import Path
import json,tempfile,subprocess
import validation_handoff_launch2d as launch
from build_validation_handoff_launch import build,render_scripts,ROOT


def main():
    outer,body=render_scripts(ROOT/"staging/training-materialize01")
    guard=body.index("--verify-only")
    assert guard<body.index("sha256sum -c SOURCE-SHA256SUMS")<body.index("JOB_PHASE=gpu_preflight")
    assert "training-materialize01" not in body and "development-materialize02" not in body
    assert "165s" in outer and "kill-after=5s" in outer
    for script in (outer,body):subprocess.run(["bash","-n"],input=script,text=True,check=True)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);inventory=root/'inventory.json';inventory.write_text(json.dumps({'files':[{'destination':'DO-NOT-READ.npy','bytes':1,'sha256':'0'*64}]}))
        release={'authorized_stage':'validation_population_materialization','split':'validation',
          'job_directory':launch.JOB,'output_directory':launch.JOB+'/results/population',
          'allocation_seconds':180,'outer_timeout_seconds':165,'outer_kill_grace_seconds':5,
          'debit_from_existing_validation_reservation_seconds':180,'remaining_validation_reservation_seconds':3120,
          'all_development_decisions_frozen':True,'future_validation_driven_selection_changes_forbidden':True,
          'other_active_disk_reservation_bytes':0,'launcher_sha256':launch.sha_file(launch.__file__),
          'input_inventory_sha256':launch.sha_file(inventory)}
        launch.validate_release_contract(release)
        for key,value in [('all_development_decisions_frozen',False),('remaining_validation_reservation_seconds',3300),('other_active_disk_reservation_bytes',True)]:
            altered=dict(release);altered[key]=value
            try:launch.validate_release_contract(altered)
            except ValueError:pass
            else:raise AssertionError('bad guard accepted: '+key)
        path=root/'release.json';path.write_text(json.dumps(release))
        class FreezeRejected(Exception):pass
        original=launch.handoff2d.verify
        def reject(*args):raise FreezeRejected('synthetic rejected freeze, before any validation inventory file')
        launch.handoff2d.verify=reject
        try:
            try:launch.verify(root/'unopened-plan.json',path,inventory)
            except FreezeRejected:pass
            else:raise AssertionError('freeze rejection not propagated')
        finally:launch.handoff2d.verify=original
        review=root/'rejected-review.json';review.write_text(json.dumps({'schema':'rejected-synthetic-metadata','passed':False}))
        try:build(root/'unopened-reference',root/'must-not-exist',root/'missing-frozen',review,0)
        except ValueError:pass
        else:raise AssertionError('builder accepted rejected review')
        assert not (root/'must-not-exist').exists()
    print('PASS: metadata-only validation freeze gates precede payload access; no admission or numerical result generated')


if __name__=='__main__':main()
