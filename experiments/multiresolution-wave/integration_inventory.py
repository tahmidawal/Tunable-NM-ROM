"""Build a scoped, content-addressed integration inventory without merging."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

CELL=Path(__file__).resolve().parent
ROOT=CELL.parents[1]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def item(relative,commit,role):
    path=CELL/relative;repo_path=str(path.relative_to(ROOT))
    payload=subprocess.check_output(['git','show',commit+':'+repo_path],cwd=ROOT)
    assert hashlib.sha256(payload).hexdigest()==digest(path)
    return dict(path=repo_path,sha256=digest(path),available_at_commit=commit,role=role)


def generate(final_archive_commit):
    source='3eb9b6586ba019b9bc076d9da21f7a0600fb1f9d'
    groups={
        'original_mathematics_geometry_acceleration':[
            ('acceleration.py','Shared analytic head/Jacobian/curvature; sufficient rank guards, exact-SVD fallback and normal-equation backward diagnostics. Geometry at unchanged timestep is the parity arm.'),
            ('ACCELERATION-DESIGN.md','Analytical bounds, finite-precision margins and screen protocol.'),
            ('acceleration-screen-config.json','Original matched-timestep geometry and separate timestep screen settings.')],
        'selected_decoder_and_timestep':[
            ('nested_head.py','Preserve accepted phase-trained head and append eight fixed linear training directions; forty configuration/eighty phase coordinates, sixty-four weak equations.'),
            ('NESTED40-DESIGN.md','Additive architecture and inclusion/rank protocol.'),
            ('acceleration-confirm-config.json','Frozen final cohort, nested checkpoint/initializer hashes, retained timestep, halfstep checks and same-job FOM controls.'),
            ('acceleration_replay.py','Full-query experiment harness; initial supplied-field fitting, physical velocity, all output costs and saved repetitions. The larger timestep is separate from mathematical parity.'),
            ('freeze_acceleration.py','Generate the bounded final-screen selector before fresh development evaluation.')],
        'training_and_nonselected_evidence':[
            ('head_accuracy.py','Matched fixed-encoder field-only/phase training and bounded capacity training; original optimized-code deployment head remains a distinct control.'),
            ('modal_projection.py','Nonselected larger-state linear-bank evolution with nonlinear output reconstruction and implicit physical velocity; not equivalent latent dynamics.'),
            ('diagnose_training_capacity.py','Training-only projection diagnosis; not an online accuracy guarantee.')],
        'verification_and_transport':[
            ('audit_acceleration.py','Independent full-field, analytic geometry, initial stationarity, reference and provenance auditor.'),
            ('analyze_acceleration.py','Audited source-derived cohort timing/error panels.'),
            ('cluster_acceleration.py','Private-directory staging, source/config preflight, submission and collection dispatch.'),
            ('collect_iterative.py','Verified bounded-memory archive parts and exact remote cleanup.'),
            ('restore_archive.py','Restore the complete full-output archives from ordered parts.')],
    }
    files={name:[item(path,source,role) for path,role in rows] for name,rows in groups.items()}
    checkpoint_commit='2cf42fcd8604177cf6ffe4fa3b81e5f0e3bff34d'
    files['frozen_selected_checkpoint']=[
        item('runs/accel10/cluster/out/pilot/head_trained_nested40.npz',checkpoint_commit,'Selected complete coefficient decoder and fixed training-code candidate library.'),
        item('runs/accel10/cluster/out/pilot/initializer_trained_nested40.npz',checkpoint_commit,'Selected training-only affine initializer; supplied-field fitting still runs online.'),
        item('runs/accel10/selection.json',checkpoint_commit,'Frozen selector binding source result/audit and selected checkpoint/initializer, with screen accuracy miss and initializer limitation.')]
    archives={'accel06':'b55fe991cff4512063c8109dcdcc72d3ed21d060',
        'accel07':'2514d3954d7a8c83429d698f4788f3176a2816fd',
        'accel08':'160b338a7d61a5111e46bff0b168610ae4a77282',
        'accel09':'452cbabf21404a32300b93747e1254117fefa62f',
        'accel10':checkpoint_commit,'accel11':source,'accel12':final_archive_commit}
    runs=[]
    for label,commit in archives.items():
        record=CELL/'runs'/label;sub=json.loads((record/'submission.json').read_text())
        arc=json.loads((record/'ARCHIVE.json').read_text())
        descriptor=item(f'runs/{label}/ARCHIVE.json',commit,'Complete ordered archive descriptor; original full arrays retained in bounded parts.')
        cleanup=json.loads((record/'cleanup.json').read_text());assert cleanup['remote_deleted_and_absence_checked']
        audit_path=record/'audit.json'
        accepted=audit_path.exists() and json.loads(audit_path.read_text())['passed']
        runs.append(dict(attempt=label,job_id=sub['job_id'],scientific_source_commit=sub['source_commit'],
            original_archive_commit=commit,archive=descriptor,archive_bytes=arc['original_bytes'],
            archive_part_count=len(arc['ordered_parts']),numerically_accepted=accepted,exact_remote_removed=True,
            current_audit=item(f'runs/{label}/audit.json',final_archive_commit,'Current strengthened audit binding result, script, source and job.') if accepted else
                item(f'runs/{label}/failure.json',commit,'Preserved operational failure before evaluation-case generation.')))
    final=json.loads((CELL/'runs/accel12/cluster/out/pilot/result.json').read_text())
    output=dict(branch='exp/2026-09-07-mr-wave2d',session_base='321ce49eeb70d97868c07b53e02b0a16accfceaa',
        final_scientific_source_commit=source,final_archive_commit=final_archive_commit,
        merge_performed=False,status='Reviewable research result; selected nested method still misses pooled all-state five-percent accuracy.',
        files=files,runs=runs,
        frozen_bank_and_original_head_lineage=json.loads((CELL/'runs/accel12/cluster/in/ORIGIN.json').read_text()),
        final_result_sha256=digest(CELL/'runs/accel12/cluster/out/pilot/result.json'),
        final_config_sha256=hashlib.sha256(json.dumps(final['config'],sort_keys=True).encode()).hexdigest(),
        integration_boundary='Parity geometry can be reviewed separately from timestep changes and the larger selected nonlinear manifold. Supporting failed/hybrid results are evidence, not selected replacements. The original frozen fresh-wave mathematical source files were not edited.',
        initializer_limitation=final['config']['selection_frozen']['initializer_limitation'])
    (CELL/'integration-inventory.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('files','runs','frozen_bank_and_original_head_lineage')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('final_archive_commit');generate(parser.parse_args().final_archive_commit)
