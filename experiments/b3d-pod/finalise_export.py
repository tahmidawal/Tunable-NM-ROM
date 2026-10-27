"""Bind the audited stopped fit and complete fallback to the lane handoff."""
from pathlib import Path
import json
from analyse import sha


def main():
    here=Path(__file__).resolve().parent;d=here/'runs/fit32/collected'
    f=here/'checks/fit32/SUMMARY.json';fit=json.loads(f.read_text())
    assert fit['accepted'] is False and fit['rollout_validation_run'] is False
    assert not (here/'runs/k32pilot').exists() and not (here/'runs/k32val').exists()
    assert not (here/'inputs/HR~ecsw~512~k32.npz').exists()
    ledger_path=here/'runs/JOB-LEDGER.json';ledger=json.loads(ledger_path.read_text())
    row=next(j for j in ledger['jobs'] if j['attempt']=='fit32')
    row.update(state='COMPLETED_CONSTRUCTION_REJECTED',elapsed_seconds=fit['elapsed_seconds'],
        spent_gpu_hours=fit['spent_gpu_hours'],exit_code=fit['exit_code'],complete_checkpoint=True,
        checksums_verified=True,results_audit='READY_FAILED_FROZEN_FIT',accepted_support=False)
    ledger['spent_gpu_hours']=sum(j['spent_gpu_hours'] for j in ledger['jobs'])
    ledger['additional_support_jobs_authorized']=False
    ledger['unrun_validation_attempts']={'k32pilot':'construction failed; not run','k32val':'construction failed; not run'}
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    p=here/'runs/EXPORT.json';export=json.loads(p.read_text())
    audit=here/'audits/results-fit32-01.md'
    export['failed_support_extension']=dict(summary_path=str(f),summary_sha256=sha(f),
        result_path=fit['result_path'],result_sha256=fit['result_sha256'],
        terminal_path=fit['terminal_path'],terminal_sha256=fit['terminal_sha256'],
        results_audit_path=str(audit),results_audit_sha256=sha(audit),
        results_audit='READY_FAILED_FROZEN_FIT',job_id=fit['job_id'],source_commit=fit['source_commit'],
        accepted=False,construction_failure_reason=fit['construction_failure_reason'],
        support_target=fit['support_target'],terminal_support=fit['terminal_support'],
        kkt=fit['kkt'],banned=fit['banned'],candidate_count=fit['candidate_count'],
        banned_fraction=fit['banned_fraction'],banned_fraction_limit=fit['banned_fraction_limit'],
        rollout_validation_run=False,HR_admission_evaluated=False,A2_release_not_reached=True,
        source_tar_sha256=sha(d/'source.tar'),collection_manifest_path=str(d/'COLLECTION.sha256'),
        collection_manifest_sha256=sha(d/'COLLECTION.sha256'),resource=dict(
            elapsed_seconds=fit['elapsed_seconds'],gpu_hours=fit['spent_gpu_hours'],
            peak_device_allocated_bytes=fit['peak_device_allocated_bytes'],sacct_path=str(d/'sacct.txt')),
        interpretation=fit['interpretation'])
    matrix=d/'output/contributions.npy'
    export['large_fit_input_retention']=dict(path=str(matrix),sha256=fit['contributions']['matrix_sha256'],
        bytes=matrix.stat().st_size,tracked_in_git=False,preserve_before_worktree_archive=True,
        supporting_manifest_path=str(d/'LOCAL-RETENTION.json'),
        candidate_path=str(d/'output/candidate_nodes.npy'),target_path=str(d/'output/target.npy'))
    export['lane_final_scope']=dict(all_lane_jobs_terminal=True,
        qualified_full_residual_POD_available=True,qualified_hyper_reduced_POD_available=False,
        qualified_HR_comparison='unresolved; frozen supports fail common fields and the one support extension failed construction',
        no_constructor_failure_implies_POD_loss=True,no_k32_rollout_was_evaluated=True,
        further_support_search_stopped=True,fair_schedule_adapter_available_to_solver_lane=True,
        actual_gpu_hours=ledger['spent_gpu_hours'],cap_gpu_hours=ledger['cap_gpu_hours'])
    p.write_text(json.dumps(export,indent=2)+'\n')
    print(json.dumps(export['lane_final_scope'],indent=2))


if __name__=='__main__':main()
