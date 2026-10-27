"""Conservative planning from retained allocation accounting, not deployment timing."""
from pathlib import Path
import hashlib
import json
import subprocess
HERE=Path(__file__).resolve().parent
def main():
    cfg=HERE/'configs/mesh02.json';ledger=json.loads((HERE/'JOB-LEDGER.json').read_text())
    failed_path=HERE/'results/mesh01/raw/output/result.json';failed=json.loads(failed_path.read_text())
    partial_path=failed_path.parent/'mesh_result.json';partial=json.loads(partial_path.read_text());mem=failed['memory']
    measured=next(e['elapsed_seconds'] for e in ledger['entries'] if e['attempt']=='tdiag02')
    rules=sum(q**3*(4097+2*1024)*8 for q in (56,64,80))
    resident=rules+63**3*1024*8+2_000_000_000
    w_ratio=((25+40)*10)/(sum((10,20,40))*4)
    m_ratio=(2052/4097)*.5*sum(q**3 for q in (32,40,48))/sum(q**3 for q in (56,64,80))
    m_steps=(15*4*4)/(sum((10,20,40))*4)
    trajectory=measured*(w_ratio+m_ratio*m_steps)
    overhead=dict(native_basis_setup_and_controls_seconds=600,additional_compile_seconds=480,
                  timing_phases_host_IO_seconds=600,safety_seconds=180)
    spent=sum(e.get('gpu_hours',0.) for e in ledger['entries'])
    out=dict(config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),hard_limit_seconds=5400,memory_guard_bytes=120_000_000_000,
        planned_static_bytes=resident,conservative_extra_scratch_bytes=60_000_000_000,planned_peak_bytes=resident+60_000_000_000,
        memory_failure_source=dict(path=str(failed_path.relative_to(HERE)),sha256=hashlib.sha256(failed_path.read_bytes()).hexdigest(),peak_bytes=mem['peak_bytes_in_use'],current_bytes=mem['bytes_in_use']),
        lifetime_policy='W native129 rows freed after derived operators/initial projection; no W quadrature cold-start tables constructed',
        source_timing=dict(attempt='tdiag02',elapsed_seconds=measured,kind='same-kernel diagnostic allocation, not deployment cost'),
        W1024_step_work_ratio=w_ratio,M2_contraction_size_ratio=m_ratio,M2_step_work_ratio=m_steps,
        trajectory_seconds_proxy=trajectory,overhead_assumptions=overhead,total_seconds_proxy=trajectory+sum(overhead.values()),
        spent_gpu_hours=spent,full_front_reserved_hours=3,mesh_reserved_hours=1.5,total_reservation_hours=spent+4.5,budget_gpu_hours=8,
        reservation_margin_seconds=5400-trajectory-sum(overhead.values()),reservation_amendment='coordinator prospectively added30minutes of slack; scientific scope unchanged',
        scopes=dict(M2_cells=2*2*3*4,W1024_cells=2*(4+4+1+1)*3,solves_per_cell=3),
        interpretation='Planning only; allocator/compiler uncertainty remains. Late120GB guard and90min Slurm cap fail closed. No standalone memory or cross-job performance claim.')
    assert out['planned_peak_bytes']<out['memory_guard_bytes'] and out['total_seconds_proxy']<5400 and spent+4.5<8
    (HERE/'MESH-BUDGET.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))
    original=json.loads(subprocess.check_output(['git','show',failed['source_commit']+':experiments/b3d-solver/MESH-BUDGET.json'],cwd=HERE,text=True))
    last=list(partial['case_results'])[-1];cases=list(partial['case_results'][last]);native=127**3*1024*8;bw=sum(q**3*1024*8 for q in (56,64,80))
    repair=dict(status='memory-lifetime amendment design READY; source/control/job audits required',failed_attempt='mesh01',job_id=failed['job_id'],
        sources=dict(context_result=out['memory_failure_source'],mesh_result=dict(path=str(partial_path.relative_to(HERE)),sha256=hashlib.sha256(partial_path.read_bytes()).hexdigest())),
        failing_query=last,failing_case=int(cases[-1]),recorded_current_bytes=mem['bytes_in_use'],recorded_peak_bytes=mem['peak_bytes_in_use'],guard_bytes=120_000_000_000,
        M2_release_before_W_bytes=next(x['before']['current_bytes'] for x in partial['shared_setup'] if x['bank']=='W1024'),
        proposed_release_native_W129_GT_bytes=native,proposed_avoid_unused_W_Bw_bytes=bw,
        projected_old_peak_minus_removed_bytes=mem['peak_bytes_in_use']-native-bw,measured_scratch_above_current_bytes=mem['peak_bytes_in_use']-mem['bytes_in_use'],
        new_conservative_scratch_assumption_bytes=out['conservative_extra_scratch_bytes'],
        historical_estimate_retraction=f"Original planning peak{original['planned_peak_bytes']/1e9:.2f}GB underpredicted recorded peak{mem['peak_bytes_in_use']/1e9:.2f}GB; it was not a measured standalone method memory claim.",
        preserve=['all numerical kernels and held c0','all trial/test modes, quadrature and predictor equations','all tolerances and120GB guard','native A/L/lambda/norms after GT release'],
        scientific_interpretation='None from incomplete allocation; all fields/logs/costs retained; no cross-job timing.')
    (HERE/'MEMORY-REPAIR.json').write_text(json.dumps(repair,indent=1)+'\n')
if __name__=='__main__':main()
