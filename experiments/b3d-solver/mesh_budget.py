"""Conservative planning from retained allocation accounting, not deployment timing."""
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
def main():
    cfg=HERE/'configs/mesh01.json';ledger=json.loads((HERE/'JOB-LEDGER.json').read_text())
    measured=next(e['elapsed_seconds'] for e in ledger['entries'] if e['attempt']=='tdiag02')
    rules=sum(q**3*(4097+3*1024)*8 for q in (56,64,80))
    resident=rules+(63**3+127**3)*1024*8+2_000_000_000
    w_ratio=((25+40)*10)/(sum((10,20,40))*4)
    m_ratio=(2052/4097)*.5*sum(q**3 for q in (32,40,48))/sum(q**3 for q in (56,64,80))
    m_steps=(15*4*4)/(sum((10,20,40))*4)
    trajectory=measured*(w_ratio+m_ratio*m_steps)
    overhead=dict(native_basis_setup_and_controls_seconds=600,additional_compile_seconds=480,
                  timing_phases_host_IO_seconds=480,safety_seconds=180)
    spent=sum(e.get('gpu_hours',0.) for e in ledger['entries'])
    out=dict(config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),hard_limit_seconds=3600,memory_guard_bytes=120_000_000_000,
        planned_static_bytes=resident,conservative_extra_scratch_bytes=30_000_000_000,planned_peak_bytes=resident+30_000_000_000,
        source_timing=dict(attempt='tdiag02',elapsed_seconds=measured,kind='same-kernel diagnostic allocation, not deployment cost'),
        W1024_step_work_ratio=w_ratio,M2_contraction_size_ratio=m_ratio,M2_step_work_ratio=m_steps,
        trajectory_seconds_proxy=trajectory,overhead_assumptions=overhead,total_seconds_proxy=trajectory+sum(overhead.values()),
        spent_gpu_hours=spent,full_front_reserved_hours=3,mesh_reserved_hours=1,total_reservation_hours=spent+4,budget_gpu_hours=8,
        scopes=dict(M2_cells=2*2*3*4,W1024_cells=2*(4+4+1+1)*3,solves_per_cell=3),
        interpretation='Planning only; allocator/compiler uncertainty remains. Late120GB guard and60min Slurm cap fail closed. No standalone memory or cross-job performance claim.')
    assert out['planned_peak_bytes']<out['memory_guard_bytes'] and out['total_seconds_proxy']<3600 and spent+4<8
    (HERE/'MESH-BUDGET.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
