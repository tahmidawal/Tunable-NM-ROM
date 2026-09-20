"""Generate lane handoff diagnostics from one immutable, audited result."""
import argparse
import hashlib
import json
from pathlib import Path


def diagnose(out):
    record=json.loads((out/'result.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert record['complete'] and summary['complete']
    operators=[]
    for info in record.get('operators',[]):
        path=out/'operators'/info['name']/'curve.json'
        curve=json.loads(path.read_text()) if path.exists() else []
        validation=[row for row in curve if 'validation_worst' in row]
        tail=validation[-10:]
        operators.append(dict(name=info['name'],parameter_count=info['parameter_count'],
            steps=info['steps_completed'],best_step=info['best_step'],seconds=info['seconds'],
            worst_validation_error=info['best_validation_worst'],exit_reason=info['exit_reason'],
            selected_in_last_tenth=info['best_step']>=.9*info['steps_completed'],
            validation_tail=tail,curve_reused_from_previous_attempt=not path.exists(),
            convergence_proved=False))
    return dict(schema='poisson3d-audited-diagnostics-v1',source_commit=record['source_commit'],
        job_id=record['job_id'],gpu=record['gpu'],result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),
        final_cohort_opened=record['final_cohort_opened'],reference=record['reference'],
        bank=record['bank'],heads=record['heads'],operators=operators,
        quadrature=[dict(intervals=mesh['intervals'],certificates=mesh['quadrature']) for mesh in record['meshes']],
        comparisons=summary['rows'],
        interpretation='All comparisons come from this one job; endpoint validation selection does not establish convergence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(diagnose(Path(a.out)),indent=2)+'\n')
