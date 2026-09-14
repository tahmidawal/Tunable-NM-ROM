"""Generate the concrete, unmerged Burgers accuracy integration inventory."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive-commit',required=True);a=p.parse_args()
    exp=Path(__file__).resolve().parent;root=exp.parents[1]
    def full_commit(value):return subprocess.check_output(['git','rev-parse',value],cwd=root,text=True).strip()
    def load(relative):return json.loads((exp/relative).read_text())
    panels={name:load(f'runs/{name}/PANEL.json') for name in ['accuracy08','accuracy09']}
    for name,panel in panels.items():
        assert panel['audit']['passed'] and load(f'runs/{name}/CLEANUP.json')['remote_absent']
        assert panel['audit']['result_sha256']==digest(exp/f'runs/{name}/archive/out/result.json')
    panel=panels['accuracy09'];result=load('runs/accuracy09/archive/out/result.json')
    rows={r['name']:r for r in panel['rows'] if r['intervals']==1024 and r['cohort']=='all'}
    assert rows['frozen_stationary']['physical_and_numerical_pass']
    rejected=[]
    for model,name,path in [('trained576','trained576_stationary','trained_checkpoint.pkl'),('trained4608','trained4608_stationary','coverage_training/trained_checkpoint.pkl')]:
        row=rows[name]
        assert row['worst_fixed_initial']>rows['frozen_stationary']['worst_fixed_initial'] and not row['physical_pass']
        rejected.append(dict(model=model,status='retain as failed-accuracy research evidence; do not adopt this checkpoint',checkpoint=f'experiments/mr-burgers2d/runs/accuracy09/archive/out/{path}',checkpoint_sha256=digest(exp/f'runs/accuracy09/archive/out/{path}'),fine_mesh_all_cases=row))
    per_case=[]
    for case in range(6):
        comparison={}
        for name in ['frozen_stationary','trained576_stationary','trained4608_stationary']:
            r=next(r for r in result['invocations'] if r['intervals']==1024 and r['name']==name and r['case']==case and r['rep']==0)
            comparison[name]=dict(initial_error=r['error']['fixed_initial_per_time'][0],worst_trajectory_error=r['error']['fixed_initial_max'],final_error=r['error']['fixed_initial_per_time'][-1],peak_output_index=max(range(6),key=lambda i:r['error']['fixed_initial_per_time'][i]))
        per_case.append(dict(case=case,comparisons=comparison))
    inv=dict(status='concrete review inventory; no merge performed',tree=str(root),branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),corrected_base=full_commit('b0e7aaa'),
        accepted_solver=dict(status='candidate for scoped integration after user merge decision',source='experiments/mr-burgers2d/accuracy_paths.py',symbols=['make_stationary_lm','make_rom','make_diagnostics'],scientific_source_commit='cbc4e4d8432763a8ea6a02fac0b433066cd521af',config=result['config']['strict'],config_path='experiments/mr-burgers2d/config-accuracy-coverage.json',rationale='Explicit stationarity stopping with adequate initial-fit and step budgets is independently verified. Keep the original network. This improves numerical reliability; it does not establish a large physical-accuracy gain or a speed improvement.'),
        accepted_checkpoint=dict(path='experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',sha256=result['checkpoint_sha256'],unchanged=True),
        accepted_benchmark_controls=[dict(attempt=name,source_commit=panels[name]['source_commit'],archive_commit=full_commit('fe0bc57' if name=='accuracy08' else a.archive_commit),result=f'experiments/mr-burgers2d/runs/{name}/archive/out/result.json',result_sha256=panels[name]['audit']['result_sha256'],audit=f'experiments/mr-burgers2d/runs/{name}/AUDIT.json',audit_sha256=digest(exp/f'runs/{name}/AUDIT.json'),restore=f'/home/tahmid/Dev/.venv/bin/python experiments/mr-burgers2d/restore_iterative.py experiments/mr-burgers2d/runs/{name}') for name in panels],
        research_only_sources=['experiments/mr-burgers2d/accuracy_replay.py','experiments/mr-burgers2d/accuracy_coverage.py','experiments/mr-burgers2d/accuracy_coverage_paths.py','experiments/mr-burgers2d/config-accuracy-coverage.json','experiments/mr-burgers2d/ACCURACY-COVERAGE-DESIGN.md'],rejected_training_checkpoints=rejected,
        failed_attempt=dict(attempt='accuracy07',scientific_source_commit='cbc4e4d8432763a8ea6a02fac0b433066cd521af',archive_commit='47b783429500b559f6a509448ca11f164183b658',failure='Instrumentation assertion interrupted evaluation; 134 persisted invocations are incomplete coverage. Original failed charged-gradient difference was lost. The frozen training and reference artifacts were explicitly inherited and retimed in accuracy08.',evidence=['runs/accuracy07/FAILURE.json','runs/accuracy07/STATIONARITY-DIAGNOSTIC.json','runs/accuracy07/TRAINING-AUDIT.json']),
        fine_case_initial_vs_rollout=per_case,limitations=['All six cases are development cases; no final paper cases were opened.','Lower mesh continuum/error failures remain in PANEL.json.','Replay preserves old decoded states and is not new PDE truth.','Training procedures also change the initial-code library and refit decoder-output EQ weights.','Initial reconstruction can improve while weak rollout accuracy worsens; attribution to dynamics training versus changed manifold or quadrature requires a separate future experiment.','No additional training arm or merge is authorized in this round.'])
    (exp/'INTEGRATION-accuracy.json').write_text(json.dumps(inv,indent=2)+'\n')
if __name__=='__main__':main()
