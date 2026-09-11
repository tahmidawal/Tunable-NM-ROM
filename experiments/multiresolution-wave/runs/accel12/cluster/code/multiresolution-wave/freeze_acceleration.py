"""Freeze the final bounded screen endpoint before fresh development confirmation."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    here=Path(__file__).resolve().parent
    record=here/'runs/accel10';native=record/'cluster/out/pilot'
    result=json.loads((native/'result.json').read_text())
    audit=json.loads((record/'audit.json').read_text())
    panel=json.loads((record/'panel.json').read_text())
    assert audit['passed'] and audit['source_and_job_match']
    assert audit['result_sha256']==panel['source_result_sha256']==sha(native/'result.json')
    assert audit['audit_script_sha256']==sha(here/'audit_acceleration.py')
    candidates=[r for r in panel['panels'] if r['method'] in
                ('trained_phase','trained_phase40','trained_nested40')]
    eligible=[r for r in candidates if r['numerical_gate_pass'] and r['time_refinement_pass']]
    assert len(candidates)==3 and eligible
    chosen=min(eligible,key=lambda r:(max(r['worst_initial_errors_percent'].values()),r['median_gpu_ms']))
    # Keep the tested full-speed setting if the existing halfstep neither passes
    # the target nor improves its complete-cohort worst error.
    half=[r for r in result['accuracy_controls'] if r['method']==chosen['method']]
    half_errors={name:100*max(r['same_grid_discrepancy'][name]['max_initial_normalized'] for r in half)
                 for name in ('displacement','velocity','energy_state')}
    assert max(half_errors.values())>=max(chosen['worst_initial_errors_percent'].values())
    checkpoint=f"head_{chosen['method']}.npz"
    selection=dict(source_result_sha256=sha(native/'result.json'),source_audit_sha256=sha(record/'audit.json'),
        scientific_source_commit=result['provenance']['source_commit'],job_id=result['provenance']['job_id'],
        selected_method=chosen['method'],dt=chosen['setting'],
        internal_configuration_dimension=40,internal_phase_dimension=80,weak_equations=64,
        physical5percent_pass=chosen['all_state_5percent_pass'],
        selected_worst_fixed_initial_errors_percent=chosen['worst_initial_errors_percent'],
        existing_halfstep_worst_fixed_initial_errors_percent=half_errors,
        selection_criterion='Lowest complete-cohort worst fixed-initial all-state error among the three numerically eligible final-screen trained/enriched heads; not a global optimum.',
        frozen_before_confirmation=True,fresh_development_seed=691115,fresh_development_indices=[0,1],
        checkpoint_sha256=sha(native/checkpoint),
        initializer_sha256=sha(native/f"initializer_{chosen['method']}.npz"),
        initializer_limitation='Training-library appended PCA scores are not residual-corrected for existing head contributions; all forty initial coordinates are nevertheless fitted to supplied fields.',
        candidates=candidates)
    assert selection['checkpoint_sha256']==result['output_sha256'][checkpoint]
    (record/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    path=here/'acceleration-confirm-config.json';cfg=json.loads(path.read_text())
    cfg['primary_method']='frozen_mlp32_seed691200'
    cfg['selection']='Settings and selected nested-head checkpoint frozen from the accepted final opened-cohort screen before fresh development. No parameter, epoch or setting selection on confirmation data. Report both cohorts and pooled worst cases; the screen all-state 5% miss remains explicit.'
    cfg['claim_scope']='True nonlinear latent evolution: original 32-coordinate head controls and selected 40-coordinate nested manifold, all in the same 64-function spatial bank. Three meshes, opened two plus separately seeded fresh-development two; final paper cohort sealed. Named CG timestep/tolerance and direct DST controls are paired within the same job.'
    for key in ('arms','refinement_arms'):
        cfg[key][-1]={'method':chosen['method'],'dt':chosen['setting']}
    cfg['head_source_run']='accel10'
    cfg['frozen_head_inputs']={chosen['method']:chosen['method']+'.npz'}
    cfg['selection_frozen']=selection
    cfg.pop('pending_capacity_screen',None)
    path.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({k:v for k,v in selection.items() if k!='candidates'},indent=2))


if __name__=='__main__':freeze()
