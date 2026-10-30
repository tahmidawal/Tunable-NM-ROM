"""Freeze same-objective fixed-hidden output QR/SVD polish; never submit."""
from pathlib import Path
import argparse,json,shutil
from build_bank_fit_launch import bank_sources
from fe_adapter import sha_file
from pilot2d import source_digest
from accuracy_v2.output_polish_launch import JOB
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'method': 'fixed_hidden_output_delta_qr_svd', 'qr_block_points': 256, 'new_adam_steps': 0, 'new_lbfgs_steps': 0, 'supervised_divergence_weight': 1.0, 'traction_weight': 1.0, 'head_seed': 30102026, 'cutoff_policy': 'float64_epsilon_times_max_rows_columns', 'relative_svd_cutoff': 1.6113974599107905e-09}

def build(out,scope,review,warm_review):
    out=Path(out).resolve();out.mkdir(exist_ok=False);review=Path(review)
    prior=json.loads((ROOT/'staging/neural-pod-guided01/PLAN.json').read_text());model=json.loads((ROOT/'checks/neural-pod-guided-traction-model-input-identities.json').read_text())
    inputs=dict(prior['inputs']);inputs.update(parent_bank=model['bank'],parent_receipt=model['model_receipt'],parent_review={'path':JOB+'/MODEL-REVIEW.json','kind':'file','sha256':sha_file(review),'bytes':review.stat().st_size})
    inputs['training_coefficients_source']={'path':inputs['prepared']['path']+'/training.npz','kind':'file',**inputs['prepared']['files']['training.npz']}
    warm_review=Path(warm_review)
    warm=json.loads((ROOT/'checks/diagnostic07-actual-model-descriptors.json').read_text())
    inputs.update(warm_bank=warm['bank'],warm_receipt=warm['model_receipt'],warm_review={'path':JOB+'/WARM-MODEL-REVIEW.json','kind':'file','sha256':sha_file(warm_review),'bytes':warm_review.stat().st_size})
    shutil.copyfile(warm_review,out/'WARM-MODEL-REVIEW.json')
    for src,name in [(scope,'SCOPE-AMENDMENT.json'),(review,'MODEL-REVIEW.json'),(ROOT/'accuracy_v2/CONFIG.json','CONFIG.json')]:shutil.copyfile(src,out/name)
    plan=dict(prior,schema='ns-obstacle-fixed-hidden-output-polish-plan-v1',job_directory=JOB,inputs=inputs,config_sha256=sha_file(out/'CONFIG.json'),scope_amendment_sha256=sha_file(out/'SCOPE-AMENDMENT.json'),refinement_protocol=PROTOCOL,warm_restart=False,new_adam_steps=0,exact_optimizer_continuation_claim=False)
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    shell=(ROOT/'staging/neural-pod-guided01/native.sh').read_text().replace('ctol_nsobsrom_teacher','ctol_nsobsrom_outpolish')
    shell=shell.replace('#SBATCH --gres=gpu:1','#SBATCH --gres=gpu:1\n#SBATCH --constraint=a100|h100|h200')
    body=(ROOT/'staging/neural-pod-guided01/native-body.sh').read_text().replace('/neural-pod-guided01','/neural-pod-guided-output-polish01').replace('-m accuracy_v2.launch','-m accuracy_v2.output_polish_launch').replace('supervised_neural_teacher','fixed_hidden_output_QR_SVD')
    (out/'native.sh').write_text(shell);(out/'native-body.sh').write_text(body)
    names=set(bank_sources())|{str(p.relative_to(ROOT)) for p in (ROOT/'accuracy_v2').glob('*.py')}
    rows=[{'source':str(ROOT/n),'destination':n,'bytes':(ROOT/n).stat().st_size,'sha256':sha_file(ROOT/n)} for n in sorted(names)]
    for name in ('CONFIG.json','SCOPE-AMENDMENT.json','MODEL-REVIEW.json','WARM-MODEL-REVIEW.json','native.sh','native-body.sh'):
        p=out/name;rows.append({'source':str(p),'destination':name,'bytes':p.stat().st_size,'sha256':sha_file(p)})
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'schema':'ns-obstacle-runtime-inventory-v1','files':rows},indent=2)+'\n')
    template=json.loads((ROOT/'checks/output-polish-packet-template.json').read_text())
    bound=template['output_projection'];assert bound['combined_bytes']<2**30
    proposal={'execution_released':False,'allocation_seconds':3600,'outer_timeout_seconds':3585,'outer_kill_grace_seconds':10,'science_deadline_seconds':3480,'consumer_internal_replay_reserve_seconds':90,'science_cutoff_to_outer_TERM_seconds':105,'outer_KILL_grace_seconds':10,'replay_time_guaranteed':False,'protocol':PROTOCOL,'allocation_memory_bytes':64*2**30,'estimated_peak_host_bytes':16*2**30,'raw_plus_archive_reservation_bytes':2**30,'storage_bound':bound,'actual_serialized_selected_full_objective_replay_required':True,'initial_cached_streamed_objective_and_child_gradient_replay_required':True,'device_payload_limit_bytes':2**30,'parameter_space':'520 output coordinates only; other22272 child parameters and embedded parent frozen','child_architecture':[21,64,64,2],'same_objective_as_parent':True,'new_Adam_steps':0,'new_LBFGS_steps':0,'selection':'minimum initial or complete polished candidate original objective','parent_allocation_seconds':json.loads(warm_review.read_text())['allocated_seconds'],'no_cross_job_speedup_claim':True,'collection_separate_measured_release_required':True,'plan_sha256':sha_file(out/'PLAN.json'),'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),'launcher_sha256':sha_file(ROOT/'accuracy_v2/output_polish_launch.py'),'pilot_source_sha256':source_digest(),'template_sha256':sha_file(ROOT/'checks/output-polish-packet-template.json'),'runtime_unmeasured':True,'no_incomplete_candidate_export':True}
    (out/'PROPOSAL.json').write_text(json.dumps(proposal,indent=2)+'\n')
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--scope',type=Path,required=True);a.add_argument('--model-review',type=Path,required=True);a.add_argument('--warm-model-review',type=Path,required=True);a=a.parse_args();print(build(a.output,a.scope,a.model_review,a.warm_model_review))
