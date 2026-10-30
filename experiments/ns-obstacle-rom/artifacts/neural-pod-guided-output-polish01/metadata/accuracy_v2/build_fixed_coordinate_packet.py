"""Freeze independent residual heads from the reviewed traction parent; never submit."""
from pathlib import Path
import argparse,json,shutil
from build_bank_fit_launch import bank_sources
from fe_adapter import sha_file
from pilot2d import source_digest
from accuracy_v2.fixed_coordinate_launch import JOB
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'max_iterations':10000,'checkpoint_every':1000,'history_size':10,'line_search_steps':20,'gradient_tolerance':1e-10,'warm_restart':True,'new_adam_steps':0,'supervised_divergence_weight':1.0,'traction_weight':1.0,'head_seed':30102026,'hidden_coordinate_scale':100.0,'output_coordinate_scale':1.0}

def build(out,scope,review,warm_review,candidate):
    out=Path(out).resolve();out.mkdir(exist_ok=False);review=Path(review)
    prior=json.loads((ROOT/'staging/neural-pod-guided01/PLAN.json').read_text());model=json.loads((ROOT/'checks/neural-pod-guided-traction-model-input-identities.json').read_text())
    inputs=dict(prior['inputs']);inputs.update(parent_bank=model['bank'],parent_receipt=model['model_receipt'],parent_review={'path':JOB+'/MODEL-REVIEW.json','kind':'file','sha256':sha_file(review),'bytes':review.stat().st_size})
    inputs['training_coefficients_source']={'path':inputs['prepared']['path']+'/training.npz','kind':'file',**inputs['prepared']['files']['training.npz']}
    warm_review=Path(warm_review);candidate=Path(candidate)
    warm=json.loads((ROOT/'checks/diagnostic06-actual-model-descriptors.json').read_text())
    inputs.update(warm_bank=warm['bank'],warm_receipt=warm['model_receipt'],warm_review={'path':JOB+'/WARM-MODEL-REVIEW.json','kind':'file','sha256':sha_file(warm_review),'bytes':warm_review.stat().st_size})
    shutil.copyfile(warm_review,out/'WARM-MODEL-REVIEW.json');shutil.copyfile(candidate,out/'COORDINATE-CANDIDATE.json')
    for src,name in [(scope,'SCOPE-AMENDMENT.json'),(review,'MODEL-REVIEW.json'),(ROOT/'accuracy_v2/CONFIG.json','CONFIG.json')]:shutil.copyfile(src,out/name)
    plan=dict(prior,schema='ns-obstacle-fixed-coordinate-residual-refinement-plan-v1',job_directory=JOB,inputs=inputs,config_sha256=sha_file(out/'CONFIG.json'),scope_amendment_sha256=sha_file(out/'SCOPE-AMENDMENT.json'),coordinate_candidate_sha256=sha_file(out/'COORDINATE-CANDIDATE.json'),refinement_protocol=PROTOCOL,warm_restart=True,new_adam_steps=0,exact_optimizer_continuation_claim=False)
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    shell=(ROOT/'staging/neural-pod-guided01/native.sh').read_text().replace('ctol_nsobsrom_teacher','ctol_nsobsrom_coordinate')
    shell=shell.replace('#SBATCH --gres=gpu:1','#SBATCH --gres=gpu:1\n#SBATCH --constraint=a100|h100|h200')
    body=(ROOT/'staging/neural-pod-guided01/native-body.sh').read_text().replace('/neural-pod-guided01','/neural-pod-guided-coordinate01').replace('-m accuracy_v2.launch','-m accuracy_v2.fixed_coordinate_launch').replace('supervised_neural_teacher','frozen_parent_independent_residual_heads_LBFGS')
    (out/'native.sh').write_text(shell);(out/'native-body.sh').write_text(body)
    names=set(bank_sources())|{str(p.relative_to(ROOT)) for p in (ROOT/'accuracy_v2').glob('*.py')}
    rows=[{'source':str(ROOT/n),'destination':n,'bytes':(ROOT/n).stat().st_size,'sha256':sha_file(ROOT/n)} for n in sorted(names)]
    for name in ('CONFIG.json','SCOPE-AMENDMENT.json','MODEL-REVIEW.json','WARM-MODEL-REVIEW.json','COORDINATE-CANDIDATE.json','native.sh','native-body.sh'):
        p=out/name;rows.append({'source':str(p),'destination':name,'bytes':p.stat().st_size,'sha256':sha_file(p)})
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'schema':'ns-obstacle-runtime-inventory-v1','files':rows},indent=2)+'\n')
    vectors=22;parameters=22792;saves=12;array_upper=saves*(vectors*parameters*8+65536);logs_upper=(200000+10000)*6144+(2*(200000+3))*1024;raw_upper=array_upper+logs_upper+64*2**20;archive_upper=raw_upper+raw_upper//50+64*2**20
    assert raw_upper+archive_upper<4*2**30
    (out/'PROPOSAL.json').write_text(json.dumps({'execution_released':False,'allocation_seconds':3600,'outer_timeout_seconds':3585,'outer_kill_grace_seconds':10,'science_deadline_seconds':3480,'final_replay_retention_margin_seconds':120,'protocol':PROTOCOL,'allocation_memory_bytes':64*2**30,'estimated_peak_host_bytes':16*2**30,'raw_plus_archive_reservation_bytes':4*2**30,'storage_bound':{'max_state_saves':saves,'vectors_per_full_state':vectors,'parameters':parameters,'state_arrays_plus_container_bytes':array_upper,'fixed_schema_scalar_record_bound_bytes':6144,'max_evaluation_event_rows':400006,'evaluation_event_record_bound_bytes':1024,'max_trial_rows':200000,'max_accepted_rows':10000,'scalar_jsonl_bound_bytes':logs_upper,'raw_upper_bytes':raw_upper,'archive_without_compression_credit':archive_upper,'combined_upper_bytes':raw_upper+archive_upper},'all_intermediate_vector_replay_available':False,'actual_serialized_selected_full_objective_replay_required':True,'initial_cached_streamed_objective_and_child_gradient_replay_required':True,'device_payload_limit_bytes':2**30,'parameter_space':'four child heads only; frozen parent embedded separately','child_architecture':[21,64,64,2],'same_objective_as_parent':True,'supervised_divergence_weight':1.0,'new_Adam_steps':0,'selection':'minimum accepted combined objective','parent_allocation_seconds':json.loads(warm_review.read_text())['allocated_seconds'],'no_cross_job_speedup_claim':True,'collection_separate_measured_release_required':True,'plan_sha256':sha_file(out/'PLAN.json'),'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),'launcher_sha256':sha_file(ROOT/'accuracy_v2/fixed_coordinate_launch.py'),'pilot_source_sha256':source_digest()},indent=2)+'\n')
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--scope',type=Path,required=True);a.add_argument('--model-review',type=Path,required=True);a.add_argument('--warm-model-review',type=Path,required=True);a.add_argument('--coordinate-candidate',type=Path,required=True);a=a.parse_args();print(build(a.output,a.scope,a.model_review,a.warm_model_review,a.coordinate_candidate))
