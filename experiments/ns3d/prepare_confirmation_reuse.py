"""Select the best trained augmented head and prepare data-free reuse assets.

Selection uses retained development representation errors only. The trained
R1536 incumbent competes even when the larger bank/head misses its gates.
"""
import argparse,json,shutil
from pathlib import Path
from extra03 import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--coverage',required=True);p.add_argument('--capacity',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    sources=[Path(a.coverage),Path(a.capacity)];raws=[];candidates=[]
    for source in sources:
        for name in ('audit.json','history_audit.json','source_audit.json'):
            assert json.loads((source.parent/name).read_text())['passed'],(source,name)
        raw=json.loads((source/'output/result.json').read_text());assert raw['complete'] and not raw['final_cohort_opened'];raws.append(raw)
        screen=json.loads((source/'output/capacity/screen.json').read_text())
        for name,record in screen['heads'].items():
            candidates.append(dict(source=str(source),head=name,trained=record['training']['steps']>0,
                bank_rank=screen['rank'],k=record['configuration']['k'],worst=record['head_initial_normalized']['worst']))
    keys=('membership_sha256','base_states_sha256','base_parameter_sha256','augmented_states_sha256')
    assert all(raws[0]['augmentation'][k]==raws[1]['augmentation'][k] for k in keys)
    selected=min((row for row in candidates if row['trained']),key=lambda row:row['worst'])
    source=Path(selected['source']);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);entries=[]
    selected_raw=raws[sources.index(source)]
    def copy(src,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        entries.append(dict(path=name,sha256=file_hash(target),source=str(src),source_sha256=file_hash(src)))
    for old,name in [('capacity/frozen_bank.pkl','frozen_bank.pkl'),('capacity/'+selected['head']+'.pkl','head.pkl'),('capacity/pod.npz','pod.npz'),('capacity/'+selected['head']+'_fields.npz','head_replay_fields.npz')]:copy(source/'output'/old,name)
    copy(source/'output/capacity/screen.json','screen.json')
    coverage=sources[0]/'output';copy(coverage/'operator_statistics.pkl','operator_statistics.pkl')
    for spec in raws[0]['config']['operators']:copy(coverage/spec['kind']/'best.pkl',spec['kind']+'/best.pkl')
    record=dict(files=entries,candidates=candidates,selected=selected,selection='minimum worst development error among all trained same-cohort heads; affine candidates recorded separately',
        source_jobs=[raw['job_id'] for raw in raws],source_commits=[raw['source_commit'] for raw in raws],
        augmentation=raws[0]['augmentation'],operator_records=raws[0]['operators'],final_cohort_opened=False)
    (out/'REUSE.json').write_text(json.dumps(record,indent=2)+'\n')
    cfg={**raws[0]['config'],'k':selected['k'],'r':selected['bank_rank'],'coverage_rank':selected['bank_rank'],'selected_head':selected['head'],
        'model_seed':selected_raw['config']['model_seed'],'bank_hidden_width':selected_raw['config'].get('bank_hidden_width',selected['bank_rank']),
        'selected_training_source_commit':selected_raw['source_commit'],'selected_training_job_id':selected_raw['job_id'],
        'selected_training_config':selected_raw['config'],
        'q_values':[0,32,64,128,256],'test_modes':2048,'rom_dt':.004,'rom_budget':120,'cold_starts':8,'gtol':1e-7,
        'evaluation_cohort':'development','timed_cases':8,'repetitions':3,'reuse_path':'reuse',
        'operator_projection_variants':[False,True],'refinement_n':128,'refinement_dt':.00025,'refinement_budget':.005,
        'final_cases_unopened':32,'physical_accuracy_target':.05,
        'pretraining':dict(teacher_seed=202609270,teacher_oversampling=64,teacher_power_iterations=2,
            trunk_steps=20000,trunk_learning_rate=.001,trunk_batch_points=2048,trunk_wall_seconds=300,
            branch_steps=20000,branch_learning_rate=.001,branch_batch_size=4,branch_wall_seconds=300,checkpoint_every=1000),
        'deep_joint_training':{**raws[0]['config']['operator_train'],'seed':202609270,'steps':40000,'wall_seconds':900}}
    Path(a.config).write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(record['selected'],indent=2))


if __name__=='__main__':main()
