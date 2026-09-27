"""Bounded DeepONet training repair and mandatory matched NM-ROM dev panel."""
import argparse,gc,json,os,shutil,subprocess,traceback
from pathlib import Path
import numpy as np
import jax
import ns3d_fom as F
import ns3d_model as D
import translation_cohort as TC
import dataset_adapter as A
import operator_adapter as O
from operators import pretrained_deeponet as PT,training as OT
from extra03 import load,file_hash,with_coordinates
from pilot import generate,write,sha
from frozen_panel import prepare_assets,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);reuse=Path(cfg['reuse_path'])
    record=json.loads((reuse/'REUSE.json').read_text());assert all(file_hash(reuse/x['path'])==x['sha256'] for x in record['files'])
    assert cfg['evaluation_cohort']=='development' and jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    report=dict(complete=False,final_cohort_opened=False,config=cfg,source_commit=os.environ.get('SOURCE_COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip())
    def stage(name):report['stage']=name;write(report,out/'result.json');print('STAGE',name,flush=True)
    try:
        stage('regenerate_shared_training_data');base,report['train_data']=generate(cfg['n'],cfg['dt'],cfg['horizon'],cfg['train_cases'],cfg['train_seed'],out/'train_data.npz')
        dev,report['dev_data']=generate(cfg['n'],cfg['dt'],cfg['horizon'],cfg['dev_cases'],cfg['dev_seed'],out/'dev_training_selection.npz')
        parameters=F.parameters(cfg['train_seed'],len(base));table=TC.membership(len(base),cfg['n'],cfg['augmentation_copies'],cfg['augmentation_seed'])
        membership=TC.manifest(table,cfg['n'],cfg['augmentation_seed'],base,parameters);train=TC.augment(base,table);membership['augmented_states_sha256']=sha(train)
        assert all(membership[k]==record['augmentation'][k] for k in ('membership_sha256','base_states_sha256','base_parameter_sha256','augmented_states_sha256'))
        report['augmentation']=membership;write(membership,out/'membership.json');del base;gc.collect()
        trds=dict(initial=train[:,0],viscosity=parameters[table[:,0],-1:],targets=train[:,1:]-train[:,0,None]);dvds=dict(initial=dev[:,0],viscosity=F.parameters(cfg['dev_seed'],len(dev))[:,-1:],targets=dev[:,1:]-dev[:,0,None])
        statistics=A.training_statistics(trds);old_statistics=load(reuse/'operator_statistics.pkl')
        assert all(np.allclose(statistics[k],old_statistics[k],rtol=1e-13,atol=1e-13) for k in statistics)
        statistics=old_statistics;O.checkpoint(out/'confirmation_statistics.pkl',statistics)
        tx,ty=A.common_model_arrays(trds,statistics);vx,vy=A.common_model_arrays(dvds,statistics)
        td=np.repeat(np.sum(trds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,1)
        vd=np.repeat(np.sum(dvds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,1)
        del train,trds,dvds;gc.collect();tx=with_coordinates(tx);vx=with_coordinates(vx)
        stage('train_only_deeponet_initialization');spec=next(s for s in cfg['operators'] if s['kind']=='deeponet3d')
        params,pretraining=PT.pretrain(tx,ty,spec,cfg['deep_joint_training'],cfg['pretraining'],out/'deeponet_candidate/pretraining',lambda p,o:write(o,p),O.checkpoint,td)
        stage('joint_deeponet_finetuning');params,info=OT.train(tx,ty,vx,vy,spec,cfg['deep_joint_training'],out/'deeponet_candidate',lambda p,o:write(o,p),O.checkpoint,td,vd,params)
        original=next(x for x in record['operator_records'] if x['spec']['kind']=='deeponet3d')
        selected=info['best_validation_worst']<original['best_validation_worst']
        report['deeponet_training']=dict(pretraining=pretraining,candidate=info,original=original,selected_candidate=selected,
            selection='minimum worst error on the original complete development cohort; original failed checkpoint retained')
        del tx,ty,vx,vy,td,vd,params;gc.collect();jax.clear_caches()
        stage('prepare_frozen_offline_assets');metadata=prepare_assets(cfg,reuse,out/'assets')
        if selected:shutil.copy2(out/'deeponet_candidate/best.pkl',out/'assets/deeponet3d/best.pkl')
        metadata['deeponet_training']=report['deeponet_training'];write(metadata,out/'assets/metadata.json')
        write(dict(files={str(p.relative_to(out/'assets')):file_hash(p) for p in sorted((out/'assets').rglob('*')) if p.is_file() and p.name!='manifest.json'},final_cohort_opened=False),out/'assets/manifest.json')
        stage('real_checkpoint_representation_replay');bank=load(reuse/'frozen_bank.pkl');head=load(reuse/'head.pkl');screen=json.loads((reuse/'screen.json').read_text());fit=screen['heads'][cfg['selected_head']]['fits']
        prediction=np.asarray(D.head(jax.device_put(head['params']),jax.device_put(np.asarray(fit['z'])))@jax.device_put(bank['extra']['bank']).T)
        with np.load(reuse/'head_replay_fields.npz') as f:expected=f['prediction'].reshape(prediction.shape)
        error=float(np.linalg.norm(prediction-expected)/np.linalg.norm(expected));assert error<1e-10
        report['checkpoint_replay']=dict(passed=True,maximum_relative=error,actual_bank_head_loaded=True,final_cohort_opened=False)
        del bank,head,prediction,expected,dev;gc.collect();jax.clear_caches()
        evaluate(cfg,out/'assets',out,cfg['dev_seed'],cfg['timed_cases'],report)
    except Exception as error:
        report['error']=str(error);report['traceback']=traceback.format_exc();write(report,out/'result.json');raise


if __name__=='__main__':main()
