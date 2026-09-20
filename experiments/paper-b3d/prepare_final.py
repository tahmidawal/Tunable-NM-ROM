"""Freeze audited development assets without importing a PDE or drawing inputs."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--confirmation',default='b3d005');p.add_argument('--conditioned')
    p.add_argument('--attempt',default='b3d007');args=p.parse_args()
    run=EXP/'runs'/args.confirmation;collected=run/'collected';accept=json.loads((run/'COLLECTED.json').read_text())
    assert accept['checksums_passed'] and accept['independent_audit_passed']
    protocol=json.loads((EXP/'final-reference-protocol.json').read_text());assert protocol['cases']==32
    assets=[];seeds=[];checkpoints=[];offline_hashes=[];replay_rows=[512,519]
    def add(path,destination):
        record=dict(source=str(path.relative_to(ROOT)),destination=destination,sha256=digest(path));assets.append(record);return record['sha256']
    conditioned=None
    if args.conditioned:
        conditioned=EXP/'runs'/args.conditioned/'collected/out'
        assert json.loads((conditioned/'audit-local.json').read_text())['passed']
        assert json.loads((conditioned/'complete.json').read_text())['complete']
        selection=[]
        for seed in (0,1):
            old=json.loads((collected/f'out/seed{seed}/result.json').read_text())
            old_worst=max(v['worst_evolved'] for v in old['operator_invocations'] if v['method']=='deeponet3d')
            new_worst=json.loads((conditioned/f'seed{seed}/development.json').read_text())['worst_evolved']
            selection.append(dict(seed=seed,original_100k_worst=old_worst,conditioned_worst=new_worst))
        # One common training recipe for both seeds, selected before final access.
        use_conditioned=all(v['conditioned_worst']<v['original_100k_worst'] for v in selection)
    else:selection=[];use_conditioned=False
    for seed in (0,1):
        source=collected/f'out/seed{seed}';result=json.loads((source/'result.json').read_text())
        assert result['complete'] and result['operator_complete'] and result['final_cohort_unopened']
        for name in ['audit-local.json','audit-stationarity-local.json']:
            assert json.loads((source/name).read_text())['passed']
        cfg=copy.deepcopy(result['config']);cfg.pop('reuse_attempt',None);cfg.pop('reuse_checkpoint_path',None)
        cfg.update(attempt=args.attempt+f'_seed{seed}',evaluation_kind='final',evaluation_seed=protocol['parameter_seed'],
            validation_rows=list(range(512,512+protocol['cases'])),final_cohort_unopened=False,refinement_cases=[])
        prefix=f'frozen/seed{seed}';ck=collected/('training/checkpoint.pkl' if seed==0 else 'training/seed1/checkpoint.pkl')
        checkpoint=pickle.loads(ck.read_bytes());tc=cfg['training'];params=checkpoint['params']
        assert checkpoint['cfg']['k']==tc['latent_dimension']==64 and checkpoint['cfg']['r']==tc['rank']==256
        assert checkpoint['Z_tr'].shape==(cfg['train_trajectories']*len(cfg['train_steps']),64)
        assert params['B'].shape==(3,tc['fourier_features']) and params['h_lin'].shape==(64,256)
        assert [tuple(w.shape) for w,b in params['g']]==[(2*tc['fourier_features'],tc['bank_width']),(tc['bank_width'],tc['bank_width']),(tc['bank_width'],256)]
        assert [tuple(w.shape) for w,b in params['h']]==[(64,tc['head_width']),(tc['head_width'],tc['head_width']),(tc['head_width'],256)]
        assert all(b.shape==(w.shape[1],) for w,b in params['g']+params['h'])
        basis=np.load(source/'bases.npz');directions=np.load(source/'directions.npz')
        assert basis['bank'].shape==((cfg['nodes']-2)**3,256) and basis['bank_R'].shape==(256,256)
        assert basis['pod'].shape==((cfg['nodes']-2)**3,256) and directions['C'].shape==(256,256)
        assert len(basis['lam'])==result['actual_test_modes']==642 and cfg['test_modes']==640
        ckhash=add(ck,prefix+'/checkpoint.pkl');checkpoints.append(ckhash)
        hashes={name:add(source/name,prefix+'/offline/'+name) for name in ['bases.npz','directions.npz','result.json']}
        offline_hashes.append(hashes);cfg['frozen_offline']=dict(directory=prefix+'/offline',files=hashes)
        metadata=json.loads((source/'operator_metadata.json').read_text())
        expected=[]
        for row in result['invocations']+result['operator_invocations']:
            if row['row'] in replay_rows and row['repetition']==0:
                if use_conditioned and row['method']=='deeponet3d':continue
                target=prefix+'/expected/'+row['artifact'];add(source/row['artifact'],target)
                expected.append(dict(method=row['method'],row=row['row'],path=target))
        for model in cfg['operators']:
            model.pop('reuse',None);name=model['name'];old=source/name
            if name=='deeponet3d' and use_conditioned:
                old=conditioned/f'seed{seed}';info=json.loads((old/'metadata.json').read_text())
                model['training']=info['config'];model['pretraining']=info['pretraining']['config']
                metadata['models']=[({**info,'name':name} if v['name']==name else v) for v in metadata['models']]
                for row in replay_rows:
                    target=prefix+f'/expected/conditioned_deeponet_row{row}.npz';add(old/f'prediction_row{row}.npz',target)
                    expected.append(dict(method=name,row=row,path=target))
            for name2 in ['best.pkl','training.json','curve.json']:add(old/name2,prefix+f'/operators/{name}/{name2}')
        for row in replay_rows:
            case=result['config']['validation_rows'].index(row);add(source/f'reference_case{case}.npz',prefix+f'/expected/reference_row{row}.npz')
        seeds.append(dict(seed_index=seed,role='prospective primary' if seed==0 else 'independent initialization robustness',
            checkpoint=prefix+'/checkpoint.pkl',config=cfg,operator_metadata=metadata,replay_expected=expected,
            replay_reference_paths={str(row):prefix+f'/expected/reference_row{row}.npz' for row in replay_rows}))
    freeze=dict(schema='b3d-final-freeze-v1',final_parameter_seed=protocol['parameter_seed'],final_cases=protocol['cases'],
        final_rows=protocol['parameter_rows'],final_fields_generated=False,primary_seed_index=0,
        architecture_selection='K64 R256, fixed actual M642; original bank primary and independently trained second seed',
        operator_selection='same recipe for both seeds; conditioned DeepONet only if expanded-development worst improves for both',
        conditioned_selected=use_conditioned,development_recipe_comparison=selection,
        classical_selection='all predeclared controls retained; within-job cost/error frontier, no final retuning',
        checkpoint_sha256=checkpoints,offline_artifact_hashes=offline_hashes,replay_rows=replay_rows,
        replay_gate=dict(maximum_absolute_field_discrepancy=1e-8,identical_iterations=True,identical_reasons=True),
        physical_reference_protocol=protocol,physical_reference_protocol_sha256=digest(EXP/'final-reference-protocol.json'),seeds=seeds,assets=assets)
    name='final-freeze.json';write(EXP/name,freeze)
    config=dict(attempt=args.attempt,workflow='final_campaign.py',wall_time='02:00:00',
        freeze_manifest=name,freeze_sha256=digest(EXP/name),assets=assets)
    write(EXP/'config-final.json',config)
    print(json.dumps(dict(freeze_sha256=config['freeze_sha256'],assets=len(assets),conditioned_selected=use_conditioned,
        final_cohort_unopened=True,required_next='retain all listed assets in Git, commit freeze and source, then stage'),indent=2))


if __name__=='__main__':main()
