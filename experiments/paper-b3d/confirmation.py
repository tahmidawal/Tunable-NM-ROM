"""Two training seeds, longer operator recipes and broader development controls.

This file does not draw the reserved final parameter seed. Both fresh long
operator runs use exactly the same architecture, schedule and update budget.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def invoke(script,*args,log):
    print('START',script,list(args),flush=True)
    with open(log,'w') as out:
        subprocess.run([sys.executable,'-u',script,*map(str,args)],stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',script,list(args),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    base=json.loads(Path(args.config).read_text());assert base['confirmation']['final_cases']==32
    write('../out/confirmation-protocol.json',base['confirmation'])
    invoke('train.py','--config',args.config,'--out','../training',log='../training.log')
    # Inspect the original finite-budget operators before replacing any recipe.
    invoke('operator_panel.py','--config',args.config,'--training','../training','--out','../out/original','--mode','train',log='../original-operators.log')
    invoke('operator_diagnostic.py','--config',args.config,'--training','../training','--out','../out/original',log='../original-diagnostic.log')
    configs=[]
    for seed_index in (0,1):
        cfg=copy.deepcopy(base);cfg.pop('confirmation');cfg.pop('wall_time',None)
        cfg['attempt']=base['attempt']+f'_seed{seed_index}'
        cfg['training_seed_index']=seed_index
        if seed_index:
            cfg.pop('reuse_attempt');cfg.pop('reuse_checkpoint_path',None)
            cfg['training']['seed']=920401
        for index,model in enumerate(cfg['operators']):
            if index>=2:
                model.pop('reuse',None);model['training']['steps']=100000
                model['training']['wall_seconds']=3600;model['training']['validation_every']=500
            elif seed_index:model.pop('reuse',None)
            if seed_index:model['training']['seed']=920410+index
        traincfg=Path(f'../out/seed{seed_index}-training-config.json');write(traincfg,cfg)
        training=Path('../training') if seed_index==0 else Path('../training/seed1')
        out=Path(f'../out/seed{seed_index}')
        if seed_index:invoke('train.py','--config',traincfg,'--out',training,log=f'../seed{seed_index}-training.log')
        invoke('operator_panel.py','--config',traincfg,'--training',training,'--out',out,'--mode','train',log=f'../seed{seed_index}-operators.log')
        invoke('operator_diagnostic.py','--config',traincfg,'--training',training,'--out',out,log=f'../seed{seed_index}-diagnostic.log')
        panel=copy.deepcopy(cfg);panel['validation_rows']=base['confirmation']['development_rows']
        panel['evaluation_kind']='expanded development confirmation';panel['refinement_cases']=[]
        panel['fom_controls']=base['confirmation']['fom_controls'];panel['fom_variants']=base['confirmation']['fom_variants']
        panel['pod_ranks']=[64,128,192,256]
        panelpath=Path(f'../out/seed{seed_index}-panel-config.json');write(panelpath,panel)
        configs.append((seed_index,panelpath,training,out))
    # Pair both learned seeds, all operators and classical controls on this GPU.
    for seed_index,cfg,training,out in configs:
        invoke('run.py','--config',cfg,'--checkpoint',training/'checkpoint.pkl','--out',out,log=f'../seed{seed_index}-driver.log')
        invoke('operator_panel.py','--config',cfg,'--training',training,'--out',out,'--mode','evaluate',log=f'../seed{seed_index}-evaluate.log')
        invoke('audit.py',out,'--checkpoint',training/'checkpoint.pkl',log=f'../seed{seed_index}-audit.log')
    write('../out/confirmation-complete.json',dict(complete=True,seeds=[0,1],final_cohort_unopened=True))


if __name__=='__main__':main()
