"""Replay frozen development cases before opening either sealed final panel."""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def invoke(script,*args,log):
    print('START',script,list(map(str,args)),flush=True)
    with open(log,'w') as stream:subprocess.run([sys.executable,'-u',script,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('DONE',script,flush=True)


def operators(seed,out):
    out.mkdir(parents=True,exist_ok=True);source=Path(seed['checkpoint']).parent/'operators'
    for model in seed['config']['operators']:shutil.copytree(source/model['name'],out/model['name'])
    dump(out/'operator_metadata.json',seed['operator_metadata'])


def replay_audit(seed,path,gate):
    result=json.loads((path/'result.json').read_text());rows=result['invocations']+result['operator_invocations'];checks=[]
    for expected in seed['replay_expected']:
        actual=next(v for v in rows if v['row']==expected['row'] and v['method']==expected['method'])
        before=np.load(expected['path']);after=np.load(path/actual['artifact'])
        error=float(np.max(np.abs(before['fields']-after['fields'])))
        same_iterations='iterations' not in before.files or np.array_equal(before['iterations'],after['iterations'])
        same_reasons='reasons' not in before.files or np.array_equal(before['reasons'],after['reasons'])
        checks.append(dict(method=actual['method'],row=actual['row'],maximum_absolute_field_discrepancy=error,
            identical_iterations=bool(same_iterations),identical_reasons=bool(same_reasons),
            passed=bool(error<=gate['maximum_absolute_field_discrepancy'] and same_iterations and same_reasons)))
    for case,row in enumerate(result['config']['validation_rows']):
        before=np.load(seed['replay_reference_paths'][str(row)]);after=np.load(path/f'reference_case{case}.npz')
        error=float(np.max(np.abs(before['fields']-after['fields'])))
        checks.append(dict(method='reference',row=row,maximum_absolute_field_discrepancy=error,
            passed=error<=gate['maximum_absolute_field_discrepancy']))
    answer=dict(passed=all(v['passed'] for v in checks),gate=gate,checks=checks,
        maximum_absolute_field_discrepancy=max(v['maximum_absolute_field_discrepancy'] for v in checks),
        final_cohort_unopened=True)
    dump(path/'replay-audit.json',answer);return answer


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert sha(cfg['freeze_manifest'])==cfg['freeze_sha256'];freeze=json.loads(Path(cfg['freeze_manifest']).read_text())
    assert sha('final-reference-protocol.json')==freeze['physical_reference_protocol_sha256']
    for asset in freeze['assets']:assert sha(asset['destination'])==asset['sha256'],asset['destination']
    shutil.copyfile(cfg['freeze_manifest'],out/'freeze.json')
    # Both seed replays finish before any process receives the reserved seed.
    for seed in freeze['seeds']:
        index=seed['seed_index'];path=out/f'replay-seed{index}';operators(seed,path)
        replay=copy.deepcopy(seed['config']);replay.pop('evaluation_seed');replay['evaluation_kind']='frozen development replay'
        replay['validation_rows']=freeze['replay_rows'];replay['repetitions']=1;replay['final_cohort_unopened']=True
        file=out/f'replay-seed{index}-config.json';dump(file,replay)
        invoke('run.py','--config',file,'--checkpoint',seed['checkpoint'],'--out',path,log=out/f'replay-seed{index}-driver.log')
        invoke('operator_panel.py','--config',file,'--training','unused','--out',path,'--mode','evaluate',log=out/f'replay-seed{index}-operators.log')
        verdict=replay_audit(seed,path,freeze['replay_gate']);assert verdict['passed'],'Frozen development replay failed; final seed remains unopened'
    dump(out/'replay-complete.json',dict(passed=True,seeds=[0,1],final_cohort_unopened=True))
    for seed in freeze['seeds']:
        index=seed['seed_index'];path=out/f'seed{index}';operators(seed,path)
        final=copy.deepcopy(seed['config']);final.update(freeze_manifest=cfg['freeze_manifest'],freeze_sha256=cfg['freeze_sha256'])
        file=out/f'seed{index}-config.json';dump(file,final)
        invoke('run.py','--config',file,'--checkpoint',seed['checkpoint'],'--out',path,log=out/f'seed{index}-driver.log')
        invoke('operator_panel.py','--config',file,'--training','unused','--out',path,'--mode','evaluate',log=out/f'seed{index}-operators.log')
        invoke('audit.py',path,'--checkpoint',seed['checkpoint'],log=out/f'seed{index}-audit.log')
    # The reference-only process reads the already recorded final parameters.
    invoke('final_references.py','--panel',out/'seed0','--protocol','final-reference-protocol.json',
        '--out',out/'physical-reference',log=out/'physical-reference.log')
    dump(out/'complete.json',dict(complete=True,final_cases=freeze['final_cases'],final_parameter_seed=freeze['final_parameter_seed'],
        seeds=[0,1],primary_seed_index=0,freeze_sha256=cfg['freeze_sha256']))


if __name__=='__main__':main()
