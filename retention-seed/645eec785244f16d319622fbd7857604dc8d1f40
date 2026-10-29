"""One bounded, explicitly released parameter group; no retry or admission."""
from pathlib import Path
import argparse,json,subprocess,sys,time,traceback,os
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--group',required=True);p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    import hashlib
    plan=json.loads(a.plan.read_text());release=json.loads(a.release.read_text())
    if release.get('population_plan_sha256')!=hashlib.sha256(a.plan.read_bytes()).hexdigest() or a.group not in release.get('allowed_groups',[]):raise ValueError('explicit exact group release required')
    if release.get('reference_infrastructure_seconds_remaining',0)<1000 or release.get('campaign_seconds_remaining',0)<1000:raise ValueError('allocation budget not reserved')
    group=next(x for x in plan['groups'] if x['id']==a.group)
    if min(release['reference_infrastructure_seconds_remaining'],release['campaign_seconds_remaining'])<group['allocation_seconds']:raise ValueError('group wall budget not reserved')
    root=Path('results');root.mkdir(exist_ok=True);
    if (root/'population-progress.json').exists() or (root/'states').exists():raise FileExistsError('population outputs already exist; no retry')
    start=time.monotonic();record={'group':a.group,'complete':False,'admission':False,'completed':[]}
    def save():(root/'population-progress.json').write_text(json.dumps(record,indent=2)+'\n')
    def run(name,command):
        remaining=min(group['runner_seconds']-(time.monotonic()-start),group['allocation_seconds']-(time.time()-float(os.environ['NSOBS_ALLOCATION_STARTED']))-20)
        if remaining<=0:raise TimeoutError('bounded runner exhausted; no automatic retry')
        tick=time.monotonic()
        with (root/(name+'.stdout')).open('w') as out,(root/(name+'.stderr')).open('w') as err:
            completed=subprocess.run(command,stdout=out,stderr=err,timeout=min(remaining,300 if name.startswith('solve-') else remaining))
        record['completed'].append({'name':name,'seconds':time.monotonic()-tick,'returncode':completed.returncode});save()
        completed.check_returncode()
    try:
        for state in group['states']:
            if state['retained']:continue
            cfg=ROOT/state['config']
            if hashlib.sha256(cfg.read_bytes()).hexdigest()!=state['config_sha256']:raise ValueError('frozen config changed')
            run('solve-'+state['id'],[sys.executable,str(ROOT/'fom.py'),'--config',str(cfg),'--out',str(root/'states'/state['id'])])
        run('common-readout',[sys.executable,str(ROOT/'g2/population_readout.py'),'--plan',str(a.plan),'--group',a.group,'--output',str(root/'common')])
        if group['Re']!=20:
            run('fresh-baseline-replay',[sys.executable,str(ROOT/'g2/population_readout.py'),'--plan',str(a.plan),'--group',a.group,'--output',str(root/'fresh-replay'),'--baseline-only'])
        record.update(complete=True,total_seconds=time.monotonic()-start);save()
    except BaseException as exc:
        record.update(failure=repr(exc),traceback=traceback.format_exc(),total_seconds=time.monotonic()-start,partial_results_retained=True);save();raise

if __name__=='__main__':main()
