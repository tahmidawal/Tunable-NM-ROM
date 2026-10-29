from pathlib import Path
import subprocess,json,hashlib
script=r'''
from pathlib import Path
import subprocess,json,hashlib
b=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027');records={}
for name,jid in [('model-development01',6636731),('fom-calibration01',6638028)]:
 p=b/name/f'slurm-{jid}.out';raw=p.read_bytes();text=raw.decode();section=text.split('Architecture:',1)[1].split('name, uuid, memory.total',1)[0];section='Architecture:'+section
 rows={line.split(':',1)[0].strip():line.split(':',1)[1].strip() for line in section.splitlines() if ':' in line}
 runtime=b/name/'results'/('runtime.json' if name=='model-development01' else 'dependency_runtime.json')
 records[name]={'log_path':str(p),'log_sha256':hashlib.sha256(raw).hexdigest(),'lscpu':rows,'runtime':json.loads(runtime.read_text()),'runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),'gpu_lines':[s for s in text.splitlines() if s.startswith('NVIDIA ')]}
a,c=[r['lscpu'] for r in records.values()];differences={k:[a.get(k),c.get(k)] for k in set(a)|set(c) if a.get(k)!=c.get(k)}
print(json.dumps({'schema':'ns-obstacle-readonly-recovery-host-comparison-v1','records':records,'lscpu_differences':differences,'stable_lscpu_equal':not(set(differences)-{'CPU(s) scaling MHz'}),'cluster_time':subprocess.check_output(['date','-Is'],text=True).strip(),'queue':subprocess.check_output(['squeue','--start','-j','6638632','-h','-o','%i|%S|%R'],text=True).strip(),'pax021_node':subprocess.check_output(['scontrol','show','node','pax021','-o'],text=True).strip(),'execution_or_scheduling_changed':False}))
'''
r=subprocess.run(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=script,text=True,capture_output=True,check=True);record=json.loads(r.stdout);out=Path(__file__).resolve().parent/'pod-recovery-host-comparison.json';out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='records'},indent=2))
