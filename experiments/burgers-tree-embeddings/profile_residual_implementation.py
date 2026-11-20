"""Run two retained-state implementation profiles sequentially in one owned job."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
record=dict(status='running_implementation_profiles',final_test_opened=False,method_admitted=False,profiles={})
for name,driver in [('jacobian','benchmark_residual_jacobian.py'),('output','benchmark_residual_output.py')]:
    dest=out/name;dest.mkdir(exist_ok=True)
    with (out/(name+'.log')).open('w') as log:
        execution=subprocess.run([sys.executable,driver,'--out',str(dest)],stdout=log,stderr=subprocess.STDOUT)
    result=dest/'result.json'
    record['profiles'][name]=dict(returncode=execution.returncode,result=json.loads(result.read_text()) if result.exists() else None)
    record['elapsed_seconds']=time.monotonic()-started
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print('implementation_profile',name,execution.returncode,flush=True)
    if execution.returncode:sys.exit(execution.returncode)
record['status']='complete_implementation_profiles'
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
