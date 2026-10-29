from pathlib import Path
import json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
record={'physical_admission':False,'new_PDE_solves':0,'completed':False};tick=time.monotonic()
try:
    command=[sys.executable,str(ROOT/'g2/export_batch_control.py'),'--plan',str(ROOT/'g2/export-batch01/plan.json'),'--out',str(OUT/'export_batch_control')]
    with (OUT/'control.stdout.log').open('w') as stdout,(OUT/'control.stderr.log').open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=900)
    record.update(command=command,returncode=result.returncode,completed=result.returncode==0)
    result.check_returncode()
except Exception as exc:
    record.update(error=repr(exc),traceback=traceback.format_exc());raise
finally:
    record['seconds']=time.monotonic()-tick;(OUT/'control_progress.json').write_text(json.dumps(record,indent=2)+'\n')
