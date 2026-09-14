"""Sequential useful tasks after complete calibration, inside a fixed allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import data as d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,required=True)
    a=p.parse_args();out=d.make_output(a.out);started=time.monotonic()
    report=dict(complete=False,tasks=[],budget_seconds=a.seconds,bulk_unlocked=False)
    path=out/'worker.json'
    def save():d.write_json(path,report)
    def run(name,script,args):
        remaining=a.seconds-(time.monotonic()-started)-120
        if remaining<=0:
            report['tasks'].append(dict(name=name,status='not_started_budget'));save();return False
        command=[sys.executable,str(d.HERE/script),*args]
        row=dict(name=name,command=command,timeout_seconds=remaining);report['tasks'].append(row);save()
        before=time.monotonic()
        with (out/f'{name}.log').open('w') as stream:
            try:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
                row['returncode']=result.returncode
                row['status']='complete' if result.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:
                row['status']='timeout_checkpoint_retained'
        row['seconds']=time.monotonic()-before;save()
        return row['status']=='complete'
    save()
    cal=json.loads(a.calibration.read_text())
    assert cal['complete'] and cal['count']==8
    if not run('refinement','refine.py',['refine','--original',str(a.calibration),'--out',str(out/'refinement'),
                                      '--seconds',str(a.seconds-(time.monotonic()-started)-180)]):
        report['complete']=True;save();return
    a.calibration=out/'refinement/index.json'
    cal=json.loads(a.calibration.read_text())
    if not cal['complete']:
        report['complete']=True;report['bulk_decision']=dict(status='locked_incomplete_refinement');save();return
    from refine import configure
    configure()
    try:
        d.read_calibration(a.calibration,256)
    except ValueError as error:
        report['bulk_decision']=dict(status='locked_refined_reference_gate',reason=str(error),
            quantified_gap=cal['gate']['by_output']['256']['candidates'])
        report['complete']=True;save();return
    # The focused-refinement decision requires a passing unchanged reference
    # gate before either diagnosis or bulk data generation proceeds.
    run('diagnosis','diagnose.py',['--reference-index',str(a.calibration),'--out',str(out/'diagnosis'),
                                 '--intervals','256','--cases','8','--reps','3','--reference-dt','.00015625'])
    try:
        selected,evidence=d.read_calibration(a.calibration,256)
    except ValueError as error:
        report['bulk_decision']=dict(status='locked_reference_gate',reason=str(error))
    else:
        costs=[s['wall_seconds_including_first_compile'] for s in cal['solves']
               if s['intervals']==selected['intervals'] and s['dt']==selected['dt']]
        predicted=1.10*max(costs)*160+120
        remaining=a.seconds-(time.monotonic()-started)-120
        report['bulk_decision']=dict(status='resource_decision',selected=selected,
                     conservative_seconds_estimate=predicted,remaining_seconds=remaining,
                     estimate_rule='1.10 times worst compile-inclusive calibrated anchor runtime times 160, plus 120 seconds; child timeout retains case checkpoints')
        if predicted>remaining:
            report['bulk_decision']['status']='deferred_resource_budget'
        else:
            report['bulk_unlocked']=True
            for split in ['train','validation']:
                if not run(split,'refine.py',['generate','--split',split,'--intervals','256',
                           '--calibration',str(a.calibration),'--out',str(out/split)]):
                    break
    report['complete']=True;report['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
