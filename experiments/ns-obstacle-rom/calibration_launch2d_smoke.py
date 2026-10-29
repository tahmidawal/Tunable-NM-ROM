"""Synthetic metadata-only tests of workload completion, never solver evidence."""
from pathlib import Path
import copy,json,tempfile
from calibration_launch2d import complete,IDS
from fom_calibrate2d import OUTPUTS
from fe_adapter import sha_file


def main():
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder);rows=[]
        for candidate in IDS:
            for re in (16.875,23.125):
                for contract in OUTPUTS:
                    p=out/f'{candidate}-{re}-{contract}.json'
                    record={'same_invocation_scores':[{'eligible_accuracy':False} for _ in range(7)],
                            'readout_control':{'passed':False,'exception':'ArithmeticError: synthetic failed control'}}
                    p.write_text(json.dumps({'record':record}))
                    rows.append({'candidate':candidate,'Re':re,'contract':contract,'elapsed_seconds':[1.]*7,
                                 'evidence':{'path':str(p),'sha256':sha_file(p)}})
        ledger={'blocks':rows,'construction_failures':[]}
        assert complete(ledger,out)['complete_planned_workload']
        assert not complete(ledger,out)['matched_accuracy']
        for mode in ('missing_block','timeout_control','budget_skipped_score'):
            altered=copy.deepcopy(ledger)
            if mode=='missing_block':altered['blocks'].pop()
            else:
                p=out/f'{mode}.json';record={'same_invocation_scores':[{} for _ in range(7)],'readout_control':{'passed':False}}
                if mode=='timeout_control':record['readout_control']['exception']='TimeoutError: synthetic deadline'
                else:record['same_invocation_scores'][0]['diagnostics_unavailable_budget']=True
                p.write_text(json.dumps({'record':record}));altered['blocks'][0]['evidence']={'path':str(p),'sha256':sha_file(p)}
            try:complete(altered,out)
            except RuntimeError:pass
            else:raise AssertionError(mode+' was accepted')
            assert not json.loads((out/'WORKLOAD-COMPLETENESS.json').read_text())['complete_planned_workload']
    print('PASS: synthetic metadata only; complete failed-accuracy workloads differ from incomplete workloads; no matched claim')


if __name__=='__main__':main()
