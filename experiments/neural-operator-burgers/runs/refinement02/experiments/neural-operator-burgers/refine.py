"""Focused timestep refinement with exact cached-solve provenance and fixed gate."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import data as d


def configure():
    refined=d.HERE/'protocol-refined.json'
    if d.PROTOCOL_PATH==refined:return
    d.PROTOCOL_PATH=refined
    d.PROTOCOL=json.loads(refined.read_text())
    d.SOURCE_FILES=[*d.SOURCE_FILES,refined,Path(__file__).resolve()]


def refinement(args):
    oldpath=Path(args.original).resolve();old=json.loads(oldpath.read_text())
    assert old['complete'] and old['count']==8 and old['kind']=='burgers-reference-calibration'
    assert old['provenance']['backend']=='gpu' and old['provenance']['f64']
    assert old['provenance']['matmul_precision']=='highest'
    # Cache reuse requires the unchanged full original numerical implementation
    # and protocol, plus exact checksums for each completed input solve.
    assert old['protocol_sha256']==d.sha(d.PROTOCOL_PATH)
    assert old['provenance']['source_sha256']==d.source_hashes()
    for row in old['solves']+old['records']:
        assert d.sha(oldpath.parent/row['path'])==row['sha256']
    configure()
    jax,e=d.gpu_modules();out=d.make_output(args.out)
    report=dict(schema_version=1,kind='burgers-reference-calibration',pde='burgers',count=8,complete=False,
                protocol_sha256=d.sha(d.PROTOCOL_PATH),provenance=d.provenance(jax),records=[],solves=[],
                reference_anchor=d.PROTOCOL['anchor'],role='focused temporal reference refinement; original failed gate remains unchanged',
                parent_calibration=dict(index_sha256=d.sha(oldpath),protocol_sha256=old['protocol_sha256'],
                    provenance=old['provenance'],gate=old['gate']))
    path=out/'index.json';d.write_json(path,report)
    settings=list(dict.fromkeys(d.anchor_settings()+[(p['intervals'],p['dt']) for p in d.PROTOCOL['candidates']]))
    oldrows={(s['intervals'],s['dt'],s['case_id']):s for s in old['solves']}
    all_fields={};started=time.monotonic()
    try:
        for case in range(8):
            record=d.case_record('calibration',case);physical=e.params_draw(record['seed'],1)[0]
            for L,dt in settings:
                key=(L,dt,record['case_id']);name=f"{record['case_id']}_{d.setting_id(L,dt)}.npz"
                if key in oldrows:
                    row=oldrows[key];assert row['seed']==record['seed']
                    shutil.copy2(oldpath.parent/row['path'],out/name)
                    with np.load(out/name) as saved:
                        fields=saved['fields'];iterations=saved['iterations'];residuals=saved['residuals']
                    newrow=dict(row,path=name,cache_reused=True,cache_parent_index_sha256=d.sha(oldpath),
                        cache_parent_source_commit=old['provenance']['source_commit'],cache_parent_protocol_sha256=old['protocol_sha256'])
                else:
                    if time.monotonic()-started+240>args.seconds:
                        report['stop_reason']='budget checkpoint before new reference solve';d.write_json(path,report);return
                    print('REFINE',L,dt,'case',case,flush=True)
                    query,dense=d.make_solver(e,L,dt,1024)
                    fields,iterations,residuals,seconds=d.solve(jax,e,query,dense,1024,physical)
                    np.savez(out/name,fields=fields,iterations=iterations,residuals=residuals)
                    newrow=dict(case_id=record['case_id'],seed=record['seed'],intervals=L,dt=dt,path=name,
                        sha256=d.sha(out/name),output_intervals=1024,cache_reused=False,
                        wall_seconds_including_first_compile=seconds,max_relative_residual=float(residuals.max()),
                        total_newton_iterations=int(iterations.sum()))
                    del query;jax.clear_caches()
                all_fields[L,dt,case]=fields
                report['solves'].append(newrow);d.write_json(path,report)
            print('REFINED CASE COMPLETE',record['case_id'],flush=True)
        report['gate']=d.evaluate_gate(report,all_fields)
        L,dt=d.PROTOCOL['anchor']['intervals'],d.PROTOCOL['anchor']['dt']
        for case in range(8):
            record=d.case_record('calibration',case);physical=e.params_draw(record['seed'],1)[0]
            report['records'].append(d.save_case(out,record,all_fields[L,dt,case],physical,1024,
                dict(intervals=L,dt=dt,role='focused temporal anchor; empirical gate controls use')))
        report['elapsed_seconds']=time.monotonic()-started;report['complete']=True;d.write_json(path,report)
        print('REFINEMENT COMPLETE',json.dumps(report['gate']['by_output']['256']['candidates']),flush=True)
    except BaseException as error:
        report['failure']=str(error);d.write_json(path,report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    c=sp.add_parser('refine');c.add_argument('--original',required=True);c.add_argument('--out',required=True);c.add_argument('--seconds',type=float,required=True);c.set_defaults(func=refinement)
    c=sp.add_parser('generate');c.add_argument('--calibration',required=True);c.add_argument('--out',required=True);c.add_argument('--split',choices=['train','validation'],required=True);c.add_argument('--intervals',type=int,default=256);c.add_argument('--count',type=int)
    def generate(a):configure();d.generate(a)
    c.set_defaults(func=generate);a=p.parse_args();a.func(a)


if __name__=='__main__':main()
