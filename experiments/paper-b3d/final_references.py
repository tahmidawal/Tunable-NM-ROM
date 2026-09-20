"""Prospectively fixed empirical references for the already opened final panel."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from reference_stream import stream
from run import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();panel=Path(a.panel);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(a.protocol).read_text());result=json.loads((panel/'result.json').read_text())
    assert result['complete'] and result['config']['evaluation_kind']=='final'
    assert result['config']['evaluation_seed']==protocol['parameter_seed']
    tab=dict(np.load(panel/'parameters.npz'));assert int(tab['m'])==protocol['cases'];pc=protocol['physical_reference']
    records=[];comparisons=[]
    def save():dump(out/'result.json',dict(protocol=protocol,records=records,comparisons=comparisons,complete=len(comparisons)==protocol['cases'],
        physical_reference_gate_passed=bool(comparisons) and all(v['spatial_passed'] and v.get('time_passed',True) for v in comparisons),
        interpretation='same-grid remains primary; empirical refinement failures are retained without relaxing either budget'))
    for case in range(protocol['cases']):
        fields=[]
        for level in pc['levels']:
            n=level['nodes'];dt=level['dt'];arrays,info=stream(tab,case,n,dt,retain_endpoints=case in pc['time_refinement_cases'])
            name=f'case{case}_n{n}_dt{dt:g}.npz';np.savez_compressed(out/name,**arrays);fields.append(arrays['fields'])
            records.append(dict(case=case,artifact=name,**info));save()
        norm=np.linalg.norm(fields[-1][0]);spatial=np.linalg.norm(fields[-1]-fields[0],axis=1)/norm
        coarse=np.load(panel/f'reference_case{case}.npz')['fields'];samegrid=np.linalg.norm(coarse-fields[-1],axis=1)/norm
        row=dict(case=case,spatial_error_by_time=spatial.tolist(),spatial_worst=float(spatial.max()),
            spatial_passed=bool(spatial.max()<=pc['spatial_refinement_budget']),same_grid_discrepancy_by_time=samegrid.tolist(),
            same_grid_discrepancy_worst=float(samegrid.max()),fine_artifact=records[-1]['artifact'])
        if case in pc['time_refinement_cases']:
            level=pc['time_refinement'];arrays,info=stream(tab,case,level['nodes'],level['dt'],retain_endpoints=True)
            name=f"case{case}_n{level['nodes']}_dt{level['dt']:g}.npz";np.savez_compressed(out/name,**arrays)
            records.append(dict(case=case,artifact=name,**info));time_error=np.linalg.norm(arrays['fields']-fields[-1],axis=1)/norm
            row.update(time_error_by_time=time_error.tolist(),time_worst=float(time_error.max()),time_passed=bool(time_error.max()<=pc['time_refinement_budget']))
        comparisons.append(row);save();print('FINAL REFERENCE',case,row['spatial_worst'],row.get('time_worst'),flush=True)


if __name__=='__main__':main()
