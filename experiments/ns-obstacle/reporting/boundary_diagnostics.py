"""Pointwise geometry and outlet diagnostics from retained boundary arrays."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',action='append',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[];sources=[]
    for entry in a.case:
        label,folder=entry.split(':',1);folder=Path(folder)
        def read(name):
            path=folder/name;raw=path.read_bytes();sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()));return json.loads(raw)
        cfg=read('config.json');boundary=read('boundary_rule.json');trace=read('essential_trace_certificate.json')
        obstacle=[r for r in boundary if r['label']=='obstacle'];outlet=[r for r in boundary if r['label']=='outlet']
        xy=np.asarray([r['point'] for r in obstacle]);normal=np.asarray([r['normal'] for r in obstacle]);radial=xy-np.asarray(cfg['center']);radius=np.linalg.norm(radial,axis=1)
        ideal=-radial/radius[:,None];unit=normal/np.linalg.norm(normal,axis=1)[:,None]
        cross=unit[:,0]*ideal[:,1]-unit[:,1]*ideal[:,0];dot=np.sum(unit*ideal,axis=1);angle=np.arctan2(np.abs(cross),dot);deviation=radius-cfg['radius']
        weights=np.asarray([r['weight'] for r in outlet]);un=np.asarray([np.dot(r['velocity'],r['normal']) for r in outlet]);backflow=float(np.dot(weights,np.maximum(-un,0)))
        arrays=a.out/(label+'.npz');np.savez_compressed(arrays,obstacle_points=xy,obstacle_normals=normal,radial_deviation=deviation,normal_angle_radians=angle,outlet_weights=weights,outlet_normal_velocity=un)
        rows.append(dict(case=label,geometry_order=cfg['geometry_order'],boundary_diagnostic_order=cfg['diagnostic_order'],obstacle_points=len(obstacle),maximum_radial_deviation=float(np.max(np.abs(deviation))),maximum_normal_angle_radians=float(np.max(angle)),outlet_backflow_over_UH=backflow/(cfg['U']*cfg['H']),trace_maximum_estimate_over_U=trace['maximum_estimate_over_U'],trace_bound_over_U=trace['upper_bound_over_U'],arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest()))
    record=dict(scope='Diagnostics only; no added thresholds or reference admission',physical_admission=False,rows=rows,sources=sources,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.out/'diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=['# Retained boundary diagnostics','',record['scope']+'.','', '| Case | Max radial deviation | Max normal angle (radians) | Outlet backflow / UH | Raw trace estimate / U | Trace bound / U |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['maximum_radial_deviation']:.9g} | {r['maximum_normal_angle_radians']:.9g} | {r['outlet_backflow_over_UH']:.9g} | {r['trace_maximum_estimate_over_U']:.9g} | {r['trace_bound_over_U']:.9g} |")
    lines+=['','## Glossary','','Radial deviation: distance of a retained obstacle quadrature point from the exact circle. Normal angle: difference between the recorded fluid-outward normal and the inward radial direction. These are sampled diagnostics, not continuous-boundary maxima. Backflow: inward outlet flux normalized by inlet mean speed times channel height. Raw trace estimate: polynomial estimate of error against prescribed FE boundary velocity. Trace bound: conservative coefficient bound including the fixed roundoff allowance.']
    (a.out/'diagnostics.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
