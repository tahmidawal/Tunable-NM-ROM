"""Lossless subset extraction of retained native arrays; no numerical fitting."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'artifacts/native-batch03';out=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((base/'results/interface/result.json').read_text());retention=json.loads((base/'RETENTION.json').read_text())
arrays={};sources={}
for key,label in [('velocity','velocity_TSQR'),('pressure','pressure_TSQR')]:
 path=base/('results/interface/'+label+'.npz');sources[key]={'path':str(path.resolve()),'sha256':sha(path)}
 with np.load(path,allow_pickle=False) as z:
  arrays[key+'_candidates']=z['candidate_coefficients'];arrays[key+'_transform']=z['blocked_transform']
np.savez_compressed(out/'BASIS-INPUTS.npz',**arrays)
meta={'scope':'lossless extraction of accepted native-batch03 basis inputs; not training/reference admission',
 'physical_admission':False,'source_job_id':retention['job_id'],'source_archive_sha256':retention['archive_sha256'],
 'source_report_sha256':sha(base/'results/interface/result.json'),'source_report_path':str((base/'results/interface/result.json').resolve()),
 'source_arrays':sources,'source_keys':{'candidates':'candidate_coefficients','transform':'blocked_transform'},
 'extraction_script_sha256':sha(Path(__file__)),
 'producer_manifest_sha256':report['producer_manifest_sha256'],'array_file_sha256':sha(out/'BASIS-INPUTS.npz'),
 'selected_indices':{k:report['TSQR'][k]['blocked']['candidate_indices'] for k in ('velocity','pressure')},
 'array_shapes':{k:list(a.shape) for k,a in arrays.items()}}
(out/'BASIS-INPUTS.json').write_text(json.dumps(meta,indent=2)+'\n')
