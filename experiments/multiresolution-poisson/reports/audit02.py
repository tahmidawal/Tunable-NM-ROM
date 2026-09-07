"""Collected follow-up artifact/provenance and independent CPU metric audit."""
import argparse
from pathlib import Path
import hashlib
import io
import json
import math
import subprocess
import tarfile
import numpy as np
from scipy.fft import dstn

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
run=a.run.resolve();tree=Path(__file__).resolve().parents[3]
d=json.loads((run/'result.json').read_text());sub=json.loads((run/'submission.json').read_text())
clean=json.loads((run/'cleanup.json').read_text());cfg=d['config']
sha=lambda b:hashlib.sha256(b).hexdigest()
archive=run/'verified-cluster.tar.gz'
if not archive.exists():
    manifest=json.loads((run/'ARCHIVE.json').read_text())
    with archive.open('wb') as stream:
        for part in manifest['ordered_parts']:
            payload=(run/part['name']).read_bytes();assert sha(payload)==part['sha256'];stream.write(payload)
assert sha(archive.read_bytes())==clean['archive_sha256']
assert clean['remote_deleted_and_absence_checked'] and d['complete']
assert d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id']
assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
files={};fields={};reference={};oracles={}
with tarfile.open(archive,'r|gz') as tar:
    for item in tar:
        if not item.isfile():continue
        name=item.name.removeprefix('cluster/')
        if name.startswith('code/') or name in ('in/model.pkl','in/ORIGIN.json','out/pilot/result.json'):
            files[name]=tar.extractfile(item).read()
        elif name.startswith('out/pilot/') and name.endswith('.npz'):
            payload=tar.extractfile(item).read()
            arrays=np.load(io.BytesIO(payload))
            if 'field' in arrays:
                field=arrays['field'];fields[sha(np.ascontiguousarray(field).tobytes())]=field
            elif 'observation' in arrays:
                case=int(Path(name).stem.split('case')[1]);reference[case]={key:arrays[key] for key in arrays.files}
            else:oracles[Path(name).stem]={key:arrays[key] for key in arrays.files}
assert files['out/pilot/result.json']==(run/'result.json').read_bytes()
pathmap={'sep_common.py':'experiments/separable-decoder/sep_common.py',
    'ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py',
    'ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
for staged,expected in sub['source_hashes'].items():
    content=files[staged];assert sha(content)==expected
    name=Path(staged).name;path=pathmap.get(name,'experiments/multiresolution-poisson/'+name)
    assert content==subprocess.check_output(['git','show',sub['source_commit']+':'+path],cwd=tree)
origin=json.loads(files['in/ORIGIN.json']);ck=files['in/model.pkl']
assert sha(ck)==d['checkpoint_sha256']==origin['checkpoint_sha256']
assert ck==subprocess.check_output(['git','show',sub['source_commit']+':'+origin['checkpoint_path']],cwd=tree)
log=(run/(sub['job_id']+'.out')).read_text()+(run/(sub['job_id']+'.err')).read_text()
assert 'jax_backend=gpu' in log and 'ALL-DONE' in log and (run/'EXIT_CODE').read_text().strip()=='0'
for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback']:assert bad not in log,bad
# Independent host regeneration of same-grid source and discrete DST reference.
truth={}
for n in cfg['intervals']:
    x=np.arange(1,n)/n
    lam1=4*n*n*np.sin(np.pi*np.arange(1,n)/(2*n))**2
    for case,(cx,cy,width,amp) in enumerate(d['cohort']['parameters']):
        f=amp*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width*width))
        truth[(n,case)]=np.pad(dstn(dstn(f,type=1,norm='ortho')/(lam1[:,None]+lam1[None,:]),type=1,norm='ortho'),1)
rel=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
errors=[];keys=set();components=0
for row in d['rows']:
    key=(row['intervals'],row['case'],row['arm'],row['requested_modes'],row['tau'],row['repetition'])
    assert key not in keys;keys.add(key)
    names=['input_seconds','fused_device_seconds','output_seconds'] if row['arm']=='rom_fused' else ['input_seconds','projection_init_seconds','solver_seconds','output_seconds']
    assert abs(row['total_seconds']-sum(row[k] for k in names))<1e-10;components+=1
    field=fields[row['field_sha256']];n=row['intervals'];case=row['case'];factor=n//cfg['observation_intervals']
    obs=field[::factor,::factor];fine=reference[case]['observation'];coarser=reference[case]['coarser_observation']
    delta=rel(coarser,fine);error=rel(obs,fine);adjusted=(error+delta)/(1-delta)
    diffs=[abs(error-row['physical_error']),abs(delta-row['reference_delta']),abs(adjusted-row['conservative_physical_error']),abs(rel(field,truth[(n,case)])-row['same_grid_error'])]
    errors+=diffs
    assert max(diffs)<1e-10
expected=len(cfg['intervals'])*cfg['cohort_count']*cfg['repetitions']*(2+2*len(cfg['taus'])*len(cfg['requested_modes_ladder']))
assert len(keys)==expected
oracle_errors=[]
for item in d['oracles']:
    n=item['intervals'];case=item['case'];saved=oracles[f'oracle_n{n}_case{case}']
    oracle_errors+=[abs(rel(saved['bank_projection'],truth[(n,case)])-item['full_bank_same_grid_error']),abs(rel(saved['best_head'],truth[(n,case)])-item['best_same_grid_error'])]
    assert item['full_bank_same_grid_error']<=item['best_same_grid_error']+1e-12
    assert max(x['qr_identity_absolute'] for x in item['rows'])<1e-10
assert max(oracle_errors)<1e-10
report=dict(passed=True,job_id=sub['job_id'],source_commit=sub['source_commit'],archive_sha256=clean['archive_sha256'],
    row_count=len(keys),distinct_preserved_field_hashes=len(fields),same_invocation_components_checked=components,
    maximum_independent_cpu_metric_difference=max(errors),maximum_oracle_cpu_error_difference=max(oracle_errors),
    source_checkpoint_and_result_match_archive_and_git=True,all_reference_differences_recomputed=True,
    all_fusion_parity_passed=all(g['passed'] for g in d['parity']),
    fusion_field_relative_max=max(g['field_relative'] for g in d['parity']),
    fusion_latent_relative_max=max(g['latent_relative'] for g in d['parity']),
    fusion_counters_all_match=all(g['counters_match'] for g in d['parity']),remote_deleted=True,
    reference_limit='Fine observation artifacts audited; no rigorous continuum bound established',
    oracle_limit='All starts saved, but stationary local fits do not prove global minima')
(run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
