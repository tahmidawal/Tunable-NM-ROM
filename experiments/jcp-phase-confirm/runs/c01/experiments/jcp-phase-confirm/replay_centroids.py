"""Replay accepted saved states to qualify physical-centroid diagnostics, without solves."""
from pathlib import Path
import argparse,json,hashlib,os,subprocess
import numpy as np
import run_mechanism as M
import jax
import jax.numpy as jnp

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
 saved=M.HERE/'saved_development';s=json.loads((saved/'summary.json').read_text());cfg=s['config'];n=cfg['n'];geom=M.F.geometry(n)
 for rel,meta in json.loads((M.HERE/'SOURCE_MANIFEST.json').read_text())['files'].items():
  if M.sha(M.HERE/rel)!=meta['sha256']:raise RuntimeError('source mismatch '+rel)
 inherited=json.loads((M.HERE/'INHERITED_MANIFEST.json').read_text())
 for rel,meta in inherited['files'].items():
  if rel.startswith('saved_development/') and M.sha(M.HERE/rel)!=meta['sha256']:raise RuntimeError('saved record mismatch '+rel)
 pbank=M.BI.load_params(M.HERE/'assets/bank_selected.npz');ck=np.load(M.HERE/'assets/bank_selected.npz')
 G,binfo=M.BI.mesh_bank(pbank,ck['T'],n,chunk=32768,abort=.05);basis=jnp.asarray(G)
 tr=M.F.make_solver(cfg['dt_truth'],round(cfg['horizon']/cfg['dt_truth']),round(cfg['truth_output_dt']/cfg['dt_truth']),scheme=cfg['truth_scheme'])
 params=M.F.parameters(cfg['eval_seed'],cfg['eval_cases'])
 result={'schema':'centroid-replay-v1','purpose':'postprocess accepted development states, no new ROM solve or fitting','source_summary_sha256':M.sha(saved/'summary.json'),'job_id':os.environ.get('SLURM_JOB_ID'),'source_commit':os.environ.get('SOURCE_COMMIT'),'backend':jax.default_backend(),'dtype':str(jnp.ones(1).dtype),'matmul_precision':str(jax.config.jax_default_matmul_precision),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),'concentration_threshold':.1,'threshold_scope':'descriptive reliability flag, not accuracy qualification','cases':[]}
 for c in s['cases']:
  ci=c['index'];par=params[ci];u0=M.F.initial(n,par);truth=np.asarray(tr(jnp.asarray(u0),float(par[-1]),geom));norm0=np.linalg.norm(u0)
  thash=hashlib.sha256(truth.tobytes()).hexdigest();out={'index':ci,'parameters_equal':bool(np.array_equal(par,np.asarray(c['parameters']))),'truth_sha256':thash,'truth_hash_matches':thash==c['truth_sha256'],'runs':[]}
  # Floating-point cross-GPU truth hashes may differ; compare saved scalar metrics directly.
  for arm in ('head_common','span_common'):
   old=next(q for q in c['runs'] if q['arm']==arm and q['rows']==584 and q['dt']==.01 and q['gtol']==1e-9)
   st=np.load(saved/f'case_{ci:03d}'/old['state_file']);metrics=[];gaps=[]
   for t,(coef,center) in enumerate(zip(st['coefficients'],st['centers'])):
    pred=M.reconstruct(basis,jnp.asarray(coef),jnp.asarray(center));m=M.field_metrics(pred,jnp.asarray(truth[t]),geom)
    current={'relative_error':float(m[0])/norm0,'shape_aligned_relative_error':float(m[1])/norm0,'physical_centroid':np.asarray(m[2]),'truth_centroid':np.asarray(m[3]),'phase_displacement':np.asarray(m[4]),'pred_centroid_concentration':np.asarray(m[8]),'truth_centroid_concentration':np.asarray(m[9])}
    for key in ('relative_error','shape_aligned_relative_error','phase_displacement'):
     gaps.append(float(np.max(np.abs(np.asarray(current[key])-np.asarray(old['metrics'][t][key])))))
    metrics.append(current)
   out['runs'].append({'arm':arm,'rows':584,'dt':.01,'gtol':1e-9,'times':st['time'],'metrics':metrics,'max_absolute_saved_metric_gap':max(gaps),'metric_reproduction_passed':max(gaps)<=1e-10})
  result['cases'].append(out);M.dump(a.out/'centroid_replay.json',result);M.log(f'centroid replay case={ci} complete')
 result['qualified']=all(c['parameters_equal'] and all(x['metric_reproduction_passed'] for x in c['runs']) for c in result['cases'])
 result['all_truth_hashes_identical']=all(c['truth_hash_matches'] for c in result['cases'])
 M.dump(a.out/'centroid_replay.json',result)
 if not result['qualified']:raise RuntimeError('saved metrics not reproduced')
if __name__=='__main__':main()
