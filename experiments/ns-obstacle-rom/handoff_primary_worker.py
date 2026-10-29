"""Private fresh native worker for released handoff2d; not a standalone admission builder."""
from pathlib import Path
import argparse,json,os,resource,sys,time
import numpy as np
import materialize_reference2d as m


def reconstruct(original,out,expected_export):
    if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID') or
        os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest'):
        raise RuntimeError('private worker requires released fresh cluster process')
    m.native_digest()
    if any(name[:-3] in sys.modules for name in json.loads((m.NATIVE/'SOURCE.json').read_text())['files']):
        raise RuntimeError('native worker module namespace is not fresh')
    if m.sha(original/'EXPORT-MANIFEST.json')!=expected_export:raise ValueError('adopted source changed in worker')
    sys.path.insert(0,str(m.NATIVE))
    from reference_primary import reconstruct_primary,load_primary_case
    from immutable_rule import array_digest
    import jax,ngsolve as ng
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or jax.config.jax_default_matmul_precision!='highest':raise RuntimeError('GPU/f64/highest required')
    print('jax_backend=gpu',flush=True);started=time.perf_counter()
    cfg,primary,_,_=load_primary_case(original);descriptors=primary['derived_full_rule_arrays']
    expanded=sum(int(np.prod(d['shape']))*np.dtype(d['dtype']).itemsize for d in descriptors.values())
    if expanded>2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+4*expanded+2*2**30>16*2**30:
        raise MemoryError('native reconstruction prospective memory cap')
    fields,replay=reconstruct_primary(original,ng);raw={key:fields['volume'][key] for key in descriptors}
    np.savez_compressed(out/'raw-full-rule.npz',**raw)
    actual={key:{'dtype':a.dtype.str,'shape':list(a.shape),'content_sha256':array_digest(a)} for key,a in raw.items()}
    record={'reconstruction':replay,'actual':actual,'expected':descriptors,'original_export_sha256':expected_export,
        'raw_archive_sha256':m.sha(out/'raw-full-rule.npz'),'elapsed_seconds':time.perf_counter()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'worker_sha256':m.sha(__file__)}
    (out/'REPLAY.json').write_text(json.dumps(record,indent=2)+'\n')
    if actual!=descriptors:raise ValueError('actual primary replay differs; raw counterexample retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-export',required=True);a=p.parse_args()
    reconstruct(a.original,a.out,a.expected_export)
