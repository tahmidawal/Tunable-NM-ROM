"""Run an already selected native POD method on every original in-range E0 case."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from open_sealed import COHORTS, packed, verify_file, sha
from panel import HERE
from published import Opened

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--pod',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_bytes())
    dim=cfg['dim']
    co,n=COHORTS[dim]
    opened=Opened(HERE,args.config)
    for rel, (_, _, _, digest) in json.loads((HERE/"pod-source.json").read_bytes()).items():
        verify_file(args.pod/rel,digest)
    import jcp_frozen_classical_pod as module
    name,initial,run=module.prepare(dim,args.pod/'experiments',cfg['asset'])
    inputs_receipt=json.loads(args.inputs.with_suffix('.receipt.json').read_bytes())
    if (inputs_receipt['dim']!=dim or inputs_receipt['freeze_sha256']!=opened.record['freeze_sha256'] or
            inputs_receipt['published_seed_commitment']!=opened.record['seed_sha256']):
        raise ValueError('POD input provenance disagrees with the approved opening')
    verify_file(args.inputs,inputs_receipt['input_sha256'])
    with np.load(args.inputs,allow_pickle=False) as data:
        inputs=dict(data) if dim==3 else data['rows']
    path=args.base/'vendor/tree/experiments/jcp-references/verdict.py'
    verify_file(path,json.loads((HERE/"upstream.json").read_bytes())["files"]["vendor/tree/experiments/jcp-references/verdict.py"])
    statuses=json.loads(verify_file(args.base/"vendor/tree/experiments/jcp-references/store/STATUS.json",opened.status_sha))
    spec=importlib.util.spec_from_file_location('e0verdict',path)
    verdict=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verdict)
    args.out.mkdir(exist_ok=True)
    (args.out/'cohort-identity.json').write_bytes(packed(inputs_receipt))
    def finite_json(obj):
        if isinstance(obj,dict): return {key:finite_json(value) for key,value in obj.items()}
        if isinstance(obj,(list,tuple)): return [finite_json(value) for value in obj]
        if isinstance(obj,(float,np.floating)) and not np.isfinite(obj): return None
        return obj
    with (args.out/'result.jsonl').open('x') as f:
        for index in range(n):
            field,nu=initial(inputs,index)
            if index==0:
                run(field,nu)  # compile/synchronize before measuring
            reps=cfg['timing_reps_first_six'] if index<6 else 1
            devmat=jnp.ones((2048,2048),jnp.float64)
            for rep in range(reps):
                t0=time.perf_counter()
                while time.perf_counter()-t0<cfg['burnin_seconds']:
                    jax.block_until_ready(devmat@devmat)
                out=run(field,nu)
                row=dict(case=index,rep=rep,subject=name,seconds=out['seconds'],finite=bool(out['finite']),
                         qualification=out['qualification'],
                         reasons=np.asarray(out['reasons']).tolist(),iterations=np.asarray(out['iterations']).tolist())
                if out['finite']:
                    for kind in ('ST','S'):
                        ref,meta=opened.load(dim,co,index,kind)
                        if out['fields'].shape!=ref.shape:
                            raise ValueError('POD lattice disagrees with original E0 reference')
                        e,et=verdict.rel_err(out['fields'],ref,meta['judge']['n0'])
                        row[kind]=dict(e=float(e),et=np.asarray(et).tolist(),eligible=meta['outcome']=='accepted' and statuses['dimension_status'][str(dim)]=='validated',
                                       outcome=meta['outcome'],eps_ref=meta['judge']['eps_ref'])
                    if index in (0,2) and rep==0:
                        np.savez_compressed(args.out/f'case{index:03d}.npz',fields=out['fields'],coeff=out['saved_coefficients'])
                f.write(json.dumps(finite_json(row),allow_nan=False)+'\n')
                f.flush()
    (args.out/'COMPLETE.json').write_bytes(packed(dict(subject=name,n=n,backend=jax.default_backend(),
        x64=jax.config.jax_enable_x64,precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
        source_commit=os.environ['SOURCE_COMMIT'],gpu=jax.devices()[0].device_kind,job_id=os.environ['SLURM_JOB_ID'])))
if __name__=='__main__':
    main()
