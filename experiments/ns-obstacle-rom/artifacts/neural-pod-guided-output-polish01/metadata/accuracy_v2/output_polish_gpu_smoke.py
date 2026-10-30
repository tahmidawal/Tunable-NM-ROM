"""Prospective bounded GPU synthetic QR kernel fixture; no campaign input paths."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from accuracy_v2.fixed_hidden_output_polish import qr_append,solve_factor


def run(directory):
    import jax,jax.numpy as jnp,jaxlib
    out=Path(directory);out.mkdir(exist_ok=False)
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
    rng=np.random.default_rng(30102027);sizes=(1792,1792,83)
    matrices=[rng.normal(size=(n,130))*np.geomspace(.001,1.,130)[None,:] for n in sizes];rights=[rng.normal(size=n) for n in sizes]
    np.savez_compressed(out/'synthetic-inputs.npz',**{k:v for i,(a,b) in enumerate(zip(matrices,rights)) for k,v in [(f'A{i}',a),(f'b{i}',b)]})
    first=jax.jit(lambda a,b:qr_append(None,a,b,jnp));update=jax.jit(lambda r,a,b:qr_append(r,a,b,jnp))
    factor=None;factors=[];start=time.monotonic()
    for index,(a,b) in enumerate(zip(matrices,rights)):
        aa,bb=jax.device_put(a),jax.device_put(b)
        factor=first(aa,bb) if factor is None else update(factor,aa,bb)
        factors.append(np.asarray(factor))
        np.savez_compressed(out/f'completed-factor-{index}.npz',factor=factors[-1])
    raw=out/'raw-qr.npz';arrays={}
    for i,(a,b,r) in enumerate(zip(matrices,rights,factors)):arrays.update({f'A{i}':a,f'b{i}':b,f'R{i}':r})
    np.savez_compressed(raw,**arrays)
    a=np.concatenate(matrices);b=np.concatenate(rights);cutoff=np.finfo(np.float64).eps*max(a.shape)
    solved=solve_factor(factors[-1],float(cutoff));dense=np.linalg.lstsq(a,b,rcond=cutoff)
    rr=a@solved['delta']-b;dense_rr=a@dense[0]-b
    gradient=2*a.T@rr
    checks=out/'checks.npz';np.savez_compressed(checks,delta=solved['delta'],dense_delta=dense[0],actual_gradient=gradient,factor_gradient=solved['factor_gradient'],residual=rr,dense_residual=dense_rr,singular_values=solved['singular_values'])
    result={'backend':jax.default_backend(),'x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision),'devices':[d.device_kind for d in jax.devices()],'jax':jax.__version__,'jaxlib':jaxlib.__version__,'elapsed_seconds':time.monotonic()-start,'row_blocks':list(sizes),'columns':130,'relative_cutoff':float(cutoff),'rank':solved['numerical_rank'],'dense_rank':int(dense[2]),'delta_gap':float(np.max(np.abs(solved['delta']-dense[0]))),'objective_gap':abs(float(rr@rr-dense_rr@dense_rr)),'gradient_gap':float(np.max(np.abs(gradient-solved['factor_gradient']))),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'checks_sha256':hashlib.sha256(checks.read_bytes()).hexdigest(),'source_files':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['output_polish_gpu_smoke.py','fixed_hidden_output_polish.py','spatial_bank.py']},'actual_TRAIN_access':False,'timing_speedup_claim':False,'device_memory':[d.memory_stats() for d in jax.devices()]}
    result['passed']=bool(all(np.isfinite(v).all() for v in arrays.values()) and result['rank']==result['dense_rank']==130 and result['delta_gap']<=1e-8 and result['objective_gap']<=1e-8 and result['gradient_gap']<=1e-8)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');assert result['passed'];return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out),indent=2))
