"""Prospective bounded actual-composite NumPy/JAX fixture; never fit or solve.

Requires an actual selected model and the frozen small point fixture. Execution
needs its own bounded authorization; this module does not submit a job.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from accuracy_v2.residual_head_bank import ResidualHeadBank
from accuracy_v2.independent_residual_heads import evaluate_heads
from accuracy_v2.spatial_bank import evaluate_layers
from accuracy_v2.train import digest

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(model,model_sha256,points_fixture,points_sha256,output):
    model=Path(model);points_fixture=Path(points_fixture);out=Path(output)
    if sha(model)!=model_sha256 or sha(points_fixture)!=points_sha256:raise ValueError('actual model and frozen fixture hashes required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        bank=ResidualHeadBank.load(model)
        with np.load(points_fixture,allow_pickle=False) as a:x=a['points'].copy()
        if x.shape!=(339,2) or x.dtype!=np.float64 or not np.isfinite(x).all():raise ValueError('same339 finitef64 physical points required')
        import jax,jax.numpy as jnp
        if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
        parent=tuple(tuple(jax.device_put(a) for a in layer) for layer in bank.parent.layers)
        heads=tuple(tuple(tuple(jax.device_put(a) for a in layer) for layer in h) for h in bank.heads)
        def evaluate(parent_layers,child_heads,xx):
            pv,pg=evaluate_layers(parent_layers,xx,jnp);cv,cg=evaluate_heads(child_heads,xx,jnp)
            return pv+cv,pg+cg,pv,pg,cv,cg
        jv,jg,pv,pg,cv,cg=map(np.asarray,jax.jit(evaluate)(parent,heads,jax.device_put(x)))
        arrays={'points':x,'jax_values':jv,'jax_physical_gradients':jg,'jax_parent_values':pv,'jax_parent_physical_gradients':pg,'jax_child_values':cv,'jax_child_physical_gradients':cg};summary=[]
        for prefix in (1,2,4):
            nv,ng=bank.evaluate(x,prefix)
            arrays[f'numpy_values_{prefix}']=nv;arrays[f'numpy_physical_gradients_{prefix}']=ng
            arrays[f'jax_values_{prefix}']=jv[:,:,:prefix];arrays[f'jax_physical_gradients_{prefix}']=jg[:,:,:,:prefix]
            summary.append({'prefix':prefix,'value_max_abs_gap':float(np.max(np.abs(nv-jv[:,:,:prefix]))),'physical_gradient_max_abs_gap':float(np.max(np.abs(ng-jg[:,:,:,:prefix]))),'passed':bool(np.allclose(nv,jv[:,:,:prefix],rtol=1e-10,atol=1e-11) and np.allclose(ng,jg[:,:,:,:prefix],rtol=1e-10,atol=1e-10))})
        np.savez_compressed(out/'arrays.npz',**arrays)
        child=np.concatenate([a.ravel() for h in bank.heads for layer in h for a in layer])
        result={'scope':'bounded339point actual composite live NumPy versus JAX formula; no fullrule/PDE/performance/universal bitwise claim','model_sha256':model_sha256,'parent_model_sha256':bank.parent_sha256,'selected_child_theta_sha256':digest(child),'model_schema':'ns-obstacle-independent-residual-head-bank-v1','evaluator_class':'accuracy_v2.residual_head_bank.ResidualHeadBank','points_fixture_sha256':points_sha256,'points_semantic_sha256':digest(x),'arrays_sha256':sha(out/'arrays.npz'),'source_sha256':sha(__file__),'source_files':{n:sha(Path(__file__).with_name(n)) for n in ('residual_head_bank.py','independent_residual_heads.py','spatial_bank.py','fast_spatial_bank.py')},'runtime':{'python':sys.version,'jax':jax.__version__,'backend':jax.default_backend(),'x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision),'device_kind':[d.device_kind for d in jax.devices()]},'prefixes':summary,'sampled_correction_amplitude':{'scope':'same339fixturepoints only; not fullrule or solved-state accuracy','parent_value_rms':float(np.sqrt(np.mean(pv*pv))),'child_value_rms':float(np.sqrt(np.mean(cv*cv))),'parent_physical_gradient_rms':float(np.sqrt(np.mean(pg*pg))),'child_physical_gradient_rms':float(np.sqrt(np.mean(cg*cg))),'parent_child_sum_value_max_abs_gap':float(np.max(np.abs(jv-(pv+cv)))),'parent_child_sum_gradient_max_abs_gap':float(np.max(np.abs(jg-(pg+cg))))},'finite_f64':all(a.dtype==np.float64 and np.isfinite(a).all() for a in arrays.values()),'passed':all(r['passed'] for r in summary)}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if not result['passed'] or not result['finite_f64']:raise AssertionError('composite evaluator fixture failed; raw evidence retained')
        print(json.dumps(result,indent=2));return result
    except BaseException as exc:
        (out/'FAILURE.json').write_text(json.dumps({'exception':repr(exc),'actual_model_sha256':model_sha256,'scientific_data_modified':False})+'\n');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);p.add_argument('--points-fixture',type=Path,required=True);p.add_argument('--points-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.model,a.model_sha256,a.points_fixture,a.points_sha256,a.output)
