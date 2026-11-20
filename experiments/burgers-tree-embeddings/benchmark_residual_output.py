"""Same-invocation actual4096 output benchmark. No trajectory physics changes."""
import argparse,gc
from common import *
from training_io import read_checkpoint
from neural.build_data import resolve
from neural import packed_residual as P
from neural.cached_output import OutputCache


def memory():
    stats=jax.devices()[0].memory_stats()
    return {k:int(v) for k,v in (stats or {}).items() if isinstance(v,(int,np.integer))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--source-job',default='b8treepde')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/a.source_job/'output/result.json',ns);r=json.loads(rp.read_text())
    assert r['status'].startswith('complete_fixed_tree') and len(r['outputs'])==5 and not r['final_test_opened']
    mp=resolve(ns/r['model_job']/'output/nonlinear_tree.pkl',ns);assert digest(mp)==r['model_sha256']
    params=jax.tree.map(jnp.asarray,read_checkpoint(mp)['params']);cells=np.asarray(r['cells'])
    tp=rp.parent/r['trajectory_codes_file'];assert digest(tp)==r['trajectory_codes_sha256']
    trajectory=np.load(tp);codes=[]
    for row in r['outputs']:
        idx=np.flatnonzero(np.isclose(trajectory['times'],row['time'],atol=1e-12,rtol=0));assert len(idx)==1
        codes.append(jnp.asarray(trajectory['codes'][idx[0]]))
        path=rp.parent/row['fields']['file'];assert digest(path)==row['fields']['sha256']
    result=dict(meta=metadata,status='running',source_job=a.source_job,source_result_sha256=digest(rp),model_sha256=digest(mp),
        trajectory_sha256=digest(tp),full_shape=[4096,4096],final_test_opened=False,method_admitted=False,
        physics_changed=False,qualification_claim=False,blocks=[],
        timing_contract='Each variant has fresh compilation caches; burn before its measured block. Includes preparation, JIT compilation, coordinate transfers, all actual4096 device output. Parity host transfers/source reads excluded and separately timed. Cache construction charged once across all five outputs. Fixed order, one repetition; not a definitive runtime comparison.',
        memory_note='Device allocator cumulative peak cannot be reset; per-block before/after snapshots plus logical cache bytes are reported. No claim that cumulative peak is a variant-specific peak.')
    save(out/'result.json',result)
    for variant in ['baseline16','host_jit64','cached64']:
        gc.collect();jax.clear_caches();conditioning=burn_gpu();before=memory();compilation=0.;assembly=0.;cache_bytes=0
        setup_start=time.perf_counter()
        optimized=None;kernels={};axis=np.linspace(0.,1.,4096);kernel=jax.jit(P.apply)
        if variant!='baseline16':optimized=OutputCache(params,cells,rows=64)
        setup_seconds=time.perf_counter()-setup_start
        if variant=='cached64':
            start=time.perf_counter();cache_bytes=optimized.build();assembly=time.perf_counter()-start
        row=dict(variant=variant,conditioning=conditioning,memory_before=before,cache_build_seconds=assembly,
                 logical_cache_bytes=cache_bytes,setup_seconds=setup_seconds,outputs=[])
        total=setup_seconds+assembly;parity_total=0.
        for source,code in zip(r['outputs'],codes):
            start=time.perf_counter()
            if variant=='baseline16':
                chunks=[]
                for lo in range(0,4096,16):
                    points=jnp.asarray(np.stack(np.meshgrid(axis[lo:lo+16],axis,indexing='ij'),-1).reshape(-1,2))
                    packed=P.prepare(params['feature'],cells,points);key=tuple(len(v) for v in packed['indices'])
                    if key not in kernels:
                        begin=time.perf_counter();kernels[key]=kernel.lower(params,code,points,packed).compile();compilation+=time.perf_counter()-begin
                    chunks.append(kernels[key](params,code,points,packed).reshape(16,4096))
                output=jnp.concatenate(chunks,axis=0);jax.block_until_ready(output)
            else:output=optimized.reconstruct(code,reuse=variant=='cached64')
            seconds=time.perf_counter()-start;total+=seconds
            begin=time.perf_counter();host=np.asarray(output)
            with np.load(rp.parent/source['fields']['file']) as f:truth=f['prediction']
            assert host.shape==truth.shape==(4096,4096) and host.dtype==truth.dtype==np.float64
            maxerr=0.;ss=den=0.
            for lo in range(0,4096,128):
                delta=host[lo:lo+128]-truth[lo:lo+128]
                assert np.isfinite(delta).all()
                maxerr=max(maxerr,float(np.max(np.abs(delta))));ss+=float(np.sum(delta**2));den+=float(np.sum(truth[lo:lo+128]**2))
            relative=float(np.sqrt(ss/den));assert maxerr<1e-11 and relative<1e-10,(variant,maxerr,relative)
            parity=time.perf_counter()-begin;parity_total+=parity
            row['outputs'].append(dict(time=source['time'],device_output_seconds=seconds,maximum_absolute_difference=maxerr,
                relative_l2_difference=relative,parity_seconds=parity,retained_prediction_sha256=source['fields']['sha256']))
            del output,host,truth
            if variant=='baseline16':del chunks,points,packed
            row.update(charged_five_output_seconds_so_far=total,excluded_parity_seconds=parity_total,memory_after=memory())
            result['active_block']=row;save(out/'result.json',result)
            print('output_profile',variant,source['time'],seconds,maxerr,relative,flush=True)
        row.update(charged_five_output_seconds=total,explicit_compilation_seconds=compilation if optimized is None else optimized.compilation_seconds,
            compilation_note='Subset of charged total; baseline child-feature implicit JIT compilation remains within reconstruction time',
            memory_after=memory())
        result['blocks'].append(row);result.pop('active_block',None);save(out/'result.json',result)
        del optimized,kernels,kernel;gc.collect()
    result['status']='complete_equivalent_output_profile';save(out/'result.json',result)


if __name__=='__main__':main()
