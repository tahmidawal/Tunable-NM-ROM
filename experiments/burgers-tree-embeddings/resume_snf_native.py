"""Finite cached native SNF import/PDE fixture; isolated writable depot overlay."""
import argparse,shutil
from common import *
from neural.build_data import resolve
from qualify_native_competitors import command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--no-compiled-modules',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight()
    ns=Path(os.environ['STUDY_NAMESPACE']);original=resolve(ns/'b6native/output/result.json',ns)
    r=json.loads(original.read_text());assert r['snf_instantiate']['returncode']==0
    old=original.parent;binary=old/'runtime/julia-1.10.10/bin/julia'
    assert digest(binary)==r['julia']['binary_sha256']
    assert digest(old/'sources/snf/Manifest.toml')==r['resolved_manifest_sha256']
    source=out/'sources/snf';shutil.copytree(old/'sources/snf',source)
    depot=out/'julia-depot';depot.mkdir()
    # Julia writes caches into the first depot; the retained dependency/artifact
    # depot is only a lookup layer. No Pkg mutation or dependency resolution here.
    env=dict(os.environ,JULIA_DEPOT_PATH=str(depot)+':'+str(old/'julia-depot'),
        JULIA_PKG_PRECOMPILE_AUTO='0',JULIA_NUM_THREADS='4',GKSwstype='100',
        CUDA_CACHE_PATH=str(out/'cuda-cache'))
    fixture=out/'native_fixture.jl'
    fixture.write_text('''using NeuralROMs, CUDA
@assert CUDA.functional()
struct ScalarFixture <: NeuralROMs.AbstractNeuralModel end
function NeuralROMs.dudx2_2D(::ScalarFixture,x,p;kwargs...)
    xx=x[1:1,:]; yy=x[2:2,:]
    u=p[1].*sin.(pi.*xx).*sin.(pi.*yy)
    dx=p[1].*pi.*cos.(pi.*xx).*sin.(pi.*yy)
    dy=p[1].*pi.*sin.(pi.*xx).*cos.(pi.*yy)
    lap=-pi^2 .* u
    vcat(u,u),(vcat(dx,dx),vcat(dy,dy)),(vcat(lap,lap),vcat(lap,lap))
end
x=CuArray(reshape(collect(range(0.01,0.99,length=64)),2,32))
p=[0.3]
uv,der,second=NeuralROMs.dudx2_2D(ScalarFixture(),x,p)
native=NeuralROMs.dudtRHS(BurgersViscous2D(0.005),ScalarFixture(),x,p,0.0)
scalar=0.005.*(second[1][1:1,:]+second[2][1:1,:])-uv[1:1,:].*(der[1][1:1,:]+der[2][1:1,:])
err=maximum(abs.(native-vcat(scalar,scalar)))
@assert eltype(native)==Float64
@assert err<1e-12
println("native_snf_gpu_import_and_scalar_equivalence_pass ",err)
''')
    result=dict(meta=metadata,status='native_snf_cached_import',source_result_sha256=digest(original),
        resolved_manifest_sha256=r['resolved_manifest_sha256'],final_test_opened=False,trained=False)
    save(out/'result.json',result)
    result['compiled_modules_disabled']=a.no_compiled_modules
    flags=['--compiled-modules=no'] if a.no_compiled_modules else []
    result['execution']=command([str(binary),'--startup-file=no']+flags+['--project='+str(source),str(fixture)],
                                out/'native_fixture.log',env,timeout=300)
    log=(out/'native_fixture.log').read_text()
    result.update(status='complete',passed=result['execution']['returncode']==0 and
        'native_snf_gpu_import_and_scalar_equivalence_pass' in log,
        scope='Native import and vector-equal-component/scalar Burgers RHS equivalence only; no trained comparator')
    save(out/'result.json',result)


if __name__=='__main__':main()
