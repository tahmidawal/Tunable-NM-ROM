"""Isolated Lux duplicate-rule compatibility patch with compiled GPU verification."""
import argparse,shutil,difflib
from common import *
from neural.build_data import resolve
from qualify_native_competitors import command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    pilot=resolve(ns/'b7parentpolish/output/result.json',ns);assert json.loads(pilot.read_text())['pilot_pass']
    previous=resolve(ns/'b6native/output/result.json',ns);r=json.loads(previous.read_text());old=previous.parent
    binary=old/'runtime/julia-1.10.10/bin/julia';assert digest(binary)==r['julia']['binary_sha256']
    assert digest(old/'sources/snf/Manifest.toml')==r['resolved_manifest_sha256']
    project=out/'sources/snf';shutil.copytree(old/'sources/snf',project)
    depot=out/'julia-depot';depot.mkdir()
    lux=old/'julia-depot/packages/Lux/68T9j';patched=depot/'packages/Lux/68T9j';shutil.copytree(lux,patched)
    rule=lux/'src/chainrules.jl';original=rule.read_text()
    expected='a75aaeb724150629c2067ef1df30ef9971cafed0aee62f81238b833eea1b203d'
    assert digest(rule)==expected
    start=original.index('function CRC.rrule(::typeof(merge),')
    end=original.index('\nfunction CRC.rrule(::typeof(_eachslice)',start)
    modified=original[:start]+'# Owned compatibility patch: use the installed ChainRules merge rule.\n'+original[end:]
    patched_rule=patched/'src/chainrules.jl'
    # copytree preserves Julia package read-only modes; change only our private copy.
    assert patched_rule.resolve().is_relative_to(out.resolve())
    patched_rule.chmod(patched_rule.stat().st_mode | 0o200)
    patched_rule.write_text(modified)
    (out/'lux-merge-compatibility.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='Lux-original/src/chainrules.jl',tofile='Lux-patched/src/chainrules.jl')))
    lookup=[str(depot),str(resolve(ns/'b6snfnocache/output/julia-depot',ns)),str(old/'julia-depot')]
    env=dict(os.environ,JULIA_DEPOT_PATH=':'.join(lookup),JULIA_PKG_PRECOMPILE_AUTO='0',JULIA_NUM_THREADS='4',
        JULIA_NUM_PRECOMPILE_TASKS='4',GKSwstype='100',CUDA_CACHE_PATH=str(out/'cuda-cache'))
    fixture=out/'native_fixture.jl'
    fixture.write_text('''using NeuralROMs, CUDA, Lux, Zygote, Random
@assert CUDA.functional()
CUDA.allowscalar(false)
v=CUDA.ones(Float64,32)
@assert maximum(abs.(v.*2 .- 2)) == 0
function merged_loss(a,b)
    nt=merge((x=[a,a*a],y=[a]),(y=[b],z=[b*b]))
    sum(nt.x)+sum(nt.y)+sum(nt.z)
end
g=Zygote.gradient(merged_loss,2.0,3.0)
@assert maximum(abs.([g[1]-5,g[2]-7]))<1e-12
gg=Zygote.gradient(a->first(Zygote.gradient(x->merged_loss(x,3.0),a)),2.0)
@assert abs(gg[1]-2)<1e-12
rng=MersenneTwister(2026111963)
model=Lux.Chain(Lux.Dense(2,16,tanh),Lux.Dense(16,1))
ps,st=Lux.setup(rng,model)
device=Lux.gpu_device()
ps=device(Lux.f64(ps));st=device(st)
x=CuArray(reshape(collect(range(0.01,0.99,length=64)),2,32))
loss(p)=sum(abs2,first(model(x,p,st)))
grad=Zygote.gradient(loss,ps)[1]
@assert eltype(grad.layer_1.weight)==Float64
@assert isfinite(maximum(abs.(grad.layer_1.weight)))
direction=CUDA.ones(Float64,size(ps.layer_1.weight))
hstep=1e-6
pp=merge(ps,(layer_1=merge(ps.layer_1,(weight=ps.layer_1.weight+hstep.*direction,)),))
pm=merge(ps,(layer_1=merge(ps.layer_1,(weight=ps.layer_1.weight-hstep.*direction,)),))
fd=(loss(pp)-loss(pm))/(2*hstep);ad=sum(grad.layer_1.weight)
@assert abs(fd-ad)/(1+abs(fd))<1e-6
struct ScalarFixture <: NeuralROMs.AbstractNeuralModel end
function NeuralROMs.dudx2_2D(::ScalarFixture,x::AbstractArray,p::AbstractVector;kwargs...)
    xx=x[1:1,:];yy=x[2:2,:]
    u=p[1].*sin.(pi.*xx).*sin.(pi.*yy)
    dx=p[1].*pi.*cos.(pi.*xx).*sin.(pi.*yy)
    dy=p[1].*pi.*sin.(pi.*xx).*cos.(pi.*yy)
    lap=-pi^2 .*u
    vcat(u,u),(vcat(dx,dx),vcat(dy,dy)),(vcat(lap,lap),vcat(lap,lap))
end
p=[0.3]
uv,der,second=NeuralROMs.dudx2_2D(ScalarFixture(),x,p)
native=NeuralROMs.dudtRHS(BurgersViscous2D(0.005),ScalarFixture(),x,p,0.0)
scalar=0.005.*(second[1][1:1,:]+second[2][1:1,:])-uv[1:1,:].*(der[1][1:1,:]+der[2][1:1,:])
err=maximum(abs.(native-vcat(scalar,scalar)))
@assert eltype(native)==Float64 && err<1e-12
println("native_snf_compiled_gpu_gradient_and_scalar_pass ",err," gradient_error=",abs(fd-ad)/(1+abs(fd)))
''')
    result=dict(meta=metadata,status='compiled_native_qualification',source_result_sha256=digest(previous),
        resolved_manifest_sha256=r['resolved_manifest_sha256'],lux_original_rule_sha256=digest(rule),
        lux_patched_rule_sha256=digest(patched/'src/chainrules.jl'),patch_sha256=digest(out/'lux-merge-compatibility.patch'),
        compiled_modules=True,shared_environment_modified=False,final_test_opened=False,trained=False)
    save(out/'result.json',result)
    result['execution']=command([str(binary),'--startup-file=no','--project='+str(project),str(fixture)],out/'native_fixture.log',env,timeout=1500)
    log=(out/'native_fixture.log').read_text()
    result.update(status='complete',passed=result['execution']['returncode']==0 and 'native_snf_compiled_gpu_gradient_and_scalar_pass' in log,
        scope='Operational compatibility patch, GPU gradients and scalar-equivalent native PDE fixture only; no trained comparator')
    save(out/'result.json',result)


if __name__=='__main__':main()
