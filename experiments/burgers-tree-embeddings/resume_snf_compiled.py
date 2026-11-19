"""Correct the owned fixture's missing CUDA adapter, reusing verified compiled caches."""
import argparse
from common import *
from neural.build_data import resolve
from qualify_native_competitors import command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    previous=resolve(ns/'b7snfcompatfix/output/result.json',ns);r=json.loads(previous.read_text());old=previous.parent
    assert not r['passed'] and r['compiled_modules']
    rule=old/'julia-depot/packages/Lux/68T9j/src/chainrules.jl';assert digest(rule)==r['lux_patched_rule_sha256']
    native=resolve(ns/'b6native/output/result.json',ns);n=json.loads(native.read_text())
    binary=native.parent/'runtime/julia-1.10.10/bin/julia';assert digest(binary)==n['julia']['binary_sha256']
    project=old/'sources/snf';assert digest(project/'Manifest.toml')==r['resolved_manifest_sha256']
    original=old/'native_fixture.jl';fixture=out/'native_fixture.jl'
    source=original.read_text().replace('using NeuralROMs, CUDA, Lux, Zygote, Random','using NeuralROMs, CUDA, Lux, LuxCUDA, Zygote, Random')
    source=source.replace('ps=device(Lux.f64(ps));st=device(st)','ps=device(Lux.f64(ps));st=device(st)\n@assert ps.layer_1.weight isa CuArray\n@assert ps.layer_2.weight isa CuArray')
    fixture.write_text(source);depot=out/'julia-depot';depot.mkdir()
    lookup=[depot,old/'julia-depot',resolve(ns/'b6snfnocache/output/julia-depot',ns),native.parent/'julia-depot']
    env=dict(os.environ,JULIA_DEPOT_PATH=':'.join(map(str,lookup)),JULIA_PKG_PRECOMPILE_AUTO='0',JULIA_NUM_THREADS='4',
        JULIA_NUM_PRECOMPILE_TASKS='4',GKSwstype='100',CUDA_CACHE_PATH=str(out/'cuda-cache'))
    result=dict(meta=meta,source_result_sha256=digest(previous),original_fixture_sha256=digest(original),fixture_sha256=digest(fixture),
        lux_patched_rule_sha256=digest(rule),resolved_manifest_sha256=r['resolved_manifest_sha256'],shared_environment_modified=False,
        fixture_correction='Load LuxCUDA trigger and assert weights are on GPU before neural gradients',final_test_opened=False,trained=False)
    save(out/'result.json',result)
    result['execution']=command([str(binary),'--startup-file=no','--project='+str(project),str(fixture)],out/'native_fixture.log',env,timeout=720)
    result.update(status='complete',passed=result['execution']['returncode']==0 and 'native_snf_compiled_gpu_gradient_and_scalar_pass' in (out/'native_fixture.log').read_text(),
        scope='Patched native GPU gradient and scalar-equivalent PDE fixture, not a trained accuracy/runtime comparator')
    save(out/'result.json',result)


if __name__=='__main__':main()
