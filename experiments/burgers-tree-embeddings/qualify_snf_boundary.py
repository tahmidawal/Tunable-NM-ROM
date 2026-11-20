"""Qualify a labelled continuous-boundary port on the actual native SNF fixture."""
import argparse, shutil, difflib, tomllib, traceback
from common import *
from neural.build_data import resolve
from qualify_snf_training import replace_once
from qualify_native_competitors import command


def patch_boundary(project):
    """Single reviewed insertion; caller must first verify the native training port."""
    path=project/'experiments_SNFROM/smoothNF.jl';original=path.read_text()
    modified=replace_once(original,'    NN = FlatDecoder(hyper, decoder)',
        '    @assert iszero(α) "Boundary port currently qualifies α=0 only"\n'
        '    decoder = B8BoundaryDecoder(decoder, metadata.x̄, metadata.σx, metadata.ū, metadata.σu)\n'
        '    metadata = (; metadata..., boundary_port="continuous physical Dirichlet envelope; modified new-PDE port")\n'
        '    NN = FlatDecoder(hyper, decoder)')
    before=digest(path);path.chmod(path.stat().st_mode|0o200);path.write_text(modified)
    return dict(file='experiments_SNFROM/smoothNF.jl',original_sha256=before,patched_sha256=digest(path)), ''.join(
        difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='smoothNF.jl',tofile='smoothNF.jl.boundary-port'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--qualification-job',default='b8snftrainfix');ap.add_argument('--timeout-seconds',type=int,default=1200)
    a=ap.parse_args();assert a.qualification_job.replace('-','').isalnum() and 0<a.timeout_seconds<=1800
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    result=dict(status='preparing',passed=False,final_test_opened=False,study_data_opened=False,
        trained_comparator=False,pde_comparator_admitted=False,
        classification='Native SNF plus continuous physical boundary envelope; modified new-PDE port')
    save(out/'result.json',result)
    try:
        assert os.environ.get('SLURM_JOB_ID');result['meta']=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
        qp=resolve(ns/a.qualification_job/'output/result.json',ns);q=json.loads(qp.read_text());assert q['passed'] and q['fixture']['passed']
        assert q['fixture']['flat_parameter_transfer_preserves_values_and_axes']
        manifest=HERE/'manifests'/f'retention-{a.qualification_job}.json';retained=json.loads(manifest.read_text())
        assert retained['local_and_remote_verified'];hashes={row['file']:row['sha256'] for row in retained['files']}
        assert digest(qp)==hashes['output/result.json']
        native=resolve(ns/'b6native/output/result.json',ns);evidence=json.loads(native.read_text())
        assert digest(native)==q['native_source_result_sha256']
        old=qp.parent/'sources/snf';patches={row['file']:row['patched_sha256'] for row in q['patches']}
        for row in evidence['sources']['snf']['files']:
            assert digest(old/row['path'])==patches.get(row['path'],row['sha256'])
        assert digest(old/'Manifest.toml')==q['resolved_manifest_sha256']
        binary=native.parent/'runtime/julia-1.10.10/bin/julia';assert digest(binary)==evidence['julia']['binary_sha256']
        compat=resolve(ns/'b7snfcompatfix/output/result.json',ns)
        rule=compat.parent/'julia-depot/packages/Lux/68T9j/src/chainrules.jl';assert digest(rule)==q['lux_patched_rule_sha256']
        project=out/'sources/snf';shutil.copytree(old,project)
        patch_record,patch=patch_boundary(project);(out/'boundary-port.patch').write_text(patch)
        for name in ['snf_boundary.jl','snf_boundary_fixture.jl']:
            shutil.copy2(HERE/'baselines'/name,out/name)
        base=qp.parent/'native_training_fixture.jl';assert digest(base)==q['fixture_sha256']==hashes['output/native_training_fixture.jl']
        text=base.read_text()
        include='include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "smoothNF.jl"))'
        text=replace_once(text,include,'include(joinpath(out, "snf_boundary.jl"))\ninclude(joinpath(out, "snf_boundary_fixture.jl"))\n'+include)
        text=replace_once(text,'metrics = Dict(', 'boundary_metrics = b8_boundary_checks(native_model)\nmetrics = Dict(')
        text=replace_once(text,'open(joinpath(out, "training-fixture.toml"), "w") do io',
                          'merge!(metrics, boundary_metrics)\nopen(joinpath(out, "training-fixture.toml"), "w") do io')
        fixture=out/'boundary_training_fixture.jl';fixture.write_text(text)
        depot=out/'julia-depot';depot.mkdir()
        lookup=[depot,qp.parent/'julia-depot']+[resolve(ns/job/'output/julia-depot',ns)
            for job in ['b7snff64','b7snfgpu','b7snfcompatfix','b6snfnocache','b6native']]
        assert all(p.is_dir() for p in lookup)
        env=dict(os.environ,JULIA_DEPOT_PATH=':'.join(map(str,lookup)),JULIA_PKG_PRECOMPILE_AUTO='0',
            JULIA_NUM_THREADS='4',JULIA_NUM_PRECOMPILE_TASKS='4',GKSwstype='100',CUDA_CACHE_PATH=str(out/'cuda-cache'),
            B8_SNF_OUTPUT=str(out),B8_SNF_PROJECT=str(project))
        result.update(status='qualifying_boundary_port',qualification_job=a.qualification_job,
            native_training_qualification_sha256=digest(qp),retention_manifest_sha256=digest(manifest),
            source_commit=q['source_commit'],resolved_manifest_sha256=digest(project/'Manifest.toml'),
            lux_patched_rule_sha256=digest(rule),boundary_patch=patch_record,boundary_patch_sha256=digest(out/'boundary-port.patch'),
            boundary_wrapper_sha256=digest(out/'snf_boundary.jl'),boundary_checks_sha256=digest(out/'snf_boundary_fixture.jl'),
            base_fixture_sha256=digest(base),fixture_sha256=digest(fixture))
        save(out/'result.json',result)
        result['execution']=command([str(binary),'--startup-file=no','--project='+str(project),str(fixture)],
            out/'boundary_training_fixture.log',env,timeout=a.timeout_seconds)
        metrics=out/'training-fixture.toml'
        if metrics.exists():result['fixture']=tomllib.loads(metrics.read_text());result['fixture_result_sha256']=digest(metrics)
        result.update(status='complete_boundary_qualification',passed=result['execution']['returncode']==0
            and result.get('fixture',{}).get('passed',False) and result.get('fixture',{}).get('boundary_wrapper_passed',False))
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:save(out/'result.json',result)


if __name__=='__main__':main()
