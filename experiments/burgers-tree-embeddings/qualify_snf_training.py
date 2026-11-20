"""Qualify the actual pinned SNF data/training path in an isolated Float64 port.

No study fields, dependency resolution, or trained-comparator claim are involved.
"""
import argparse, shutil, difflib, re, tomllib
from common import *
from neural.build_data import resolve
from qualify_native_competitors import command


def replace_once(text, old, new):
    assert text.count(old) == 1, ('Pinned source pattern changed', old)
    return text.replace(old, new)


def precision_literals(text):
    return re.sub(r'(?<![\w.])(\d+(?:\.\d*)?)f([+-]?\d+)', r'\1e\2', text)


AUDIT_HELPERS = '''
# B8 qualification instrumentation: outside differentiated arithmetic.
const B8_FLOAT64_AUDIT = Dict{String,Int}("arrays" => 0, "steps" => 0)
function b8_assert_gpu_float64(x::Union{Tuple,NamedTuple})
    foreach(b8_assert_gpu_float64, x)
    nothing
end
function b8_assert_gpu_float64(x::AbstractArray)
    if eltype(x) <: AbstractFloat
        @assert eltype(x) === Float64
        raw = x isa ComponentArray ? getdata(x) : x
        @assert raw isa CUDA.CuArray
        @assert all(isfinite, raw)
        B8_FLOAT64_AUDIT["arrays"] += 1
    end
    nothing
end
b8_assert_gpu_float64(x) = nothing

'''


def patched_sources(project):
    patches = {}
    name = 'experiments_SNFROM/cases.jl'; original = (project/name).read_text()
    modified = replace_once(original, '@assert eltype(x) === Float32', '@assert eltype(x) === Float64')
    modified = replace_once(modified, '@assert eltype(u) === Float32', '@assert eltype(u) === Float64')
    start = modified.index('function make_optimizer(')
    end = modified.index('\n#======================================================#', start)
    modified = modified[:start]+precision_literals(modified[start:end]).replace('Inf32', 'Inf')+modified[end:]
    patches[name] = original, modified
    name = 'experiments_SNFROM/smoothNF.jl'; original = (project/name).read_text()
    old = '''    # normalize
    x, x̄, σx = normalize_x(x)
    u, ū, σu = normalize_u(u)
    t, t̄, σt = normalize_t(t)
'''
    new = '''    # B8 data contract: fit statistics on training trajectories/times only.
    _, x̄, σx = normalize_x(x[:, Ix])
    _, ū, σu = normalize_u(u[:, Ix, _Ib, _It])
    _, t̄, σt = normalize_t(t[_It])
    x = normalizedata(x, x̄, σx)
    u = normalizedata(u, ū, σu)
    t = normalizedata(t, t̄, σt)
'''
    modified = replace_once(original, old, new)
    start = modified.index('function makedata_SNF(')
    end = modified.index('\n#===========================================================#', start)
    data_function = modified[start:end]
    assert data_function.count('zeros(Float32,') == 4
    modified = modified[:start]+data_function.replace('zeros(Float32,', 'zeros(Float64,')+modified[end:]
    start = modified.index('function train_SNF(')
    end = modified.index('\n#===========================================================#', start)
    training = precision_literals(modified[start:end])
    training = replace_once(training, 'train_model(NN, _data; rng,', 'train_model(NN, _data, data_; rng,')
    modified = modified[:start]+training+modified[end:]
    patches[name] = original, modified
    name = 'src/train.jl'; original = (project/name).read_text()
    modified = replace_once(original,
        'STATS = Int[], Float32[], Float32[], Float32[], Float32[], Float32[], Float32[]',
        'STATS = Int[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]')
    modified = replace_once(modified, '    p, st = (p, st) |> device\n',
        '    p, st = Lux.f64((p, st) |> device)\n    b8_assert_gpu_float64(p)\n')
    modified = replace_once(modified,
        '@assert eltype(p) ∈ (Float32, ComplexF32)', '@assert eltype(p) ∈ (Float64, ComplexF64)')
    modified = replace_once(modified,
        '        for batch in _loader\n            loss = Loss(NN, st, batch, lossfun)',
        '        for batch in _loader\n            b8_assert_gpu_float64(batch)\n            b8_assert_gpu_float64(p)\n            loss = Loss(NN, st, batch, lossfun)')
    modified = replace_once(modified,
        '            l, st, stats, g = grad(loss, p)\n            opt_st, p = Optimisers.update!(opt_st, p, g)',
        '            l, st, stats, g = grad(loss, p)\n            @assert l isa Float64\n            b8_assert_gpu_float64(g)\n            opt_st, p = Optimisers.update!(opt_st, p, g)\n            b8_assert_gpu_float64(p)\n            B8_FLOAT64_AUDIT["steps"] += 1')
    patches[name] = original, AUDIT_HELPERS+modified
    return patches


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--timeout-seconds', type=int, default=1200)
    a = ap.parse_args(); assert 0 < a.timeout_seconds <= 1800
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'result.json').exists()
    metadata = preflight(); ns = Path(os.environ['STUDY_NAMESPACE'])
    native = resolve(ns/'b6native/output/result.json', ns); evidence = json.loads(native.read_text())
    qualified = resolve(ns/'b7snff64/output/result.json', ns); q = json.loads(qualified.read_text()); assert q['passed']
    compat = resolve(ns/'b7snfcompatfix/output/result.json', ns); c = json.loads(compat.read_text())
    old_project = compat.parent/'sources/snf'
    assert digest(old_project/'Manifest.toml') == q['resolved_manifest_sha256'] == c['resolved_manifest_sha256']
    for row in evidence['sources']['snf']['files']:
        assert digest(old_project/row['path']) == row['sha256']
    binary = native.parent/'runtime/julia-1.10.10/bin/julia'
    assert digest(binary) == evidence['julia']['binary_sha256']
    lux_rule = compat.parent/'julia-depot/packages/Lux/68T9j/src/chainrules.jl'
    assert digest(lux_rule) == q['lux_patched_rule_sha256']
    project = out/'sources/snf'; shutil.copytree(old_project, project)
    records, patch = [], []
    for name, (original, modified) in patched_sources(project).items():
        path = project/name; before = digest(path)
        assert path.resolve().is_relative_to(out.resolve())
        path.chmod(path.stat().st_mode | 0o200); path.write_text(modified)
        records.append(dict(file=name, original_sha256=before, patched_sha256=digest(path)))
        patch.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                          fromfile=name, tofile=name+'.b8-float64-training'))
    patchfile = out/'snf-training-float64.patch'; patchfile.write_text(''.join(patch))
    fixture = out/'native_training_fixture.jl'
    shutil.copy2(HERE/'baselines/snf_training_fixture.jl', fixture)
    depot = out/'julia-depot'; depot.mkdir()
    lookup = [depot, qualified.parent/'julia-depot', resolve(ns/'b7snfgpu/output/julia-depot', ns),
              compat.parent/'julia-depot', resolve(ns/'b6snfnocache/output/julia-depot', ns), native.parent/'julia-depot']
    assert all(path.is_dir() for path in lookup)
    env = dict(os.environ, JULIA_DEPOT_PATH=':'.join(map(str, lookup)), JULIA_PKG_PRECOMPILE_AUTO='0',
               JULIA_NUM_THREADS='4', JULIA_NUM_PRECOMPILE_TASKS='4', GKSwstype='100',
               CUDA_CACHE_PATH=str(out/'cuda-cache'), B8_SNF_OUTPUT=str(out), B8_SNF_PROJECT=str(project))
    result = dict(meta=metadata, status='native_training_path_qualification', source_commit=evidence['sources']['snf']['commit'],
                  native_source_result_sha256=digest(native), operational_qualification_sha256=digest(qualified),
                  resolved_manifest_sha256=digest(project/'Manifest.toml'), lux_patched_rule_sha256=digest(lux_rule),
                  patches=records, patch_sha256=digest(patchfile), fixture_sha256=digest(fixture),
                  final_test_opened=False, study_data_opened=False, shared_environment_modified=False,
                  trained_comparator=False,
                  scope='Tiny native SNF makedata_SNF/train_SNF/FlatDecoder optimization fixture; not study training or PDE qualification')
    save(out/'result.json', result)
    result['execution'] = command([str(binary), '--startup-file=no', '--project='+str(project), str(fixture)],
                                  out/'native_training_fixture.log', env, timeout=a.timeout_seconds)
    metrics = out/'training-fixture.toml'
    if metrics.exists():
        result['fixture'] = tomllib.loads(metrics.read_text()); result['fixture_result_sha256'] = digest(metrics)
    result.update(status='complete', passed=result['execution']['returncode'] == 0
                  and bool(result.get('fixture', {}).get('passed', False)))
    save(out/'result.json', result)


if __name__ == '__main__':
    main()
