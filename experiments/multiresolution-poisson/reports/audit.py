"""Read-only audit of collected provenance, precision, timing and accuracy rows."""
import argparse
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import math

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
run=a.run.resolve();tree=Path(__file__).resolve().parents[3]
submission=json.loads((run/'submission.json').read_text())
result=json.loads((run/'result.json').read_text())
cleanup=json.loads((run/'cleanup.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
archive=run/'verified-cluster.tar.gz'
assert sha(archive.read_bytes())==cleanup['archive_sha256']
assert cleanup['remote_deleted_and_absence_checked']
assert result['provenance']['backend']=='gpu' and result['provenance']['x64']
assert result['provenance']['matmul_precision']=='highest'
assert result['provenance']['commit']==submission['source_commit']
assert result['provenance']['job_id']==submission['job_id']
origin=json.loads((run/'checkpoint-origin.json').read_text())
map_path={
    'sep_common.py':'experiments/separable-decoder/sep_common.py',
    'ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py',
    'ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
with tarfile.open(archive) as tar:
    for staged,wanted in submission['source_hashes'].items():
        content=tar.extractfile('cluster/'+staged).read()
        assert sha(content)==wanted
        filename=Path(staged).name
        source=map_path.get(filename,'experiments/multiresolution-poisson/'+filename)
        committed=subprocess.check_output(['git','show',submission['source_commit']+':'+source],cwd=tree)
        assert content==committed
    checkpoint=tar.extractfile('cluster/in/model.pkl').read()
    assert sha(checkpoint)==result['checkpoint_sha256']==origin['checkpoint_sha256']
    committed=subprocess.check_output(['git','show',submission['source_commit']+':'+origin['checkpoint_path']],cwd=tree)
    assert checkpoint==committed
    assert tar.extractfile('cluster/out/pilot/result.json').read()==(run/'result.json').read_bytes()
log=(run/(submission['job_id']+'.out')).read_text()+(run/(submission['job_id']+'.err')).read_text()
assert 'jax_backend=gpu' in log and 'ALL-DONE' in log
for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback']:
    assert bad not in log,bad
assert (run/'EXIT_CODE').read_text().strip()=='0'
keys=set();checksum_groups={}
for r in result['rows']:
    key=(r['intervals'],r['arm'],r['tau'],r['case'],r['repetition'])
    assert key not in keys;keys.add(key)
    for name in ['total_seconds','physical_error','same_grid_error','conservative_physical_error']:
        assert r[name] is not None and math.isfinite(r[name])
    assert r['total_seconds']>0 and r['conservative_physical_error']>=r['physical_error']
    assert r['scipy_dst_parity']<1e-11
    assert abs(r['total_seconds']-sum(r[x] for x in ['input_seconds','projection_init_seconds','solver_seconds','output_seconds']))<1e-10
    ckey=key[:-1]
    checksum_groups.setdefault(ckey,set()).add(r['field_sha256'])
assert all(len(x)==1 for x in checksum_groups.values())
for ref in result['references']:
    assert ref['reference_converging']
    assert ref['uncertainty']<=min(result['config']['targets'])*result['config']['reference_fraction']
for setup in result['setup']:
    assert setup['retained_modes']>result['checkpoint_config']['k']
    assert setup['retained_bank_rank']==setup['stored_features']
    assert max(max(g.values()) for g in setup['weak_operator_gates'])<1e-9
report=dict(passed=True,source_commit=submission['source_commit'],job_id=submission['job_id'],
    archive_sha256=cleanup['archive_sha256'],row_count=len(keys),
    source_and_checkpoint_match_git_history=True,archived_json_equals_export=True,
    timing_components_sum=True,repeated_outputs_identical=True,
    reference_resolves_all_declared_uncertainty_budgets=True,
    no_accepted_gpu_warning_patterns=True,remote_cleaned=True,
    limitations=['Single development cohort; independent confirmation not run',
        'Reference bound empirical, based on nested FD refinement',
        'Host-source to host-field contract includes component synchronization overhead',
        'No per-resolution training or exhaustive classical mesh selection'])
(run/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
