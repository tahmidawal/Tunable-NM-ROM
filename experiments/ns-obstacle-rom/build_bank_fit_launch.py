"""Prepare only the bank launch source; real prepared data/release stay pending."""
from pathlib import Path
import argparse, json
from build_preparation_launch import source_closure, ROOT
from fe_adapter import sha_file
import stage2d


def bank_sources():
    return sorted(set(source_closure()) | {'stage2d.py','paired_model2d.py','validation2d.py','bank_fit_launch2d.py'})


def build(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('payload/\n')
    outer = (ROOT/'staging/fine-preparation01/native.sh').read_text()
    outer = outer.replace('ctol_nsobsrom_prepare','ctol_nsobsrom_bank').replace('00:22:30','00:35:00').replace('1335s','2085s')
    (out/'native.sh').write_text(outer)
    body = '''#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?reviewed batch allocation required}"
JOB_DIR=$(pwd -P)
[ "$JOB_DIR" = /cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/bank-fit01 ] || exit 64
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" CUDA_CACHE_PATH="$JOB_DIR/cache/cuda"
export MPLCONFIGDIR="$JOB_DIR/cache/matplotlib"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" results
JOB_PHASE=source_verification
finish() {
  task_status=$?
  trap - EXIT
  set +e
  "$PY" - "$task_status" "$JOB_PHASE" <<'PY'
import json,sys
from pathlib import Path
p=Path('results/fit/STAGE-RESULT.json');r=json.loads(p.read_text()) if p.exists() else {}
Path('results/batch_summary.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'last_stage':sys.argv[2],
'training_completed':r.get('status')=='execution_completed_pending_independent_review',
'physical_admission_created':False,'automatic_retry':False},indent=2)+'\\n')
PY
  exit "$task_status"
}
trap finish EXIT
trap 'JOB_PHASE=allocation_deadline; exit 143' TERM
sha256sum -c SOURCE-SHA256SUMS
cat SOURCE_COMMIT.txt
"$PY" bank_fit_launch2d.py --candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json --inventory RUNTIME-INVENTORY.json --verify-only
JOB_PHASE=gpu_preflight
"$PY" - <<'PY'
import json,sys,platform,hashlib,jax,jaxlib,optax,numpy,scipy
from pathlib import Path
r={'python':platform.python_version(),'jax':jax.__version__,'jaxlib':jaxlib.__version__,'optax':optax.__version__,
'numpy':numpy.__version__,'scipy':scipy.__version__,'jax_backend':jax.default_backend(),
'jax_x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision),
'python_executable_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
Path('results/runtime.json').write_text(json.dumps(r,indent=2)+'\\n')
print('jax_backend='+r['jax_backend'],flush=True);print(json.dumps(r),flush=True)
sys.exit(0 if r['jax_backend']=='gpu' and r['jax_x64'] and r['matmul_precision']=='highest' else 42)
PY
lscpu
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
JOB_PHASE=fixed_bank_fit
"$PY" bank_fit_launch2d.py --candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json --inventory RUNTIME-INVENTORY.json
JOB_PHASE=completed
'''
    (out/'native-body.sh').write_text(body)
    rows = [{'source':name,'destination':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha_file(ROOT/name)} for name in bank_sources()]
    for name in ('native.sh','native-body.sh'):
        path = out/name
        rows.append({'source':str(path.relative_to(ROOT)),'destination':name,'bytes':path.stat().st_size,'sha256':sha_file(path)})
    inventory = {'schema':'ns-obstacle-bank-fit-runtime-inventory-v1','files':rows,
                 'stage_source_sha256':stage2d.source_digest(),'builder_sha256':sha_file(__file__)}
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    plan = {'schema':'ns-obstacle-bank-fit-source-plan-v1','execution_released':False,
        'actual_stage_candidate_pending':True,'actual_preparation_and_replay_review_pending':True,
        'job_directory':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/bank-fit01',
        'allocation_seconds':2100,'outer_timeout_seconds':2085,'outer_kill_grace_seconds':10,
        'optimizer_time_cap_seconds':1800,'fixed_steps_cap':3000,'fixed_seed':27102027,
        'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),
        'launcher_sha256':sha_file(ROOT/'bank_fit_launch2d.py'),'stage_source_sha256':stage2d.source_digest(),
        'input_contract':'Existing stage2d candidate binds every file in actual accepted prepared tree and separately reviewed consumer release.',
        'prepared_storage':'A hash-bound immutable warm PointVolumeStore is required; compact primary alone is not a consumer input.',
        'reconstruction_if_needed':'Use existing primary_volume.materialize with exact source/runtime; charge all reconstruction and setup within released resources.',
        'validation_data_access':False,'automatic_retry':False,'resume_authorized':False,
        'pending_before_execution':['actual fine preparation/full replay/retention acceptance',
            'actual consumer release including measured-capacity and native-result review bindings',
            'actual train stage candidate from existing stage2d builder; all data paths must exist on execution host',
            'fresh stage-specific memory/disk/global quota and unique job-directory checks',
            'exact separate train release plus launch/runtime/candidate bindings and SOURCE-SHA256SUMS']}
    (out/'SOURCE-PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(p.parse_args().out),indent=2))
