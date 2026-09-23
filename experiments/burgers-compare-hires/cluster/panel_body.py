"""The panel job's file list and sbatch body (imported by cluster/stage.py for `stage.py panel ...`).

Operator checkpoints are not committed (hundreds of MB each). They are staged from the collected training job
(`runs/<train attempt>/archive/out/<arm>/best.pt`) and from b-panel's archive for the zero-shot FNO, and each one
is re-hashed against the SHA256 recorded in the COMMITTED `operators-<L>.json`; a mismatch refuses to stage.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def files_and_body(ROOT, LANE, LIBS, OPS, CHECKPOINT, cfg, a):
    L = int(cfg['intervals'])
    files = LIBS + OPS + [f'{LANE}/cmp.py', f'{LANE}/sfit.py', f'{LANE}/gridarm.py', f'{LANE}/audit_cmp.py',
                          f'{LANE}/lib/hops.py', f'{LANE}/lib/hfast.py', f'{LANE}/lib/xfast.py', f'{LANE}/{a.config}',
                          'experiments/b-panel/inputs/directions_qtd02.npz', CHECKPOINT,
                          f'{LANE}/inputs/rotation_R512.npz', f'{LANE}/inputs/rotation_R512.json']
    for rung in cfg['rungs']:
        for rs in rung['rules']:
            files += [f"experiments/b-panel/inputs/{part['file']}" for part in rs['parts'] if 'file' in part]
    opfile = f'{LANE}/operators-{L}.json'
    ops = json.loads((ROOT / opfile).read_text())
    files.append(opfile)
    extra = []
    for op in ops['operators']:
        src = ROOT / op['local_path'] if not op['local_path'].startswith('/') else Path(op['local_path'])
        got = sha(src)
        assert got == op['sha256'], (op['name'], got, op['sha256'])
        extra.append((src, f"{LANE}/opckpt/{op['name']}.pt"))
    body = f'''cd {LANE}
"$PY" cmp.py --config {a.config} --checkpoint "$TASK_ROOT/{CHECKPOINT}" --inputs "$TASK_ROOT/experiments/b-panel/inputs" --out output || echo "CMP FAILED (operators and audit still run)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
mkdir -p output/optiming
'''
    tij = getattr(a, 'train_in_job', None)
    if tij:                                       # DESIGN A4: an operator trained inside this allocation (H200-only arm)
        files += [f'{LANE}/opdata.py', f'{LANE}/specs/{tij}.json', f'{LANE}/inputs/pinned/train-index.json',
                  f'{LANE}/inputs/pinned/validation-index.json', f'{LANE}/inputs/opconfigs/fno-large.json',
                  f'{LANE}/inputs/opconfigs/don-small.json']
        body += (f'"$PY" opdata.py --mesh {L} --out data && mkdir -p logs && "$PY" ops/worker.py specs/{tij}.json '
                 f'|| echo "IN-JOB TRAINING FAILED"\n'
                 f'"$PY" cluster/late_ops.py operators-{L}.json "$PWD" '
                 f'{",".join(x["name"] for x in json.loads((ROOT / LANE / "specs" / (tij + ".json")).read_text())["arms"])} 60 '
                 f'output/operators-injob.json || cp operators-{L}.json output/operators-injob.json\n'
                 f'rm -rf data\n')
        base_rec = 'output/operators-injob.json'
        for x in json.loads((ROOT / LANE / 'specs' / (tij + '.json')).read_text())['arms']:
            arm = x['name']
            body += (f'[ -f opckpt/{arm}.pt ] && "$PY" ops/optime.py --checkpoint opckpt/{arm}.pt --index output/opcohort/index.json '
                     f'--out output/optiming --fields output/fields --name {arm} --role "retrained at {L}^2 (in this allocation)" '
                     f'--repetitions 5 --burn-in 20 || echo "IN-JOB OPERATOR MISSING OR FAILED {arm}"\n')
    else:
        base_rec = f'operators-{L}.json'
    late = getattr(a, 'late', None)
    if late:                                      # DESIGN A3: arms from a concurrently running training job
        tjob, arms_, wait = late.split(':')
        files.append(f'{LANE}/cluster/late_ops.py')
        body += (f'"$PY" cluster/late_ops.py {base_rec} /cluster/tufts/paralab/tawal01/bcmp_20260923/{tjob}/{LANE} '
                 f'{arms_} {wait} output/operators-runtime.json || cp {base_rec} output/operators-runtime.json\n')
        for arm in arms_.split(','):
            body += (f'[ -f opckpt/{arm}.pt ] && "$PY" ops/optime.py --checkpoint opckpt/{arm}.pt --index output/opcohort/index.json '
                     f'--out output/optiming --fields output/fields --name {arm} --role "retrained at {L}^2 (late, {tjob})" '
                     f'--repetitions 5 --burn-in 20 || echo "LATE OPERATOR MISSING OR FAILED {arm}"\n')
    else:
        body += f'cp {base_rec} output/operators-runtime.json\n'
    for op in ops['operators']:
        body += (f'"$PY" ops/optime.py --checkpoint opckpt/{op["name"]}.pt --index output/opcohort/index.json '
                 f'--out output/optiming --fields output/fields --name {op["name"]} --role "{op["role"]}" '
                 f'--repetitions 5 --burn-in 20 || echo "OPERATOR FAILED {op["name"]}"\n')
    body += '''"$PY" audit_cmp.py output --operators output/operators-runtime.json --out output/audit-remote.json || echo "REMOTE AUDIT FAILED"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    return files, body, extra
