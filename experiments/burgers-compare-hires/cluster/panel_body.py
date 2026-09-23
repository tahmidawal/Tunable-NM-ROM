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
                          'experiments/b-panel/inputs/directions_qtd02.npz', CHECKPOINT]
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
"$PY" cmp.py --config {a.config} --checkpoint "$TASK_ROOT/{CHECKPOINT}" --inputs "$TASK_ROOT/experiments/b-panel/inputs" --out output
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
mkdir -p output/optiming
'''
    for op in ops['operators']:
        body += (f'"$PY" ops/optime.py --checkpoint opckpt/{op["name"]}.pt --index output/opcohort/index.json '
                 f'--out output/optiming --fields output/fields --name {op["name"]} --role "{op["role"]}" '
                 f'--repetitions 5 --burn-in 20 || echo "OPERATOR FAILED {op["name"]}"\n')
    body += '''"$PY" audit_cmp.py output --operators operators-''' + str(L) + '''.json --out output/audit-remote.json || echo "REMOTE AUDIT FAILED"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    return files, body, extra
