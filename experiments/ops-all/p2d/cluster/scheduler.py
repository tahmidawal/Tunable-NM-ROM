"""Keep at most CAP p2d jobs queued/running; submit training jobs in plan order, and each cell's panel once its four
training jobs have left the queue. Pulls small outputs (JSON + logs, never checkpoints) of finished jobs.

    python cluster/scheduler.py [--once]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / 'runs'
NS = '/cluster/tufts/paralab/tawal01/opsall_20260924/p2d'
CAP = 2   # coordinator 2026-09-24 ~17:40: Burgers 2D / Heat 2D priority
H200_MAX = 2
PANEL_NAME = {'s256': 's256q', 'l2048': 'l2048r'}   # s256p (4290446) failed at import (ModuleNotFoundError pbk3_core); retired
FAMS = ['fno', 'unet', 'transolver', 'deeponet']
ANY = 'a100-80G|h100-80G|h200-141G'

# cell: (problem, mesh, name prefix, lshape train count, cg batch)
CELLS = [('square', 256, 's256', 0, 32), ('lshape', 256, 'l256', 2611, 256), ('lshape', 512, 'l512', 2611, 128),
         ('square', 1024, 's1024', 0, 32), ('lshape', 1024, 'l1024', 1024, 64), ('square', 4096, 's4096', 0, 32),
         ('square', 2048, 's2048', 0, 32), ('lshape', 2048, 'l2048', 256, 32),
         ('square', 1024, 's1024r', 0, 32)]   # DESIGN A6: one network per GPU rerun
SQ_CG = [0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03, 0.01, 0.001, 0.0001]
LS_CG = [0.7, 0.5, 0.3, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001]
SQ_ROWS = {256: (255, 64), 1024: (128, 64), 2048: (256, 64), 4096: (128, 32)}   # pbk config-aba-*.json
SQ_REF = json.loads((HERE / 'configs' / 'table1_reference.json').read_text())


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True).stdout


# plan: (cell, job, members [config files], mem_share, gpu constraint, host mem)
H2 = 'h200-141G'
FOUR = ['fno.json', 'unet.json', 'transolver.json', 'deeponet.json']
PLAN = [('s256', 's256f', ['fno.json'], 1.0, ANY, '128G'), ('s256', 's256u', ['unet.json'], 1.0, ANY, '128G'),
        ('s256', 's256t', ['transolver.json'], 1.0, ANY, '128G'), ('s256', 's256d', ['deeponet.json'], 1.0, ANY, '128G'),
        ('l256', 'l256a', FOUR, 0.24, ANY, '128G'), ('l512', 'l512a', FOUR, 0.24, ANY, '128G'),
        ('s1024', 's1024a', ['fno.json', 'transolver.json'], 0.5, ANY, '128G'),
        ('s1024', 's1024b', ['unet.json', 'deeponet.json'], 0.5, ANY, '128G'),
        ('l1024', 'l1024a', ['fno.json', 'transolver.json'], 0.5, ANY, '160G'),
        ('l1024', 'l1024b', ['unet.json', 'deeponet.json'], 0.5, ANY, '160G'),
        ('s4096', 's4096fb', ['fno.json'], 1.0, ANY, '240G'), ('s4096', 's4096ub', ['unet_small.json'], 1.0, ANY, '240G'),
        ('s4096', 's4096tb', ['transolver_small.json'], 1.0, ANY, '240G'), ('s4096', 's4096d', ['deeponet.json'], 1.0, ANY, '240G'),
        ('s2048', 's2048fc', ['fno.json'], 1.0, ANY, '240G'), ('s2048', 's2048ub', ['unet.json'], 1.0, ANY, '128G'),
        ('s2048', 's2048b', ['transolver.json', 'deeponet.json'], 0.5, ANY, '128G'),
        ('l2048', 'l2048fc', ['fno.json'], 1.0, ANY, '240G'), ('l2048', 'l2048u', ['unet.json'], 1.0, ANY, '160G'),
        ('l2048', 'l2048b', ['transolver.json', 'deeponet.json'], 0.5, ANY, '160G'),
        ('s1024r', 's1024rf', ['fno.json'], 1.0, ANY, '128G'), ('s1024r', 's1024ru', ['unet.json'], 1.0, ANY, '128G'),
        ('s1024r', 's1024rt', ['transolver.json'], 1.0, ANY, '128G'), ('s1024r', 's1024rd', ['deeponet.json'], 1.0, ANY, '128G')]
CHECKPOINT_BLOCKS = {'s2048fc', 'l2048fc'}   # DESIGN A4
LEGACY_SINGLE = {'s256f', 's256u', 's256t', 's256d'}   # staged before co-scheduling: output/<stem>/best.pt


def cell_info(pre):
    return [c for c in CELLS if c[2] == pre][0]


def train_items():
    out = []
    for pre, job, members, share, gpu, mem in PLAN:
        problem, mesh, _, ntr, cgb = cell_info(pre)
        args = ['train', job, '--problem', problem, '--mesh', str(mesh), '--train-count', str(ntr), '--cg-batch', str(cgb),
                '--gpu-constraint', gpu, '--mem', mem] + (['--time', '04:30:00'] if pre == 'l2048' else [])
        if job not in LEGACY_SINGLE:
            args += ['--members', ','.join(members), '--mem-share', str(share)]
        if job in CHECKPOINT_BLOCKS:
            args += ['--checkpoint-blocks']
        out.append(dict(job=job, cell=pre, members=members, args=args))
    return out


def checkpoint_paths(pre):
    ops = {}
    for it in train_items():
        if it['cell'] != pre:
            continue
        for m in it['members']:
            stem = m.replace('.json', '')
            fam = stem.split('_')[0]
            jid = (RUNS / it['job'] / 'JOBID.txt').read_text().strip() if (RUNS / it['job'] / 'JOBID.txt').exists() else None
            path = (f'{NS}/{it["job"]}/output/{stem}/best.pt' if it['job'] in LEGACY_SINGLE
                    else f'{NS}/{it["job"]}/output/{stem}/{stem}/best.pt')
            ops[fam] = dict(job=it['job'], job_id=jid, config=m, co_scheduled=[x.replace('.json', '') for x in it['members']],
                            checkpoint=path)
    return ops


def panel_config(problem, mesh, pre):
    ref = SQ_REF[problem][str(mesh)]
    cfg = dict(problem=problem, intervals=mesh, cg_maxiter=200000, order_seed=20260924 + mesh,
               repetitions=3, cg_repetitions=2 if mesh < 4096 else 1, burn_seconds=0.1, cooldown_seconds=5.0,
               phase_dummy_seconds=2.0, table1=ref)
    if problem == 'square':
        cfg.update(nmrom_R=[512, 128], rows_per_chunk=SQ_ROWS[mesh][0], eval_rows=SQ_ROWS[mesh][1],
                   cg_tols=SQ_CG if mesh < 4096 else [0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03, 0.01],
                   expected_nmrom_worst={'R512_linear': ref['R512_linear']['worst'], 'R128_linear': ref['R128_linear']['worst']})
    else:
        cfg.update(nmrom_R=[128, 64], cg_tols=LS_CG,
                   expected_nmrom_worst={'R128_linear': ref['R128_linear']['worst'], 'R64_linear': ref['R64_linear']['worst']})
    cfg['operators'] = checkpoint_paths(pre)
    if pre == 'l2048':
        cfg['xla_flags'] = '--xla_gpu_autotune_level=0'   # DESIGN A7
        cfg['xla_mem_fraction'] = 0.6                      # DESIGN A9
    return cfg


def queue():
    txt = sh("ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 tufts-login \"squeue -u tawal01 -h -o '%i %j %T'\"")
    return [l.split() for l in txt.strip().splitlines() if l.strip()]


def pull(job):
    d = RUNS / job
    if (d / 'PULLED').exists():
        return
    (d / 'pull').mkdir(exist_ok=True)
    sh(f"rsync -a -e 'ssh -o ConnectTimeout=30 -o ServerAliveInterval=15' --include='*/' --include='*.json' --include='*.out' --include='*.err' --include='*.log' --exclude='*' "
       f"tufts-login:{NS}/{job}/ {d}/pull/", check=False)
    (d / 'PULLED').write_text(time.strftime('%F %T') + '\n')
    log('pulled', job)
    subprocess.run(f'cd {HERE} && python3 summarize.py', shell=True, capture_output=True)   # results.json, REPORT.md, cost_points.json


def step():
    q = queue()
    mine = [r for r in q if r[1].startswith('p2d_')]
    ids = {r[0] for r in q}
    # finished = submitted and not in queue
    finished = set()
    for d in RUNS.iterdir():
        if (d / 'JOBID.txt').exists() and (d / 'JOBID.txt').read_text().strip() not in ids:
            finished.add(d.name)
            pull(d.name)
    free = CAP - len(mine)
    items = train_items()
    h200_active = 0
    for d in RUNS.iterdir():
        if (d / 'JOBID.txt').exists() and d.name not in finished and (d / 'STAGE.json').exists():
            h200_active += json.loads((d / 'STAGE.json').read_text()).get('gpu_constraint') == 'h200-141G'
    todo = []
    # panels first when ready (they unblock results), then training in plan order
    for problem, mesh, pre, _, _ in CELLS:
        pj = PANEL_NAME.get(pre, f'{pre}p')
        tj = [it['job'] for it in train_items() if it['cell'] == pre]
        if not (RUNS / pj / 'JOBID.txt').exists() and all(t in finished for t in tj):
            todo.append(('panel', pj, problem, mesh, pre))
    for it in items:
        if not (RUNS / it['job'] / 'JOBID.txt').exists():
            todo.append(('train', it))
    for t in todo:
        if free <= 0:
            break
        wants_h200 = (t[0] == 'panel' and t[3] == 4096) or (t[0] == 'train' and 'h200-141G' in t[1]['args'])
        if wants_h200 and h200_active >= H200_MAX:
            continue
        if t[0] == 'panel':
            _, pj, problem, mesh, pre = t
            cj = RUNS / f'{pj}-panel.json'
            cj.write_text(json.dumps(panel_config(problem, mesh, pre), indent=1) + '\n')
            gpu = ANY   # DESIGN A2: H200 unavailable; the Table 1 4096^2 job itself ran on an A100 80GB
            if not (RUNS / pj).exists():
                sh(f"cd {HERE} && python3 cluster/stage.py panel {pj} --config-json {cj} --gpu-constraint '{gpu}' --mem 240G --time 05:00:00")
            job = pj
        else:
            it = t[1]
            job = it['job']
            if not (RUNS / job).exists():
                sh(f"cd {HERE} && python3 cluster/stage.py " + ' '.join(f"'{x}'" for x in it['args']))
        r = subprocess.run(f'bash {HERE}/cluster/submit.sh {job}', shell=True, capture_output=True, text=True)
        log('submit', job, r.returncode, r.stdout.strip().splitlines()[-1:] if r.stdout else '', r.stderr.strip()[-300:])
        if r.returncode != 0:
            break
        free -= 1
        h200_active += wants_h200
    remaining = [t[1] if t[0] == 'panel' else t[1]['job'] for t in todo]
    log('queue', [(r[1], r[2]) for r in mine], 'free', free, 'todo', len(remaining))
    return len(todo) == 0 and not mine


if __name__ == '__main__':
    (RUNS / 'scheduler.pid').write_text(str(__import__('os').getpid()))
    once = '--once' in sys.argv
    while True:
        try:
            done = step()
        except Exception as exc:  # keep polling on transient ssh errors
            log('error', repr(exc)[:300])
            done = False
        if once or done:
            break
        time.sleep(120)
