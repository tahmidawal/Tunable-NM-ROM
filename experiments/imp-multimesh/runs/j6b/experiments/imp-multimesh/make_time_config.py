"""Write configs/time.json (J6) from the pulled target-job JSONs: per mesh, ours' six settings with the J5 archive
directory, and every competitor arm with its MEASURED-Q1 configurations (only those can be witnesses, A3.1).
Also the fresh-process compile probes of A5.1 (mmcompile.py) as separate commands.

    python make_time_config.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARC = '/cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh'
TJOB = {256: 'j1', 1024: 'j3', 4096: 'j4'}
OURS = [dict(name='R128', Rp=128, rule='fib4181', schemes=['BE', 'CNR']),
        dict(name='R256', Rp=256, rule='gauss128', schemes=['BE', 'CNR']),
        dict(name='R384', Rp=384, rule='gauss128', schemes=['BE', 'CNR'])]


def main():
    meshes, cmds = {}, []
    only = [int(x) for x in sys.argv[1:]] or list(TJOB)
    for L, job in TJOB.items():
        T = json.loads((HERE / 'runs' / job / 'pulled' / 'output' / f'target{L}.json').read_text())
        arms = []
        for name, A in T['arms'].items():
            q = [cid for cid, c in A['configs'].items() if c.get('status') == 'MEASURED-Q1']
            if not q:
                continue
            arm = next(a for a in T['config']['arms'] if a['name'] == name)
            arms.append(dict(name=name, kind=arm['kind'], source=arm['source'], source_dir=arm['source_dir'],
                             build_dir=f'{ARC}/{job}/output', configs=sorted(q)))
        if L in only:
          meshes[str(L)] = dict(ours=dict(edges=[0, 128, 256, 384, 512], configs=OURS, j5_dir=f'{ARC}/j5/output/ours{L}'),
                              arms=arms)
        cmds.append(dict(name=f'compile_ours_R128_{L}', cmd=f'mkdir -p $OUT/compile && $PY mmcompile.py ours {L} 128 fib4181 BE $OUT/compile/ours_{L}_R128.json', timeout_s=300, continue_on_fail=True, always_run=True))
        cmds.append(dict(name=f'compile_ours_R384_{L}', cmd=f'$PY mmcompile.py ours {L} 384 gauss128 CNR $OUT/compile/ours_{L}_R384.json', timeout_s=300, continue_on_fail=True, always_run=True))
        if arms:
            a = arms[0]
            for cid in (a['configs'][0], a['configs'][-1]):
                r, k = int(cid.split('|')[0][1:]), int(cid.split('|')[1][1:])
                cmds.append(dict(name=f'compile_{a["name"]}_r{r}k{k}_{L}',
                                 cmd=f'$PY mmcompile.py comp {L} {a["name"]} {a["source"]} {a["source_dir"]} {a["build_dir"]} {r} {k} $OUT/compile/comp_{L}_{a["name"]}_r{r}k{k}.json',
                                 timeout_s=300, continue_on_fail=True, always_run=True))
    cfg = dict(commands=[dict(name='time', cmd='$PY mmtime.py configs/time.json $OUT',
                              timeout_s=9000 if len(only) == 3 else 5400)] + cmds,
               meshes=meshes, retry_after_seconds=0, import_check=['mmtime', 'mmcompile', 'import p1common'])
    name = 'time' if len(only) == 3 else 'time_' + '_'.join(map(str, only))
    cfg['commands'][0]['cmd'] = f'$PY mmtime.py configs/{name}.json $OUT'
    (HERE / 'configs' / f'{name}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print({L: [(a['name'], len(a['configs'])) for a in m['arms']] for L, m in meshes.items()})


if __name__ == '__main__':
    main()
