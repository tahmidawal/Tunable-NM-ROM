"""Materialize prospective options solely from previous, committed non-secret panel configurations."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
BASE = 'a2eae048a976c4d2124bdccb957c50391157dd47'
REF = 'cc04f4b54a1b717d00eb803a276d7efb86994773'
P = 'experiments/burgers-quadrature-plateau'
E = 'experiments/jcp-references'

def data(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'])

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')

def hash_(b):
    return hashlib.sha256(b).hexdigest()

def main():
    (HERE/'configs').mkdir(exist_ok=True)
    selections = {'2':{}, '3':{}}
    for job in ('p2d256b','p2d512','p2d1024','p2d2048','p2d4096','p3d65b','p3d129','p3d257c'):
        src = f'{P}/results/{job}/result.json'
        raw = data(BASE,src)
        prev = json.loads(raw)
        cfg = prev['config']
        dim, mesh = cfg['dim'], cfg['mesh']
        key = f'd{dim}m{mesh}'
        cfg['cohort'] = {2:'sealed128',3:'sealed64'}[dim]
        cfg['cases'] = list(range({2:128,3:64}[dim]))
        selections[str(dim)][key] = {'old_panel':src, 'old_result_sha256':hash_(raw),
            'chosen_lattice':{2:'fib987',3:'lat4096'}[dim], 'chosen_gauss':{2:'gauss45',3:'gl20'}[dim],
            'independent_anchor':cfg['anchor_check_arm'], 'reference_rule':cfg['conv_arm']}
        # Preserve the full ladder and full-solver families, even the classical methods that beat us.
        save(HERE/'configs'/f'{key}.json',cfg)
        if key in ('d2m256','d3m65'):
            cfg = dict(cfg, cohort={2:'dev6',3:'dev4'}[dim], cases=[0,1])
            cfg['timing'] = dict(cfg['timing'],cases=2,reps=2,confirm_reps=2)
            cfg['audit_cases']=[0]
            save(HERE/'configs'/f'pilot-{key}.json',cfg)
    save(HERE/'selection.json', selections)
    runs = []
    for d,co,n in [(2,'sealed128',4),(3,'sealed64',2)]:
        for part in range(n):
            attempt=f'fd{d}d{co}p{part}'
            path=f'{E}/runs/{attempt}/output/result.json'
            raw=data(REF,path)
            result=json.loads(raw)
            runs.append(dict(dim=d, cohort=co, attempt=attempt, result_sha256=hash_(raw),
                manifest_cipher_sha256=result['sealed_manifest']['cipher_sha256'],
                manifest_plain_sha256=result['sealed_manifest']['plain_sha256'],
                cases=result['config']['cases'], ciphertexts=result['sealed_manifest']['ciphertexts']))
    save(HERE/'original_archive.json', {
        'store_version': json.loads(data(REF,E+'/store/STATUS.json'))['store_version'],
        'reference_commit':REF, 'original_batches':runs, 'seed_provenance':json.loads(data(REF,E+'/checks/sealed_seeds.json')),
    })
    filenames=[line.decode() for line in subprocess.check_output(['git','ls-tree','--full-tree','-r','--name-only',BASE,'--',P+'/vendor']).splitlines()]
    filenames += [P+'/'+x for x in ('plateau.py','plateau2d.py','plateau3d.py','qlat.py','qlat3d.py','rules2d.npz','rules3d.npz','rules2d_manifest.json','rules3d_manifest.json')]
    save(HERE/'upstream.json',{'commit':BASE,'files':{s[len(P)+1:]:hash_(data(BASE,s)) for s in filenames}})
if __name__=='__main__':
    main()
