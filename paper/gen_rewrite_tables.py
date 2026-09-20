"""Render paired CG tables from a committed, hash-pinned report snapshot.
The snapshot was generated from run JSONs; source paths/hashes remain in it.
No experiment selection or live run access occurs during normal builds.
"""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
E=HERE/'evidence/paired-cg-2026-09-20'
PIN='86bed934'
REPO=Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude')
SOURCE='reports/2026-09-20-iterative-cg-comparisons.json'

def refresh():
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse',PIN]).decode().strip()
    raw=subprocess.check_output(['git','-C',str(REPO),'show',f'{commit}:{SOURCE}'])
    E.mkdir(parents=True,exist_ok=True)
    (E/'results.json').write_bytes(raw)
    (E/'manifest.json').write_text(json.dumps(dict(commit=commit,path=SOURCE,sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')

def esc(s):
    return str(s).replace('_',r'\_').replace('%',r'\%')

def label(r):
    m=r['method']
    if m.startswith(('linear','d_linear')): return 'Linear-bank ROM'
    if 'q256' in m: return r'NM-ROM, $q=256$'
    if 'q32' in m: return r'NM-ROM, $q=32$'
    if 'cholesky' in m: return 'NM-ROM'
    return r'NM-ROM, $q=0$'

def table(name,headers,rows,align):
    lines=[r'% Generated from the paired CG run-report snapshot; do not edit.',r'\small',r'\begin{tabular}{@{}'+align+'@{}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines+=[' & '.join(row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    (HERE/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n')
    (HERE/'tables-md'/f'{name}.md').write_text('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows))

if __name__=='__main__':
    import sys
    if '--refresh' in sys.argv: refresh()
    raw=(E/'results.json').read_bytes(); meta=json.loads((E/'manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    d=json.loads(raw)
    for r in d['rows']:
        assert abs(r['cg_ms']/r['method_ms']-r['speedup'])<1e-10
        assert r['cg_error_pct']<=r['error_pct']
        assert r['job_id'] and r['source'] in d['source_sha256']
    rows=[]; full=[]
    for r in d['rows']:
        problem=r['problem'].replace('Reflective wave2D','Wave').replace('2D','')
        vals=[f"{r[k]:.4f}" for k in ['error_pct','method_ms','cg_error_pct','cg_ms']]+[f"{r['speedup']:.2f}"+r'$\times$']
        if r['intervals']==1024: rows.append([problem,label(r)]+vals)
        full.append([problem,str(r['intervals']),label(r)]+vals+[esc(r['cg']),r'\texttt{'+r['job_id']+'}'])
    headers=['Problem','Method','Error (\\%)','GPU ms','CG error (\\%)','CG ms','$S$']
    table('TR_cg_main',headers,rows,'llrrrrr')
    table('TR_cg_all',['Problem','$N$','Method']+headers[2:]+['CG setting','Job'],full,'lllr rrrrll'.replace(' ',''))
    (HERE/'tables/rewrite-provenance.json').write_text(json.dumps(dict(snapshot=meta,source_sha256=d['source_sha256'],rows=d['rows']),indent=2)+'\n')
    print(f'Paired CG tables: {len(rows)} main rows; {len(full)} appendix rows; hashes and ratios verified.')
