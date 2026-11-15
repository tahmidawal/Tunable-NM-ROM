"""Materialize/verify the authoritative lane roster against each historical Git object."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from setup_selection import BASE, REF, P, E, data, hash_
from open_sealed import packed

HERE=Path(__file__).resolve().parent
FORK='93f399faad67f94ea2b56fe25f13003143e64694'
README='experiments/jcp-consolidated/README.md'
ROSTER_PATH=HERE/'roster.json'
METHODS={
    'quadrature-study':'2D continuous bank and trained checkpoint through pinned plateau vendor',
    'quadrature-burgers3d':'3D bank, frozen basis and upstream rule family through plateau vendor',
    'jcp-references':'E0 reference, floor, public metadata and six original encrypted commitments',
    'jcp-fom-pareto':'reference-discretization FOM controls via the fixed bank panels',
    'jcp-fom2d':'2D classical full-order comparator ladder via the fixed bank panels',
    'jcp-speed3d':'3D full-order and sampling solver library through pinned plateau vendor',
    'jcp-hr-baselines':'native POD historical solver kernels and training assets',
    'imp-multimesh':'one native 2D continuous POD/ECSW selection',
    'b3d-pod':'one native 3D unreduced-sampling POD selection',
    'b3d-solver':'fine 3D native POD selection source',
    'burgers-quadrature-plateau':'eight fixed bank and full-order panels; four prospective fixed quadrature rules'}
REASON={
 'jcp-time2':'alternative time-integration hypotheses', 'jcp-wide-bank':'larger bank widths',
 'jcp-smooth-bank':'alternative training regularization','jcp-combined':'alternate combined recipe',
 'jcp-qualify':'separate retrospective solver certification study','jcp-regimes':'different development regime',
 'jcp-robust':'alternative training seeds','jcp-mechanism':'training mechanism and ablations',
 'jcp-heads':'alternative trained output heads','jcp-dims':'alternative latent dimensions',
 'jcp-budget':'alternative training and solver budgets','jcp-p1-scaling':'historical time/cost scaling',
 'imp-faststep':'alternative integrator','imp-fronts':'out-of-family steep fronts','imp-geometry':'new geometry',
 'imp-hrcost':'historical cost sweep','imp-varcoef':'different coefficient PDE','imp-box':'different domain',
 'b3d-refine':'different FOM refinement study',
}
def git(*args):
    return subprocess.check_output(['git',*args], text=True).strip()

def proof(commit,path):
    value=data(commit,path)
    return {'commit':commit,'path':path,'blob':git('rev-parse',f'{commit}:{path}'),'sha256':hash_(value)}

def build(fixed=None):
    original=re.findall(r'^\| (exp/[^ ]+) \| ([0-9a-f]+) \|$',data(FORK,README).decode(),re.M)
    assert len(original)==19
    archive={branch:git('rev-parse',short+'^{commit}') for branch,short in original}
    others=git('for-each-ref','--format=%(refname:short)|%(objectname)','refs/heads/exp/2026-10*','refs/heads/exp/2026-11*').splitlines()
    if fixed is not None:
        others=[b+'|'+v['commit'] for b,v in fixed['roster'].items()]
    for pair in others:
        branch,commit=pair.split('|')
        if '2026-11-15' not in branch:
            archive.setdefault(branch,commit)
    assert set(METHODS).issubset({b.rsplit('/',1)[1][11:] for b in archive})
    listing={}
    for branch,commit in sorted(archive.items()):
        slug=branch.rsplit('/',1)[1][11:]
        reason=REASON.get(slug, 'separate solver, application, hardware campaign or manuscript scope')
        listing[branch]={'commit':commit,'disposition':'included' if slug in METHODS else 'excluded',
                         'role':METHODS.get(slug,reason), 'tree':git('rev-parse',commit+'^{tree}')}
    entries=[proof(FORK,README),proof(REF,E+'/DESIGN.md'),proof(BASE,P+'/DESIGN.md'),
             proof(REF,E+'/checks/sealed_seeds.json'),proof(REF,E+'/store/STATUS.json'),proof(REF,E+'/store/MANIFEST.sha256')]
    sol=json.loads((HERE/'selection.json').read_text())
    for dim in sol.values():
        for panel in dim.values():
            path=panel['old_panel']
            e=proof(BASE,path)
            assert e['sha256']==panel['old_result_sha256']
            name=path.split('/')[-2]
            summary_path=f'{P}/checks/{name}-summary.json'
            summary=json.loads(data(BASE,summary_path))['plateau_ST']
            assert summary['demonstrated'] and summary['anchor']['passed']
            for arm in [panel['chosen_lattice'],panel['chosen_gauss']]:
                rungs=next(f['rungs'] for f in summary['families'].values() if any(r['arm']==arm for r in f['rungs']))
                i=next(i for i,r in enumerate(rungs) if r['arm']==arm)
                assert all(r['glob']=='pass' for r in rungs[i:])
            entries.extend([e,proof(BASE,summary_path),proof(BASE,f'{P}/checks/{name}-audit.json')])
    for batch in json.loads((HERE/'original_archive.json').read_text())['original_batches']:
        e=proof(REF,f"{E}/runs/{batch['attempt']}/output/result.json")
        assert e['sha256']==batch['result_sha256']
        entries.append(e)
    c2=archive['exp/2026-10-21-imp-multimesh']
    c3=archive['exp/2026-10-26-b3d-pod']
    p2='experiments/imp-multimesh/checks/analysis.json'
    p3='experiments/b3d-pod/inputs/FINE-SELECTION.json'
    choice=json.loads(data(c2,p2))['verdicts']['worst|A_dep']['table']['256+1024+4096|q10']['R384|BE']['best']
    assert choice['arm']=='CPU-256' and set(choice['choice'].values())=={'r256|k8|CNR'}
    selected=json.loads(data(c3,p3))['selected']
    assert selected['subject_id']=='POD512_none_CN_25steps_g1e-4'
    entries.extend([proof(c2,p2),proof(c3,p3)])
    for _,(c,p,blob,want) in json.loads((HERE/'pod-source.json').read_text()).items():
        assert proof(c,p)=={'commit':c,'path':p,'blob':blob,'sha256':want}
    for p,h in json.loads((HERE/'upstream.json').read_text())['files'].items():
        assert hash_(data(BASE,P+'/'+p))==h
    return {'coordinator_scope':'Original 19 merged lanes plus subsequent experimental branches up to 2026-11-14. Included names describe the selected use only. All unlisted populations, comparators, and methods are excluded. All other prior projects are excluded.',
            'roster':listing,'selected_evidence':entries,'native_pod_selections':[choice,selected]}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--write',action='store_true')
    args=ap.parse_args()
    got=packed(build(None if args.write else json.loads(ROSTER_PATH.read_bytes())))
    if args.write: ROSTER_PATH.write_bytes(got)
    elif ROSTER_PATH.read_bytes()!=got: raise ValueError('authoritative roster/proofs changed')
    print(hashlib.sha256(got).hexdigest())
