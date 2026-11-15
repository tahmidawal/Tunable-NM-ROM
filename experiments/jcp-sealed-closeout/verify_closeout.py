"""Verify and index the fixed ten completed jobs against F1, the original publication and local mirrors."""
import argparse
import hashlib
import json
from pathlib import Path
from frozen_decisions import aggregate,MESHES,RULES
from open_sealed import HERE,STORE,sha,packed

def need(p,msg):
    if not p:raise ValueError(msg)

def checked_inventory(root,name):
    entries={}
    for line in (root/name).read_text().splitlines():
        h,raw=line.split(None,1)
        f=Path(raw.strip())
        need(not f.is_absolute() and '..' not in f.parts and f not in entries,'Unsafe or duplicate archive entry')
        with (root/f).open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        need(actual==h,'Raw archive mismatch: '+str(f))
        entries[f]=h
    need(bool(entries),'Empty inventory')
    return entries

def load(path):return json.loads(path.read_bytes())
def validate(raw):
    jobs=load(HERE/'results/jobs.json')
    frozen=load(HERE/'F1.json')
    for name,h in frozen['files'].items():need(sha((HERE/name).read_bytes())==h,'Changed F1 source '+name)
    rev=load(HERE/'F1-REVIEW.json')
    pub=load(HERE/'results/opening.json')
    need(pub['file_count']==768 and sha((HERE/'F1.json').read_bytes())==rev['freeze_sha256']==pub['freeze_sha256'] and
            sha((HERE/'F1-REVIEW.json').read_bytes())==pub['review_sha256'] and pub['store_version']==STORE,'Bad opening record')
    expected={f'sealed{d}d{m}' for d in MESHES for m in MESHES[d]}|{'pod2','pod3'}
    need(set(jobs)==expected,'Incomplete job ledger')
    panels=[];panel_info={};pod={}
    for task,j in jobs.items():
        root=raw/task
        source=checked_inventory(root,'SOURCE.sha256'); outputs=checked_inventory(root,'OUTPUTS.sha256')
        archive=checked_inventory(root,'ARCHIVE.sha256')
        # The archive inventory excludes its own basename at every depth. Historical pilot
        # inventories inside the frozen source are independently checked by SOURCE.sha256.
        for path,h in source.items():
            if path.name=='ARCHIVE.sha256' and path.parts[0]=='experiments':continue
            need(archive[path]==h,'source missing from archive')
        for path,h in outputs.items():need(archive[path]==h,'output missing from archive')
        for name in ('F1.json','F1-REVIEW.json'):
            need(source[Path('experiments/jcp-sealed-closeout')/name]==sha((HERE/name).read_bytes()),'Staged F1 diverged')
        need(root.joinpath(j+'.err').stat().st_size==0,'Nonempty Slurm stderr; inspect required')
        stdout=(root/(j+'.out')).read_text()
        need('jax_backend=gpu' in stdout and 'out of memory' not in stdout.lower() and 'captured constant' not in stdout.lower(),'GPU or runtime log')
        needed=(HERE/'receipts'/task/'ARCHIVE.sha256').read_bytes()
        need(sha(needed)==sha((root/'ARCHIVE.sha256').read_bytes()),'Local cold receipt changed')
        acct=(root/'accounting.txt').read_text().strip().split('|')
        need(acct[:3]==[j,'COMPLETED','0:0'],'Invalid Slurm accounting')
        base=dict(job_id=j,completed=acct[1],elapsed=acct[3],node=acct[4],archive_inventory_sha256=sha(needed))
        if task.startswith('pod'):
            p=load(root/'output/COMPLETE.json');d=int(task[3]);r=load(HERE/f'results/sealed{d}d{256 if d==2 else 65}-pod-and-comparators.json')['pod']
            need(p['job_id']==j and outputs[Path('output/result.jsonl')]==r['result_sha256'],'POD provenance mismatch')
            for k in ('ST','S'):need(r[k]['all']['total']==p['n'],'Incomplete POD cohort')
            replay=load(HERE/f'receipts/independent-native/{task}.json')
            need(replay['agreement'] and replay['freeze_sha256']==pub['freeze_sha256']
                and replay['review_sha256']==pub['review_sha256'] and replay['publication_sha256']==pub['published_sha256']
                and replay['store_sha256']==STORE and str(replay['production_job'])==j
                and replay['archive_manifest_sha256']==base['archive_inventory_sha256']
                and replay['result_sha256']==r['result_sha256'],'POD replay not bound to original outputs')
            pod[task]=dict(**base,offline_replay_job=replay['audit_job'],max_abs_score_difference=replay['max_abs_difference'],
                **{k:r[k] for k in ('ST','S','steps','nonstationary_steps','finite_fields','valid_cases','result_sha256')})
            continue
        panel=load(HERE/f'results/{task}-decision.json')
        cmp=load(HERE/f'results/{task}-comparators.json')
        a=load(root/'output/witness-audit.json')
        rawsha=outputs[Path('output/result.json')]
        need(panel['result_sha256']==cmp['result_sha256']==a['result_sha256']==rawsha and str(a['job_id'])==panel['job_id']==j,'Result and audits must bind one allocation')
        need(a['passed'] and a['checks']['G_audit']['passed'] and a['checks']['fault_injections']['passed'],'Failed rescore')
        d,m=panel['dim'],panel['mesh'];n={2:128,3:64}[d]
        need(task==f'sealed{d}d{m}','Mesh and job mismatch')
        ident=load(root/'output/cohort-identity.json')
        need(set(ident['original_cohort_sha256_by_batch'].values())=={ident['used_sha256']},'Canonical cohort differs from original jobs')
        record={}
        for k in ('ST','S'):
            result=panel['kinds'][k]
            need(result['n_all']==n,'Incomplete classifier verification')
            record[k]={'n_all':n,'eligible':len(result['eligible_cases']),'status_counts':result['status_counts'],
                'rules':{rule:{'decision':result['rules'][rule]['decision'],'median':result['arm_stats'][rule]['T'],
                               'maximum':result['arm_stats'][rule]['W']} for rule in RULES[d]}}
        panel_info[task]=dict(**base,result_sha256=rawsha,native_cohort_sha256=ident['used_sha256'],
                witness_count=a['checks']['G_audit']['compared'],max_witness_discrepancy=a['checks']['G_audit']['max_rel'],
                timing_gate=cmp['timing']['gates'],**record)
        panels.append(panel)
    verdicts=aggregate(panels)
    (HERE/'results/four-rule-decisions.json').write_bytes(packed(verdicts))
    return dict(opening=pub,panels=panel_info,pod=pod,decisions=verdicts)
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('raw_archive_root',type=Path);a=cli.parse_args()
    result=validate(a.raw_archive_root)
    (HERE/'results/verification.json').write_bytes(packed(result))
