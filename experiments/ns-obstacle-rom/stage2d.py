"""Prepare hash inventories from existing FD1 inputs; execute only a separate release.

Candidate building never creates admission, data, a release or a training result.
Paths point at warm inputs; audit-only cold archives are not duplicated here.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,shutil,sys,time
from qualification import require_qualification
from fe_adapter import sha_file

ROOT=Path(__file__).parent
INPUTS={'prepare':{'training_manifest','preparation_release'},
        'train':{'prepared','preparation_release'},
        'development':{'prepared','preparation_release','bank','training_evidence','probes','development'},
        'validation_accuracy':{'prepared','preparation_release','bank','training_evidence','probes','frozen_models','rom_review','population'},
        'paired':{'prepared','preparation_release','bank','training_evidence','probes','frozen_models','rom_review',
                  'fom_selection','fom_review','fom_eligibility','truth_admission','fom_release','population','uncertainty','uncertainty_review'}}


def source_digest():
    from prepare2d import preparation_digest
    from pilot2d import source_digest as pilot_digest
    return hashlib.sha256(json.dumps({'entry':sha_file(__file__),'preparation':preparation_digest(),
        'pilot':pilot_digest(),'frozen_rebuild':sha_file(ROOT/'frozen_rom2d.py'),
        'paired_runner':sha_file(ROOT/'paired_model2d.py'),'validation_runner':sha_file(ROOT/'validation2d.py')},sort_keys=True).encode()).hexdigest()


def inventory(path):
    path=Path(path).resolve()
    if path.is_file():return {'path':str(path),'kind':'file','sha256':sha_file(path),'bytes':path.stat().st_size}
    if not path.is_dir():raise FileNotFoundError(path)
    files={str(p.relative_to(path)):{'sha256':sha_file(p),'bytes':p.stat().st_size}
           for p in sorted(path.rglob('*')) if p.is_file()}
    if not files:raise ValueError('nonempty warm input directory required')
    return {'path':str(path),'kind':'directory','files':files}


def qualification(stage,inputs):
    if stage=='prepare':
        from fe_adapter import read_training_manifest
        return require_qualification(read_training_manifest(inputs['training_manifest']))
    from pilot2d import read_preparation_metadata
    metadata,_=read_preparation_metadata(inputs['prepared'],inputs['preparation_release'])
    return require_qualification(metadata['reference_qualification'])


def training_provenance(inputs):
    from bank import FrozenBank
    bank=FrozenBank.load(inputs['bank']);m=bank.metadata;folder=Path(inputs['training_evidence'])
    config=json.loads((folder/'CONFIG.json').read_text());final=json.loads((folder/'FINAL.json').read_text())
    checkpoint=m.get('last_checkpoint')
    if (m.get('trained') is not True or m.get('seed')!=27102027 or m.get('steps_cap')!=3000 or
        m.get('time_cap_seconds')!=1800 or not isinstance(checkpoint,dict) or final.get('status')!='completed' or
        final.get('last_checkpoint')!=checkpoint or final.get('cumulative_completed_steps')!=m['steps_executed'] or
        not 0<=m['steps_executed']<=3000 or config.get('step_cap')!=3000 or config.get('time_cap_seconds')!=1800 or
        config.get('resume_authorized') is not False or checkpoint.get('resume_authorized') is not False):
        raise ValueError('original fixed training/checkpoint budget required; no resume/reset')
    if sha_file(folder/'CONFIG.json')!=checkpoint['configuration_sha256']:
        raise ValueError('original checkpoint configuration changed')
    archive=folder/checkpoint['arrays']
    if Path(checkpoint['arrays']).name!=checkpoint['arrays'] or sha_file(archive)!=checkpoint['arrays_sha256']:
        raise ValueError('original checkpoint arrays changed')
    if config['training_array_sha256']!=m['training_array_sha256']:
        raise ValueError('bank/checkpoint training population differs')
    return {'bank_sha256':sha_file(inputs['bank']),'configuration_sha256':sha_file(folder/'CONFIG.json'),
        'final_sha256':sha_file(folder/'FINAL.json'),'checkpoint':checkpoint,
        'steps_executed':m['steps_executed'],'original_steps_cap':3000,'original_time_cap_seconds':1800,
        'resume_authorized':False}


def build_candidate(stage,inputs,output):
    if stage not in INPUTS or set(inputs)!=INPUTS[stage]:raise ValueError('exact stage-specific warm inputs required')
    q=qualification(stage,inputs)
    if stage=='validation_accuracy':
        from validation2d import verify_review
        verify_review(inputs)
    if stage=='paired':
        from paired_model2d import verify_decisions
        verify_decisions(inputs)  # Before even hashing validation-population metadata.
    candidate={'schema':'ns-obstacle-rom-stage-candidate-v1','status':'candidate_not_released',
        'stage':stage,'source_sha256':source_digest(),'reference_qualification':q,
        'inputs':{k:inventory(v) for k,v in inputs.items()},'training_executed':False,
        'fixed_training_budget':{'steps':3000,'seconds':1800,'seed':27102027},
        'validation_inputs_allowed':stage in ('paired','validation_accuracy'),'automatic_retry':False}
    if stage in ('development','paired','validation_accuracy'):candidate['training_provenance']=training_provenance(inputs)
    with Path(output).open('x') as stream:json.dump(candidate,stream,indent=2);stream.write('\n')
    return candidate


def verify(candidate_path,release_path):
    c=json.loads(Path(candidate_path).read_text());r=json.loads(Path(release_path).read_text())
    if (c.get('schema')!='ns-obstacle-rom-stage-candidate-v1' or c.get('status')!='candidate_not_released' or
        c.get('stage') not in INPUTS or c.get('source_sha256')!=source_digest() or
        r.get('schema')!='ns-obstacle-rom-stage-release-v1' or r.get('authorized_stage')!=c['stage'] or
        r.get('candidate_sha256')!=sha_file(candidate_path) or r.get('source_sha256')!=source_digest() or
        r.get('single_execution') is not True):raise RuntimeError('exact separately reviewed stage release required')
    if require_qualification(r.get('reference_qualification'))!=require_qualification(c['reference_qualification']):
        raise ValueError('stage release scope differs')
    if (c.get('fixed_training_budget')!={'steps':3000,'seconds':1800,'seed':27102027} or
        c.get('validation_inputs_allowed') is not (c['stage'] in ('paired','validation_accuracy')) or c.get('automatic_retry') is not False):
        raise ValueError('fixed stage budget/data-access policy changed')
    if set(c['inputs'])!=INPUTS[c['stage']]:raise ValueError('undeclared stage input')
    inputs={k:v['path'] for k,v in c['inputs'].items()}
    ordered=sorted(c['inputs'],key=lambda key:key in ('population','uncertainty','uncertainty_review'))
    for name in ordered:
        if c['stage']=='validation_accuracy' and name=='population':
            from validation2d import verify_review
            verify_review(inputs)
        if c['stage']=='paired' and name in ('population','uncertainty','uncertainty_review'):
            from paired_model2d import verify_decisions
            verify_decisions(inputs)  # Exact decision inputs checked before validation metadata.
        item=c['inputs'][name]
        if inventory(item['path'])!=item:raise ValueError('warm input changed: '+name)
    if qualification(c['stage'],inputs)!=require_qualification(c['reference_qualification']):raise ValueError('actual input scope differs')
    if c['stage'] in ('development','paired','validation_accuracy') and training_provenance(inputs)!=c.get('training_provenance'):
        raise ValueError('original training/checkpoint binding changed')
    if c['stage']=='paired':
        from paired_model2d import verify_decisions
        verify_decisions(inputs)
    return c,r,inputs


def run(candidate_path,release_path,output):
    started=time.perf_counter();c,r,inputs=verify(candidate_path,release_path)
    if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or
        not os.environ.get('SLURM_JOB_ID') or os.environ.get('JAX_ENABLE_X64')!='true' or
        os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest'):raise RuntimeError('released native GPU allocation required')
    out=Path(output)
    if shutil.disk_usage(out.parent).free<10*2**30:raise OSError('ten-GiB disk reserve required')
    if Path(r['job_directory']).resolve()!=Path.cwd().resolve():raise ValueError('released job directory differs')
    with (Path.cwd()/'STAGE-CLAIM.json').open('x') as claim:
        json.dump({'release_sha256':sha_file(release_path),'candidate_sha256':sha_file(candidate_path),
                   'job_id':os.environ['SLURM_JOB_ID']},claim)
    out.mkdir(exist_ok=False)
    record={'stage':c['stage'],'candidate_sha256':sha_file(candidate_path),'release_sha256':sha_file(release_path),
            'reference_qualification':c['reference_qualification'],'physical_admission':False,'status':'started'}
    try:
        import jax
        if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':
            raise RuntimeError('actual GPU/f64/highest preflight failed')
        if c['stage']=='prepare':
            from prepare2d import prepare
            inner=json.loads(Path(inputs['preparation_release']).read_text())
            prepare(inputs['training_manifest'],inputs['preparation_release'],out/'preparation',
                    inner.get('point_block_volume',False),inner.get('verify_primary_volume',False))
        elif c['stage']=='train':
            from pilot2d import train_from_preparation
            train_from_preparation(inputs['prepared'],inputs['preparation_release'],out/'bank.npz')
        elif c['stage']=='validation_accuracy':
            from validation2d import run as run_validation
            run_validation(inputs,r,out/'validation')
        elif c['stage']=='paired':
            from paired_model2d import run as run_pair
            run_pair(inputs,r,out/'paired')
        else:
            from pilot2d import load_preparation,_read_arrays,FrozenBank,array_digest,load_reference_population,run_accuracy_pilot
            arrays,metadata,inner=load_preparation(inputs['prepared'],inputs['preparation_release'])
            for key in ('bank','probes','development'):
                expected={'bank':'bank_sha256','probes':'probes_sha256','development':'development_manifest_sha256'}[key]
                if sha_file(inputs[key])!=inner[expected]:raise ValueError('inner preparation release differs from '+key)
            bank=FrozenBank.load(inputs['bank'])
            if bank.metadata.get('trained') is not True or bank.metadata.get('training_array_sha256')!=metadata['training_array_sha256']:
                raise ValueError('bank must come from these actual training arrays')
            probes=_read_arrays(inputs['probes'],2**30)
            rule=array_digest({k:arrays['volume'][k] for k in ('points_dimensionless','weights_dimensionless')})
            development_meta=json.loads(Path(inputs['development']).read_text())
            if development_meta.get('reference_admission_sha256')!=metadata['admission_evidence']['reference_admission_sha256']:
                raise ValueError('development and prepared truth identities differ')
            development_identity={'manifest_sha256':inner['development_manifest_sha256'],'reference_admission_sha256':development_meta['reference_admission_sha256'],'volume_rule_sha256':rule}
            development=load_reference_population(inputs['development'],inner['development_manifest_sha256'],'development',rule)
            def no_validation():raise AssertionError('development stage must not open validation')
            run_accuracy_pilot(arrays,metadata,probes,bank,development,no_validation,out/'development',development_only=True,development_identity=development_identity)
        record['status']='execution_completed_pending_independent_review'
    except Exception as exc:record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-started,peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        try:(out/'STAGE-RESULT.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        except Exception:
            if record['status']!='failed':raise
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    build=sub.add_parser('candidate');build.add_argument('--stage',choices=INPUTS,required=True)
    build.add_argument('--input',action='append',required=True);build.add_argument('--out',type=Path,required=True)
    execute=sub.add_parser('run');execute.add_argument('--candidate',type=Path,required=True)
    execute.add_argument('--release',type=Path,required=True);execute.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='candidate':
        pairs=[item.split('=',1) for item in args.input]
        if len({p[0] for p in pairs})!=len(pairs):raise ValueError('duplicate input name')
        build_candidate(args.stage,dict(pairs),args.out)
    else:run(args.candidate,args.release,args.out)

if __name__=='__main__':main()
