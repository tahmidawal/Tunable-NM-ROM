"""Consume an independently frozen development-only FOM choice; never select one.

The runtime release independently pins selection, review, candidate eligibility,
FD1 truth admission, reference population, sources, CPU policy and output contract.
No native import or validation field read occurs before those bindings pass.
Selection predates frozen eligibility: it hashes candidate inputs, not the later
eligibility file that references selection. The independent runtime release
binds both, avoiding a circular hash dependency.
"""
from pathlib import Path
import hashlib,importlib,json,os,platform,resource,sys,time
import numpy as np
from fe_adapter import sha_file,load_npz_bounded
from qualification import require_qualification
from pilot2d import DEV_RE,VALIDATION_RE,OUTPUTS,array_digest,source_digest as pilot_digest

ROOT=Path(__file__).parent
NATIVE=ROOT/'fom_native'
NAMES=('shared_lift','candidate_lift','scalars','reference_export','reference_scope',
       'oriented_normals','g2.exact_lookup','fom_outputs','fom_query')


def read_json(path):
    path=Path(path)
    if path.stat().st_size>64*2**20:raise MemoryError('bounded metadata required')
    return json.loads(path.read_text())


def native_digest():
    manifest=read_json(NATIVE/'SOURCE.json')
    expected={name.replace('.','/')+'.py' for name in NAMES}
    if set(manifest['files'])!=expected:raise ValueError('complete frozen native source closure required')
    for name,digest in manifest['files'].items():
        if sha_file(NATIVE/name)!=digest:raise ValueError('frozen native source differs: '+name)
    return sha_file(NATIVE/'SOURCE.json')


def source_digest():
    return hashlib.sha256(json.dumps({'consumer':sha_file(__file__),'native':native_digest(),
        'pilot':pilot_digest(),'matching':sha_file(ROOT/'fom_match2d.py'),
        'uncertainty':sha_file(ROOT/'uncertainty2d.py')},sort_keys=True).encode()).hexdigest()


def checked_file(parent,descriptor):
    if not isinstance(descriptor.get('path'),str) or not descriptor['path']:raise ValueError('declared input path required')
    path=(parent/descriptor['path']).resolve()
    if sha_file(path)!=descriptor.get('sha256'):raise ValueError('selected input hash differs: '+str(path))
    return path


def verify_documents(selection_path,review_path,eligibility_path,truth_path,release_path):
    paths={k:Path(p) for k,p in [('selection',selection_path),('selection_review',review_path),
        ('eligibility',eligibility_path),('reference_admission',truth_path)]}
    release=read_json(release_path)
    if (release.get('schema')!='ns-obstacle-frozen-fom-runtime-release-v1' or
        release.get('authorized_stage')!='paired_comparison' or release.get('consumer_source_sha256')!=source_digest()):
        raise RuntimeError('exact independently reviewed FOM consumer release required')
    for key,path in paths.items():
        if sha_file(path)!=release.get(key+'_sha256'):raise ValueError('runtime release differs from '+key)
    selection=read_json(paths['selection']);review=read_json(paths['selection_review'])
    qualification=require_qualification(read_json(paths['reference_admission']))
    if any(require_qualification(r.get('reference_qualification'))!=qualification for r in (release,selection,review)):
        raise ValueError('FOM selection/runtime truth scope differs')
    if (selection.get('schema')!='ns-obstacle-fom-development-selection-v1' or
        selection.get('selection_frozen') is not True or selection.get('development_only') is not True or
        selection.get('validation_used') is not False or selection.get('matched_accuracy') is not True or
        selection.get('development_Re')!=list(DEV_RE)):
        raise ValueError('frozen development-only matched-accuracy selection required')
    if (review.get('schema')!='ns-obstacle-fom-selection-review-v1' or review.get('passed') is not True or
        review.get('development_only') is not True or review.get('matched_accuracy') is not True or
        review.get('selection_sha256')!=release['selection_sha256']):
        raise ValueError('independent exact selection review required')
    if any(r.get('reference_admission_sha256')!=release['reference_admission_sha256'] for r in (selection,review)):
        raise ValueError('selection/review truth admission differs')
    from fom_match2d import source_digest as matching_digest
    if selection.get('matching_source_sha256')!=matching_digest():raise ValueError('exact bank-specific matching selector required')
    target=selection.get('bank_target')
    if (not isinstance(target,dict) or target.get('model') not in ('bank_r2','bank_r4','bank_r8') or
        target.get('method') not in ('exact_tensor_weak_LS','fitted_convection_weak_LS') or
        target.get('prefix') not in (2,4,8) or target['model']!='bank_r'+str(target['prefix']) or
        release.get('bank_target')!=target):raise ValueError('runtime must bind exact frozen bank/prefix/target')
    for key in ('fingerprint','frozen_models_sha256','development_summary_sha256','rom_review_sha256'):
        if not isinstance(target.get(key),str) or len(target[key])!=64:raise ValueError('actual frozen target evidence required')
    for key in ('uncertainty_sha256','uncertainty_review_sha256'):
        if not isinstance(selection.get(key),str) or len(selection[key])!=64:raise ValueError('matched development uncertainty evidence required')
    if selection.get('native_source_sha256')!=native_digest():raise ValueError('selected FOM source differs')
    if selection.get('field_rule_mode') not in ('native_exact','certified_requested_points'):
        raise ValueError('frozen development field evaluation mode required')
    contract=release.get('output_contract');population=release.get('population')
    expected={'development':list(DEV_RE),'validation':list(VALIDATION_RE)}
    if (contract not in OUTPUTS or selection.get('output_contract')!=contract or population not in expected or
        release.get('reynolds')!=expected[population]):raise ValueError('explicit output contract/population release required')
    if type(release.get('cores')) is not int or not 1<=release['cores']<=16:raise ValueError('released CPU policy required')
    cap=release.get('max_common_points')
    if type(cap) is not int or not 0<cap<=1300000:raise ValueError('explicit bounded full output rule required')
    for value in (release.get('reference_population_sha256'),selection.get('volume_rule_sha256')):
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('released population and original common rule digests required')
    versions=selection.get('runtime_versions')
    if (not isinstance(versions,dict) or set(versions)!={'python','ngsolve','numpy','scipy','jax'} or
        any(not isinstance(v,str) or not v for v in versions.values()) or
        type(selection.get('common_rule_points')) is not int or not 0<selection['common_rule_points']<=cap):
        raise ValueError('frozen runtime versions and full point count required')
    base=paths['selection'].parent
    files={key:checked_file(base,selection[key]) for key in
        ('config','mesh','layout','boundary','common_geometry','development_evidence')}
    cfg=read_json(files['config']);layout=read_json(files['layout'])
    fixed={'kind':'dfg','outlet':'gradient','xmin':0.,'xmax':2.2,'H':.41,'center':[.2,.2],'radius':.05,'U':.2,'D':.1}
    if (any(cfg.get(k)!=v for k,v in fixed.items()) or cfg.get('threads')!=release['cores'] or
        cfg.get('precision')!='float64'):
        raise ValueError('selected candidate geometry/scales/thread policy differs')
    eligibility=read_json(paths['eligibility'])
    if (eligibility.get('role')!='frozen_fom_comparator' or eligibility.get('selection_frozen') is not True or
        eligibility.get('selection_sha256')!=release['selection_sha256'] or
        eligibility.get('selection_review_sha256')!=release['selection_review_sha256'] or
        eligibility.get('truth_reference_admission_sha256')!=release['reference_admission_sha256']):
        raise ValueError('separate frozen candidate eligibility bound after selection required')
    return release,selection,qualification,files,cfg,layout


def load_native_modules():
    """Load the exact unchanged closure; reject conflicting cached module versions."""
    native_digest();manifest=read_json(NATIVE/'SOURCE.json')
    for name in NAMES:
        module=sys.modules.get(name)
        if module is not None and sha_file(module.__file__)!=manifest['files'][name.replace('.','/')+'.py']:
            raise RuntimeError('conflicting already imported native module: '+name)
    sys.path.insert(0,str(NATIVE))
    try:modules={name:importlib.import_module(name) for name in NAMES}
    finally:sys.path.pop(0)
    for name,module in modules.items():
        if sha_file(module.__file__)!=manifest['files'][name.replace('.','/')+'.py']:
            raise RuntimeError('actual loaded native module differs: '+name)
    return modules


class FrozenFOMComparator:
    @classmethod
    def open(cls,selection_path,review_path,eligibility_path,truth_path,release_path,output_directory):
        release,selection,qualification,files,cfg,layout=verify_documents(
            selection_path,review_path,eligibility_path,truth_path,release_path)
        return cls._construct(release,selection,qualification,files,cfg,layout,eligibility_path,release_path,output_directory)

    @classmethod
    def _construct(cls,release,selection,qualification,files,cfg,layout,eligibility_path,release_path,output_directory):
        # Each public caller has distinct fail-closed metadata/role gates.
        started=time.perf_counter()
        out=Path(output_directory);out.mkdir(parents=True,exist_ok=False)
        report={'status':'constructing','physical_admission':False,'training_executed':False,
            'consumer_source_sha256':source_digest(),'release_sha256':sha_file(release_path),
            'selection_sha256':release.get('selection_sha256'),'reference_qualification':qualification}
        stage='runtime_policy'
        try:
            if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or
                not os.environ.get('SLURM_JOB_ID') or os.environ.get('JAX_ENABLE_X64')!='true' or
                os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest' or
                any(os.environ.get(k)!=str(release['cores']) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
                raise RuntimeError('released native allocation/f64/thread policy required')
            import jax
            if (jax.default_backend()!='gpu' or not jax.config.x64_enabled or
                str(jax.config.jax_default_matmul_precision)!='highest'):
                raise RuntimeError('GPU/f64/highest preflight required')
            stage='common_rule'
            geometry=load_npz_bounded(files['common_geometry']);keys={'points_physical','points_dimensionless','weights_dimensionless','jacobians_physical'}
            if set(geometry)!=keys:raise ValueError('complete common geometry required')
            n=len(geometry['weights_dimensionless'])
            shapes={'points_physical':(n,2),'points_dimensionless':(n,2),'weights_dimensionless':(n,),'jacobians_physical':(n,2,2)}
            if (not 0<n<=release['max_common_points'] or n!=selection.get('common_rule_points') or
                any(geometry[k].shape!=v or not np.isfinite(geometry[k]).all() for k,v in shapes.items()) or
                np.any(geometry['weights_dimensionless']<=0) or np.any(np.linalg.det(geometry['jacobians_physical'])<=0) or
                not np.array_equal(geometry['points_dimensionless'],geometry['points_physical']/cfg['D'])):
                raise ValueError('invalid full common geometry/scaling')
            rule_hash=array_digest({k:geometry[k] for k in ('points_dimensionless','weights_dimensionless')})
            if rule_hash!=selection.get('volume_rule_sha256'):raise ValueError('original common rule digest differs')
            for array in geometry.values():array.setflags(write=False)
            stage='native_imports';modules=load_native_modules();import ngsolve as ng
            import scipy
            runtime={'python':platform.python_version(),'ngsolve':ng.__version__,'numpy':np.__version__,
                     'scipy':scipy.__version__,'jax':jax.__version__}
            if runtime!=selection.get('runtime_versions'):raise ValueError('selected native runtime versions differ')
            report['runtime_versions']=runtime
            ng.SetNumThreads(release['cores'])
            stage='candidate_lift'
            lift=modules['candidate_lift'].CandidateStokesLift.from_eligibility(eligibility_path,
                sha_file(files['mesh']),layout['sha256'],cfg['U'],cfg['D'],cfg,
                expected_truth_admission_sha256=release['reference_admission_sha256'],
                expected_selection_sha256=release.get('selection_sha256'))
            stage='query_construction'
            factory=modules['fom_outputs'].make_output_factory(ng,
                {'points':geometry['points_physical'],'weights':geometry['weights_dimensionless']*cfg['D']**2},
                read_json(files['boundary']),out/'normal_orientation.json',
                field_rule_mode=selection['field_rule_mode'],requested_point_cap=release['max_common_points'])
            query=modules['fom_query'].SteadyFOMQuery(ng,files['mesh'],cfg,lift,factory)
            self=cls();self.native=query;self.release=release;self.qualification=qualification
            self.release_file_sha256=sha_file(release_path)
            self.rule_hash=rule_hash;self.points=geometry['points_physical'];self.weights=geometry['weights_dimensionless']
            self.configuration_digest=modules['candidate_lift'].configuration_sha256
            self.expected_configuration=self.configuration_digest(cfg)
            self.binding_digest=self.configuration_digest({'release':release,'qualification':qualification})
            self.check_bindings()
            report.update(status='constructed_pending_actual_paired_invocations',native_query_offline_seconds=query.offline_seconds,
                          lift_load_seconds=lift.offline_load_seconds,common_rule_points=n,volume_rule_sha256=rule_hash,
                          field_rule_mode=selection['field_rule_mode'])
            return self
        except Exception as exc:
            report.update(status='construction_failed',failure_stage=stage,exception=repr(exc));raise
        finally:
            report.update(total_offline_seconds=time.perf_counter()-started,
                          peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            try:(out/'CONSTRUCTION.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            except Exception:
                if report['status']!='construction_failed':raise

    def check_bindings(self):
        lift=self.native.lift;r=self.release
        if self.configuration_digest({'release':r,'qualification':self.qualification})!=self.binding_digest:
            raise ValueError('runtime release/scope binding changed')
        if (lift.usage_role!='frozen_fom_comparator' or lift.selection_frozen is not True or
            lift.selection_sha256!=r['selection_sha256'] or lift.selection_review_sha256!=r['selection_review_sha256'] or
            lift.eligibility_sha256!=r['eligibility_sha256'] or
            lift.truth_reference_admission_sha256!=r['reference_admission_sha256'] or
            lift.configuration_sha256!=self.expected_configuration or
            self.configuration_digest(self.native.cfg)!=self.expected_configuration):
            raise ValueError('frozen FOM binding changed')

    def decorate(self,invocation):
        invocation=dict(invocation)
        invocation['frozen_comparator']={k:self.release[k] for k in
            ('selection_sha256','selection_review_sha256','eligibility_sha256','reference_admission_sha256',
             'consumer_source_sha256','reference_population_sha256','output_contract','population','bank_target')}
        invocation['frozen_comparator'].update(reference_qualification=self.qualification,
            runtime_release_sha256=self.release_file_sha256,volume_rule_sha256=self.rule_hash)
        return invocation

    def query(self,reynolds):
        self.check_bindings()
        if reynolds not in self.release['reynolds']:raise ValueError('parameter outside explicitly released population')
        try:return self.decorate(self.native.query(reynolds,self.release['output_contract']))
        except Exception as exc:
            if isinstance(getattr(exc,'invocation',None),dict):exc.invocation=self.decorate(exc.invocation)
            raise

    def paired(self,reynolds,population_path,rom_calls,rom_score,output_directory):
        """Concrete actual FOM callable plus caller's frozen ROM callables, one job."""
        from pilot2d import load_reference_population,paired_queries
        from fom_bridge2d import score_fom_invocation
        self.check_bindings()
        if reynolds not in self.release['reynolds'] or not rom_calls or 'FOM' in rom_calls:
            raise ValueError('released parameter and separate ROM methods required')
        expected=self.release['reference_population_sha256']
        if sha_file(population_path)!=expected:raise ValueError('released reference population differs')
        record=read_json(population_path)
        if record.get('reference_admission_sha256')!=self.release['reference_admission_sha256']:
            raise ValueError('reference population truth identity differs')
        cases=load_reference_population(population_path,expected,self.release['population'],self.rule_hash)
        reference=next(row for row in cases if row['Re']==reynolds)
        if require_qualification(reference['reference_qualification'])!=self.qualification:raise ValueError('reference scope differs')
        out=Path(output_directory);out.mkdir(parents=True,exist_ok=False)
        calls={'FOM':lambda:self.query(reynolds),**rom_calls}
        def score(method,invocation):
            value=(score_fom_invocation(self.native,invocation,reference,self.weights,self.points) if method=='FOM'
                   else rom_score(method,invocation,reference))
            return dict(value,reference_population_sha256=expected,reference_Re=reynolds)
        return paired_queries(calls,self.release['output_contract'],self.release['cores'],score,out)
