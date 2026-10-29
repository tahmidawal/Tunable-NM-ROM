"""Production point-block consumer: full moments, bounded fixed-test storage.

A model retains only its small-prefix trial fields. The much larger fixed test bank
is consumed in original point blocks; no full gradient test bank is concatenated.
"""
from dataclasses import replace
import time
import resource
import numpy as np
from fe_adapter import check_allocation,CAPS
from weak import WeakRule,WeakOperator,force_moments
from spaces import weighted_gram,projection_floor,inf_sup
from interpolation import CoefficientPCHIP


class BlockRule:
    def __init__(self,store,fields):
        self.store=store;self.fields=fields;self.weights=store['weights_dimensionless']
    def iter_rules(self):
        f=self.fields
        for start,end,block in self.store.iter_blocks():
            s=slice(start,end)
            yield start,end,WeakRule(self.weights[s],f['lift'][s],f['lift_gradient'][s],
                f['pressure_lift'][s],f['velocity'][s],f['velocity_gradient'][s],f['pressure'][s],
                block['momentum_tests'],block['momentum_tests_gradient_dimensionless'],block['continuity_tests'])
    def tensor(self):
        total={}
        for start,end,rule in self.iter_rules():
            for key,value in rule.tensor_moments().items():
                if key not in total:total[key]=value
                else:total[key]+=value
        return WeakOperator(**total)
    def direct(self,state,nu):
        total=None
        for start,end,rule in self.iter_rules():
            values=rule.direct(state,nu)
            if total is None:total=[np.array(x,copy=True) for x in values]
            else:
                for i,x in enumerate(values):total[i]+=x
        return tuple(total)
    def galerkin(self):
        f=self.fields
        return WeakRule(self.weights,f['lift'],f['lift_gradient'],f['pressure_lift'],
            f['velocity'],f['velocity_gradient'],f['pressure'],f['velocity'],f['velocity_gradient'],f['pressure']).tensor(formulation='mixed_Galerkin')
    def fitted_convection(self,operator,fit):
        if not fit.get('passed'):raise ValueError('positive fitted rule required')
        indices=np.asarray(fit['support'],dtype=int);weights=np.asarray(fit['weights'])
        if len(indices)!=len(set(indices)) or np.min(indices)<0 or np.max(indices)>=len(self.weights):raise ValueError('invalid fitted support')
        if not np.isfinite(weights).all() or np.min(weights)<=0:raise ValueError('positive fitted weights required')
        total=np.zeros_like(operator.convection)
        for start,end,rule in self.iter_rules():
            selected=(indices>=start)&(indices<end);local=indices[selected]-start
            if not len(local):continue
            u=np.concatenate((rule.lift[:,:,None],rule.velocity),axis=2)[local]
            du=np.concatenate((rule.lift_gradient[:,:,:,None],rule.velocity_gradient),axis=3)[local]
            total+=np.einsum('n,nim,nja,nijb->mab',weights[selected],rule.tests[local],u,du,optimize=True)
        return replace(operator,convection=total)


def build_block_model(kind,prefix,arrays,metadata,probes,bank):
    from pilot2d import Model2D,TRAIN_RE,source_digest
    from geometry import CHANNEL
    started=time.perf_counter();store=arrays['volume'];n=store.point_count
    p=min(prefix,store.manifest['fields']['pressure']['columns']);nv=prefix+p
    if kind=='bank' and (bank is None or bank.geometry!=CHANNEL or bank.rank<prefix):raise ValueError('exact frozen bank required')
    if kind=='POD' and store.manifest['fields']['pod_velocity']['columns']<prefix:raise ValueError('POD prefix unavailable')
    shapes={'lift':(n,2),'lift_gradient':(n,2,2),'pressure_lift':(n,),
        'velocity':(n,2,nv),'velocity_gradient':(n,2,2,nv),'pressure':(n,p)}
    # Unlike fixed test gradients, max-prefix trial arrays remain below the original
    # single-array cap for reviewed h3 projections. Check actual sizes before allocate.
    total_bytes=sum(int(np.prod(shape))*8 for shape in shapes.values())
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+total_bytes>CAPS['process_rss_bytes']:
        raise MemoryError('trial fields plus process RSS exceed cap')
    for shape in shapes.values():check_allocation(shape)
    fields={'volume':{key:np.empty(shape,dtype=np.float64) for key,shape in shapes.items()}}
    def reconstruct(source):
        if kind=='bank':
            base,derivative=bank.evaluate(source['points_physical'],prefix);derivative*=metadata['scales']['D']
        else:
            base=source['pod_velocity'][...,:prefix];derivative=source['pod_velocity_gradient_dimensionless'][...,:prefix]
        return {'lift':source['lift_u'][...,0],'lift_gradient':source['lift_u_gradient_dimensionless'][...,0],
            'pressure_lift':source['lift_p_per_nu'][...,0],
            'velocity':np.concatenate((base,source['supremizers'][...,:p]),axis=-1),
            'velocity_gradient':np.concatenate((derivative,source['supremizers_gradient_dimensionless'][...,:p]),axis=-1),
            'pressure':source['pressure'][...,:p]}
    for start,end,source in store.iter_blocks():
        for key,value in reconstruct(source).items():fields['volume'][key][start:end]=value
    fields['boundary']=reconstruct(arrays['boundary']);fields['probes']=reconstruct(probes)
    f=fields['volume'];weights=store['weights_dimensionless'];rule=BlockRule(store,f);operator=rule.tensor()
    divergence=np.einsum('n,np,niiv->pv',weights,f['pressure'],f['velocity_gradient'])
    h1=weighted_gram(f['velocity'],weights)+weighted_gram(f['velocity_gradient'],weights)
    stability=inf_sup(divergence,h1,weighted_gram(f['pressure'],weights),metadata['reference_inf_sup'][str(p)]['beta'])
    boundary=arrays['boundary'];ids=boundary['labels']=='obstacle';bf=fields['boundary']
    force=force_moments(boundary['weights_dimensionless'][ids],-boundary['normals_fluid'][ids],
        bf['velocity_gradient'][ids],bf['pressure'][ids],bf['lift_gradient'][ids],bf['pressure_lift'][ids],1.)
    training=arrays['training']
    cu,_=projection_floor(f['velocity'],training['velocity_corrections'],weights)
    cp,_=projection_floor(f['pressure'],training['pressure_corrections'],weights)
    interpolation=None
    if kind=='POD':
        pod,_=projection_floor(f['velocity'][...,:prefix],training['velocity_corrections'],weights)
        interpolation=CoefficientPCHIP(TRAIN_RE,np.c_[pod,np.zeros((9,p)),cp],{'split':'training',
            'supremizer_coefficients':'identically zero; interpolation uses POD modes only',
            'adapter_manifest_sha256':metadata['manifest_sha256']})
    return Model2D(kind,prefix,p,fields,rule,operator,force,stability,np.c_[cu,cp],interpolation,
        {'source_sha256':source_digest(),'adapter_manifest_sha256':metadata['manifest_sha256'],
         'offline_build_seconds':time.perf_counter()-started,'pressure_gauge_shift':False,
         'native_data_admission':metadata.get('admission_evidence'),
         'fixed_tests_storage':'original point blocks','bank_evaluation':'one point block at a time'})
