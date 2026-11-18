"""Common local configurations and joint physical projection, never truth-assisted online."""
from common import *
import data, geometry as geo, decoder as dec, fit
from training_io import errors,summary

PAIRS=sorted([(i,j) for i in range(4) for j in range(4)],key=lambda ij:(sum(ij),ij[0]))
_PROJECTORS={}

def polynomial_inputs(v):
    x,y=v[...,0]/1.5,v[...,1]/1.5
    def polys(t):return [jnp.ones_like(t),t,(3*t*t-1)/2,(5*t**3-3*t)/2]
    px,py=polys(x),polys(y)
    return jnp.stack([px[i]*py[j] for i,j in PAIRS],axis=-1)

def polynomial_matrix(x,cells,rank):
    return jnp.concatenate([geo.window(x,c)[:,None]*polynomial_inputs(geo.local_inputs(x,c))[:,:rank] for c in cells],axis=1)

def configurations(x,w,initial_field,residual,rank,cap=128,maxdepth=3,seed=0):
    largest=4*((cap-16)//(4*rank))
    uni=geo.uniform(2) if 16+20*rank<=cap else geo.uniform(1)
    if largest<4:return [[]]*8
    prescribed=geo.allocated(x,initial_field**2,w,cap,rank,maxdepth)
    random=geo.allocated(x,initial_field**2,w,cap,rank,maxdepth,seed)
    oracle=geo.allocated(x,residual**2,w,cap,rank,maxdepth)
    def shallow(c):return c[:min(12,len(c))] if len(c)>12 else c[:4]
    return [geo.uniform(1),uni,prescribed,random,oracle,shallow(prescribed),shallow(random),shallow(oracle)]

def padded_trees(dataset,root_predictions,rank,seed):
    x=np.asarray(dataset['xy']);w=np.asarray(dataset['w']);P=4*((128-16)//(4*rank))
    cells=np.empty((len(dataset['parameters']),8,P,3),dtype=np.int32)
    masks=np.zeros(cells.shape[:-1]);trees=[]
    residual=np.asarray(dataset['Y']-root_predictions).reshape(-1,11,len(x))
    for i,p in enumerate(dataset['parameters']):
        # Callable IC is the sole input used by the prescribed branch.
        cfg=configurations(x,w,np.asarray(data.initial(x,p)),np.sqrt(np.mean(residual[i]**2,axis=0)),rank,seed=seed+i)
        trees.append(cfg)
        for q,cs in enumerate(cfg):
            geo.valid(cs);cells[i,q]=[1,0,0];cells[i,q,:len(cs)]=cs;masks[i,q,:len(cs)]=1
    return jnp.asarray(cells),jnp.asarray(masks),trees

def global_matrix(model,x):
    return dec.global_features(model['global_params'],x)@jnp.asarray(model['global_transform'])

def project_tree(model,local,child,x,w,Y,root_codes,cells,rank=4,iterations=25,starts=3,polynomial=False):
    parent=model['heads']['16'];n=len(cells);G0=global_matrix(model,x)
    B=polynomial_matrix(x,cells,rank) if polynomial else dec.local_matrix(local,x,cells,16 if child is not None else rank)
    G=jnp.concatenate([G0,B],axis=1)
    # QR reduces only the MEASUREMENT; physical feature supports stay unchanged.
    Q,R=jnp.linalg.qr(G*jnp.sqrt(w[:,None]),mode='reduced');target=(Y*jnp.sqrt(w)[None])@Q
    rootpred=dec.head(parent,root_codes)@G0.T
    direct,_,_=fit.linear_fit(B,Y-rootpred,w)
    if child is None:
        localstart=direct
    else:
        localstart=(direct.reshape(len(Y),n,16)@jnp.linalg.pinv(child['linear']).T).reshape(len(Y),-1)
    start=jnp.concatenate([root_codes,localstart],axis=1)
    zero=jnp.concatenate([root_codes,jnp.zeros_like(localstart)],axis=1)
    coefficient=lambda z:dec.coefficients(parent,child,z,n,rank)
    ident=(id(parent),id(child),n,rank,iterations)
    if ident not in _PROJECTORS:_PROJECTORS[ident]=fit.make_project(coefficient,iterations)
    projector=_PROJECTORS[ident]
    z,_,station,restarts=fit.multi_fit(projector,R,target,jnp.ones(R.shape[0]),[start,zero][:starts])
    if starts>=3:
        warm=jnp.concatenate([z[:1],z[:-1]],axis=0)
        perturb=start+jax.random.normal(jax.random.PRNGKey(101),start.shape,dtype=jnp.float64)*.01
        z,_,station,restarts2=fit.multi_fit(projector,R,target,jnp.ones(R.shape[0]),[z,warm,perturb])
        restarts=jnp.concatenate([restarts,restarts2],axis=0)
    pred=jax.vmap(coefficient)(z)@G.T;err=errors(pred,Y,w)
    return dict(codes=z,pred=pred,error=err,stationarity=station,restart_errors=restarts,
                active_variables=16+n*rank,feature_channels=G.shape[1])

def evaluate_pair(model,local,child,dataset,root_codes,rank=4,iterations=25,starts=3,cap=128,polynomial=False):
    x,w=dataset['xy'],dataset['w'];Y=dataset['Y'];rows=[];saved=[]
    for kind in ['uniform','prescribed']:
        allerr=[];allstat=[];allcodes=[];alltrees=[]
        for i,p in enumerate(dataset['parameters']):
            if kind=='uniform':
                depth=2 if 16+20*rank<=cap else 1
                cells=geo.uniform(depth) if 16+4*rank<=cap else []
            else:cells=geo.allocated(x,np.asarray(data.initial(x,p))**2,w,cap,rank)
            if not cells:continue
            fitrow=project_tree(model,local,child,x,w,Y[i*11:(i+1)*11],root_codes[i*11:(i+1)*11],cells,rank,iterations,starts,polynomial)
            allerr.append(np.asarray(fitrow['error']));allstat.append(np.asarray(fitrow['stationarity']))
            allcodes.append(np.asarray(fitrow['codes']));alltrees.append(cells)
        if not allerr:continue
        row=dict(allocation=kind,cap=cap,rank=rank,projection=summary(np.asarray(allerr)),
                 max_stationarity=float(np.max(allstat)),stationarity=np.asarray(allstat),
                 active_variables=[16+len(c)*rank for c in alltrees],trees=alltrees)
        rows.append(row);saved.append(dict(allocation=kind,codes=allcodes,trees=alltrees))
    return rows,saved
