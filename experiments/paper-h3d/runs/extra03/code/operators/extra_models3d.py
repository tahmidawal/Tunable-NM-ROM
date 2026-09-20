"""Double-precision DeepONet and patchified structured-mesh Transolver in 3D.

Transolver physics attention follows THUML's MIT-licensed implementation pinned
in upstream/PROVENANCE.json. Copyright (c) 2024 THUML @ Tsinghua University;
see upstream/LICENSE. Changes: functional JAX, f64, 3D patch tokens, explicit
coordinate channels, optional periodic convolution/padding; no dropout.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from .models3d import _dense, _conv, _linear, _convolve


def _norm_params(width):
    return dict(scale=jnp.ones(width, jnp.float64), bias=jnp.zeros(width, jnp.float64))


def _bias_free(key, width):
    return dict(w=_dense(key,width,width)['w'])


def _norm(p, x):
    mu=jnp.mean(x, axis=-1, keepdims=True)
    return (x-mu)*jax.lax.rsqrt(jnp.mean((x-mu)**2,axis=-1,keepdims=True)+1e-5)*p['scale']+p['bias']


def _split_channels(x, spec):
    indices=tuple(i % x.shape[-1] for i in spec.get('coordinate_channels', [-3,-2,-1]))
    assert len(indices)==3 and len(set(indices))==3
    fields=tuple(i for i in range(x.shape[-1]) if i not in indices)
    assert fields, 'At least one supplied field/coefficient channel is required'
    return x[...,jnp.array(fields)],x[...,jnp.array(indices)]


def _pool2(x, periodic):
    if periodic:
        x=jnp.pad(x,((0,0),*((0,d%2) for d in x.shape[1:4]),(0,0)),mode='wrap')
    return jax.lax.reduce_window(x,0.,jax.lax.add,(1,2,2,2,1),(1,2,2,2,1),'VALID' if periodic else 'SAME')/8.


def _adaptive_pool(x, bins):
    """Exact separable average of fixed spatial bins (overlap at fractional cuts)."""
    for axis in (1,2,3):
        n=x.shape[axis]
        x=jnp.stack([jnp.mean(jnp.take(x,jnp.arange(i*n//bins,((i+1)*n+bins-1)//bins),axis=axis),axis=axis)
                     for i in range(bins)],axis=axis)
    return x


def _trunk_features(coords, spec):
    features=[coords]
    for frequency in spec.get('trunk_frequencies', [1.,2.,4.]):
        features.extend([jnp.sin(jnp.pi*frequency*coords),jnp.cos(jnp.pi*frequency*coords)])
    return jnp.concatenate(features,axis=-1)


def init_physics(key, width, heads, slices):
    assert width%heads==0
    keys=iter(jax.random.split(key,9));dim=width//heads
    # The slice projection is orthogonal, as in upstream (transposed IO layout).
    orth=jax.nn.initializers.orthogonal()(next(keys),(dim,slices),jnp.float64)
    return dict(x=_conv(next(keys),width,width),fx=_conv(next(keys),width,width),
                slice=dict(w=orth,b=jnp.zeros(slices,jnp.float64)),
                q=_bias_free(next(keys),dim),k=_bias_free(next(keys),dim),v=_bias_free(next(keys),dim),
                out=_dense(next(keys),width,width),temperature=jnp.full((1,heads,1,1),.5,jnp.float64))


def physics_core(p, x_mid, fx_mid):
    """B H N D -> B H N D; exposed for independent NumPy equation verification."""
    weights=jax.nn.softmax(_linear(p['slice'],x_mid)/jnp.clip(p['temperature'],.1,5.),axis=-1)
    norm=jnp.sum(weights,axis=2)
    token=jnp.einsum('bhnd,bhns->bhsd',fx_mid,weights,precision='highest')/(norm[...,None]+1e-5)
    # Upstream Q/K/V are bias-free.
    q=token@p['q']['w'];k=token@p['k']['w'];v=token@p['v']['w']
    attention=jax.nn.softmax(jnp.einsum('bhsd,bhtd->bhst',q,k,precision='highest')/jnp.sqrt(q.shape[-1]),axis=-1)
    out_token=attention@v
    return jnp.einsum('bhsd,bhns->bhnd',out_token,weights,precision='highest')


def _physics(p, x, heads, periodic):
    b,nx,ny,nz,width=x.shape;d=width//heads
    def headed(y):return y.reshape(b,nx*ny*nz,heads,d).transpose(0,2,1,3)
    out=physics_core(p,headed(_convolve(p['x'],x,periodic)),headed(_convolve(p['fx'],x,periodic)))
    return _linear(p['out'],out.transpose(0,2,1,3).reshape(b,nx,ny,nz,width))


def init_extra(key, spec, in_channels, out_channels):
    assert in_channels>3, 'Explicit three coordinate channels plus supplied field required'
    keys=iter(jax.random.split(key,100));width=spec['width']
    if spec['kind']=='deeponet3d':
        rank=spec.get('rank',128);tw=spec.get('trunk_width',max(width,rank))
        assert tw>=rank, 'Declared trunk rank requires trunk_width >= rank'
        blocks=[];ci=in_channels-3
        for level in range(spec.get('levels',3)):
            co=width*2**level;blocks.append([_conv(next(keys),ci,co),_conv(next(keys),co,co)]);ci=co
        nfeatures=3*(1+2*len(spec.get('trunk_frequencies',[1.,2.,4.])))
        return dict(branch=blocks,branch_hidden=_dense(next(keys),ci*spec.get('pool_bins',4)**3,tw),
                    branch_read=_dense(next(keys),tw,rank*out_channels,scale=.1),
                    trunk=[_dense(next(keys),nfeatures,tw),_dense(next(keys),tw,tw),_dense(next(keys),tw,rank)],
                    bias=jnp.zeros(out_channels,jnp.float64))
    if spec['kind']=='transolver3d':
        heads=spec.get('heads',4);slices=spec.get('slices',16);patch=spec.get('patch',2)
        ref=spec.get('reference_grid',4);ratio=spec.get('mlp_ratio',2);blocks=[]
        for _ in range(spec.get('depth',4)):
            blocks.append(dict(norm1=_norm_params(width),attention=init_physics(next(keys),width,heads,slices),
                               norm2=_norm_params(width),mlp1=_dense(next(keys),width,ratio*width),
                               mlp2=_dense(next(keys),ratio*width,width)))
        return dict(lift1=_dense(next(keys),(in_channels-3)*patch**3+ref**3,2*width),
                    lift2=_dense(next(keys),2*width,width),placeholder=jax.random.uniform(next(keys),(width,),jnp.float64)/width,
                    blocks=blocks,norm=_norm_params(width),read=_dense(next(keys),width,out_channels*patch**3,scale=.1))
    raise ValueError(spec['kind'])


def _patches(fields, coords, spec):
    b,nx,ny,nz,ci=fields.shape;p=spec.get('patch',2)
    pads=[((-n)%p//2,(-n)%p-(-n)%p//2) for n in (nx,ny,nz)]
    fields=jnp.pad(fields,((0,0),*pads,(0,0)),mode='wrap' if spec.get('periodic',False) else 'constant')
    sizes=tuple(n//p for n in fields.shape[1:4])
    tokens=fields.reshape(b,sizes[0],p,sizes[1],p,sizes[2],p,ci).transpose(0,1,3,5,2,4,6,7).reshape(b,*sizes,p**3*ci)
    # Explicit affine lattice coordinates preserve interior vs nodal vs periodic
    # conventions. Extrapolate token centres through padding, never infer endpoints.
    origin=coords[:,0,0,0,:]
    deltas=[coords[:,1,0,0,:]-origin,coords[:,0,1,0,:]-origin,coords[:,0,0,1,:]-origin]
    centres=jnp.broadcast_to(origin[:,None,None,None,:],(b,*sizes,3))
    for axis,(size,pad,delta) in enumerate(zip(sizes,pads,deltas)):
        position=jnp.arange(size,dtype=jnp.float64)*p+(p-1)/2-pad[0]
        shape=[1,1,1,1,1];shape[axis+1]=size
        centres=centres+position.reshape(shape)*delta[:,None,None,None,:]
    bounds=spec.get('coordinate_bounds',[-1.,1.]);lo,hi=bounds
    unit=(centres-lo)/(hi-lo)
    ref=spec.get('reference_grid',4);a=jnp.linspace(0.,1.,ref,dtype=jnp.float64)
    reference=jnp.stack(jnp.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)
    distances=jnp.sqrt(jnp.sum((unit[...,None,:]-reference)**2,axis=-1)+1e-30)
    return jnp.concatenate((tokens,distances),axis=-1),pads


def apply_extra(params, x, spec):
    fields,coords=_split_channels(x,spec)
    if spec['kind']=='deeponet3d':
        h=fields
        for block in params['branch']:
            for p in block:h=jax.nn.gelu(_convolve(p,h,spec.get('periodic',False)),approximate=False)
            h=_pool2(h,spec.get('periodic',False))
        h=_adaptive_pool(h,spec.get('pool_bins',4)).reshape(x.shape[0],-1)
        h=jax.nn.gelu(_linear(params['branch_hidden'],h),approximate=False)
        coeff=_linear(params['branch_read'],h).reshape(x.shape[0],-1,spec.get('rank',128))
        trunk=_trunk_features(coords,spec)
        for p in params['trunk'][:-1]:trunk=jax.nn.tanh(_linear(p,trunk))
        trunk=_linear(params['trunk'][-1],trunk)
        return jnp.einsum('bcr,bxyzr->bxyzc',coeff,trunk,precision='highest')/jnp.sqrt(trunk.shape[-1])+params['bias']
    if spec['kind']=='transolver3d':
        h,pads=_patches(fields,coords,spec)
        h=_linear(params['lift2'],jax.nn.gelu(_linear(params['lift1'],h),approximate=False))+params['placeholder']
        for block in params['blocks']:
            h=h+_physics(block['attention'],_norm(block['norm1'],h),spec.get('heads',4),spec.get('periodic',False))
            h=h+_linear(block['mlp2'],jax.nn.gelu(_linear(block['mlp1'],_norm(block['norm2'],h)),approximate=False))
        y=_linear(params['read'],_norm(params['norm'],h));p=spec.get('patch',2);b,nx,ny,nz,packed=y.shape
        y=y.reshape(b,nx,ny,nz,p,p,p,packed//p**3).transpose(0,1,4,2,5,3,6,7).reshape(b,nx*p,ny*p,nz*p,-1)
        return y[:,pads[0][0]:pads[0][0]+x.shape[1],pads[1][0]:pads[1][0]+x.shape[2],pads[2][0]:pads[2][0]+x.shape[3],:]
    raise ValueError(spec['kind'])
