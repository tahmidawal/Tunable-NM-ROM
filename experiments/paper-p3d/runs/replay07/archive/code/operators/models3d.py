"""Reusable double-precision 3D FNO and U-Net primitives, channels last.

API: init_model(key, spec, in_channels, out_channels); apply_model(p, x, spec).
Inputs have shape (batch, nx, ny, nz, channels), all three axes spatial.
Physical normalization, coordinates, boundary handling and output times belong to
an explicit task adapter, not hidden PDE-specific parameter inputs in this module.
All trainable arrays are real float64, including real/imaginary Fourier weights.
"""
from __future__ import annotations
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def _dense(key, ci, co, scale=1.):
    return {'w': scale * jax.random.normal(key, (ci, co), dtype=jnp.float64) / jnp.sqrt(ci),
            'b': jnp.zeros((co,), dtype=jnp.float64)}


def _conv(key, ci, co, kernel=3, scale=1.):
    return {'w': scale * jax.random.normal(key, (kernel,)*3+(ci,co), dtype=jnp.float64)/jnp.sqrt(ci*kernel**3),
            'b': jnp.zeros((co,), dtype=jnp.float64)}


def _linear(p, x):
    return x @ p['w'] + p['b']


def _convolve(p, x, periodic=False):
    if periodic:
        pad=p['w'].shape[0]//2
        x=jnp.pad(x, ((0,0),(pad,pad),(pad,pad),(pad,pad),(0,0)),mode='wrap')
        padding='VALID'
    else: padding='SAME'
    return jax.lax.conv_general_dilated(x,p['w'],(1,1,1),padding,
              dimension_numbers=('NDHWC','DHWIO','NDHWC'),precision=jax.lax.Precision.HIGHEST)+p['b']


def init_model(key, spec, in_channels, out_channels):
    if spec['kind'] in ('deeponet3d', 'transolver3d'):
        from .extra_models3d import init_extra
        return init_extra(key, spec, in_channels, out_channels)
    keys=iter(jax.random.split(key,100));width=spec['width'];kind=spec['kind']
    if kind=='fno3d':
        modes=tuple(spec['modes']);layers=[]
        for _ in range(spec.get('depth',4)):
            shape=(2*modes[0],2*modes[1],modes[2],width,width,2)
            # Independent positive/negative x/y modes; final-axis rFFT half plane.
            spectral=jax.random.normal(next(keys),shape,dtype=jnp.float64)/(width*jnp.sqrt(2.))
            layers.append({'spectral':spectral,'local':_dense(next(keys),width,width)})
        return {'lift':_dense(next(keys),in_channels,width),'layers':layers,
                'read1':_dense(next(keys),width,2*width),
                'read2':_dense(next(keys),2*width,out_channels,scale=.1)}
    if kind=='unet3d':
        widths=[width*2**i for i in range(spec.get('levels',3))]
        enc=[];ci=in_channels
        for co in widths:
            enc.append([_conv(next(keys),ci,co),_conv(next(keys),co,co)]);ci=co
        dec=[]
        for co in widths[-2::-1]:
            dec.append([_conv(next(keys),ci+co,co),_conv(next(keys),co,co)]);ci=co
        return {'encoder':enc,'decoder':dec,'read':_conv(next(keys),width,out_channels,kernel=1,scale=.1)}
    raise ValueError(kind)


def spectral_convolve(weights, x, modes):
    nx,ny,nz=x.shape[1:4];mx,my,mz=modes
    assert 2*mx<=nx and 2*my<=ny and mz<=nz//2+1
    spectrum=jnp.fft.rfftn(x,axes=(1,2,3))
    ix=jnp.concatenate((jnp.arange(mx),jnp.arange(nx-mx,nx)))
    iy=jnp.concatenate((jnp.arange(my),jnp.arange(ny-my,ny)))
    selected=spectrum[:,ix[:,None],iy[None,:],:mz,:]
    complex_weights=weights[...,0]+1j*weights[...,1]
    transformed=jnp.einsum('bxyzc,xyzco->bxyzo',selected,complex_weights,precision='highest')
    out=jnp.zeros((x.shape[0],nx,ny,nz//2+1,weights.shape[-2]),dtype=jnp.complex128)
    out=out.at[:,ix[:,None],iy[None,:],:mz,:].set(transformed)
    return jnp.fft.irfftn(out,s=(nx,ny,nz),axes=(1,2,3))


def apply_model(params, x, spec):
    assert x.dtype==jnp.float64
    if spec['kind'] in ('deeponet3d', 'transolver3d'):
        from .extra_models3d import apply_extra
        return apply_extra(params, x, spec)
    if spec['kind']=='fno3d':
        nx,ny,nz=x.shape[1:4];pad=spec.get('padding',0)
        h=_linear(params['lift'],x)
        if pad:h=jnp.pad(h,((0,0),(0,pad),(0,pad),(0,pad),(0,0)))
        for layer in params['layers']:
            h=jax.nn.gelu(spectral_convolve(layer['spectral'],h,tuple(spec['modes']))+_linear(layer['local'],h),approximate=False)
        h=h[:,:nx,:ny,:nz]
        return _linear(params['read2'],jax.nn.gelu(_linear(params['read1'],h),approximate=False))
    if spec['kind']=='unet3d':
        periodic=spec.get('periodic',False);h=x;skips=[]
        for i,block in enumerate(params['encoder']):
            for p in block:h=jax.nn.silu(_convolve(p,h,periodic))
            skips.append(h)
            if i<len(params['encoder'])-1:
                # Explicit spatial average pooling; no reduction over channels/batch.
                if periodic:
                    h=jnp.pad(h,((0,0),*( (0,dim%2) for dim in h.shape[1:4]),(0,0)),mode='wrap')
                h=jax.lax.reduce_window(h,0.,jax.lax.add,(1,2,2,2,1),(1,2,2,2,1),'VALID' if periodic else 'SAME')/8
        for block,skip in zip(params['decoder'],skips[-2::-1]):
            if periodic:
                for axis,size in zip((1,2,3),skip.shape[1:4]):
                    position=(jnp.arange(size,dtype=jnp.float64)+.5)*h.shape[axis]/size-.5
                    left=jnp.floor(position).astype(jnp.int32);fraction=position-left
                    shape=[1]*5;shape[axis]=size;fraction=fraction.reshape(shape)
                    h=(1-fraction)*jnp.take(h,left%h.shape[axis],axis=axis)+fraction*jnp.take(h,(left+1)%h.shape[axis],axis=axis)
            else:
                h=jax.image.resize(h,(h.shape[0],*skip.shape[1:4],h.shape[-1]),method='linear')
            h=jnp.concatenate((h,skip),axis=-1)
            for p in block:h=jax.nn.silu(_convolve(p,h,periodic))
        return _convolve(params['read'],h,periodic)
    raise ValueError(spec['kind'])


def parameter_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
