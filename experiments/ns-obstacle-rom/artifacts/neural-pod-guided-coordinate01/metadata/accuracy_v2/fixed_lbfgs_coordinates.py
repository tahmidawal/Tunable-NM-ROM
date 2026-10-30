"""Prospective fixed-coordinate L-BFGS algebra; no trainer integration or scales.

Physical theta = anchor + D*z, g_z=D*g_theta. With physical s,y pairs,
this two-loop is exactly the z-coordinate direction mapped back by D in exact
arithmetic. Pair acceptance and convergence remain explicit physical tests.
"""
import numpy as np


def diagonal(hidden_scale,output_scale):
    if isinstance(hidden_scale,(bool,np.bool_)) or isinstance(output_scale,(bool,np.bool_)):raise ValueError('boolean scales are not a protocol')
    scales=np.asarray([hidden_scale,output_scale],dtype=np.float64)
    if scales.shape!=(2,) or not np.isfinite(scales).all() or np.min(scales)<=0:raise ValueError('fixed finite positive scales required')
    head=np.concatenate([np.full(21*64+64+64*64+64,scales[0]),np.full(64*2+2,scales[1])])
    out=np.tile(head,4)
    assert out.shape==(22792,)
    return out


def validate(vector,scale):
    v=np.asarray(vector);d=np.asarray(scale)
    if v.dtype!=np.float64 or d.dtype!=np.float64 or v.ndim!=1 or d.shape!=v.shape or not np.isfinite(v).all() or not np.isfinite(d).all() or np.min(d)<=0:raise ValueError('finite vector and matching positive diagonal required')
    return v,d


def physical(z,anchor,scale):
    z,d=validate(z,scale);a=np.asarray(anchor)
    if a.dtype!=np.float64 or a.shape!=z.shape or not np.isfinite(a).all():raise ValueError('finite same-layout anchor required')
    theta=a+d*z
    if not np.isfinite(theta).all():raise FloatingPointError('coordinate map overflow')
    return theta


def gradient_z(gradient_theta,scale):
    g,d=validate(gradient_theta,scale);out=d*g
    if not np.isfinite(out).all():raise FloatingPointError('scaled gradient overflow')
    return out


def direction(gradient_theta,pairs,scale):
    g,d=validate(gradient_theta,scale);q=g.copy();alpha=[];history=[]
    for s,y in pairs:
        s,_=validate(s,d);y,_=validate(y,d);sy=float(np.dot(s,y))
        if not np.isfinite(sy) or sy<=0:raise ValueError('retained positive curvature pairs required')
        rho=1/sy
        if not np.isfinite(rho) or rho<=0:raise FloatingPointError("invalid reciprocal curvature")
        history.append((s,y,rho))
    for s,y,rho in reversed(history):
        a=rho*np.dot(s,q);alpha.append(a);q-=a*y
    gamma=1.
    if history:
        s,y,_=history[-1]
        with np.errstate(over="ignore",under="ignore",invalid="ignore"):
            scaled_y=d*y;denominator=float(np.dot(scaled_y,scaled_y))
        if not np.isfinite(denominator) or denominator<=0:raise FloatingPointError("invalid scaled curvature denominator")
        gamma=float(np.dot(s,y)/denominator)
        if not np.isfinite(gamma) or gamma<=0:raise FloatingPointError("invalid coordinate midpoint scale")
    q=gamma*d*(d*q)
    for (s,y,rho),a in zip(history,reversed(alpha)):q+=s*(a-rho*np.dot(y,q))
    out=-q;slope=float(np.dot(g,out));reset=not np.isfinite(slope) or slope>=0
    if reset:out=-d*(d*g);slope=float(np.dot(g,out))
    if not np.isfinite(out).all() or not np.isfinite(slope):raise FloatingPointError('direction overflow')
    return out,{'H0_gamma_z':gamma,'slope_physical':slope,'direction_reset':reset,'physical_gradient_inf':float(np.max(np.abs(g))),'scaled_gradient_inf_diagnostic':float(np.max(np.abs(d*g))),'physical_stopping_required':True}
