"""Fixed increment, saturation and analytic-J fixtures; no new cohort outcomes."""
import numpy as np
import shared_initializer as I


def check_network(params,anchor):
    anchor=np.asarray(anchor);k=len(anchor)
    sine=np.sin(np.arange(1,k+1,dtype=np.float64));sine/=np.linalg.norm(sine)
    directions=[('sine',sine)]+[(f'coordinate_{i}',v) for i,v in enumerate(np.eye(k))]
    jac=np.asarray(I.H.head_jacobian(params,I.jnp.asarray(anchor)))
    rows=[];centrals=[];passed=True
    for name,direction in directions:
        action=jac@direction;scale=max(np.linalg.norm(action),1e-30);local=[]
        for h in (1e-2,1e-4,1e-6,1e-8,1e-10,1e-12):
            d=h*direction
            increment=np.asarray(I.head_increment(params,I.jnp.asarray(anchor),I.jnp.asarray(d)))
            direct=np.asarray(I.H.head_apply(params,I.jnp.asarray(anchor+d))-I.H.head_apply(params,I.jnp.asarray(anchor)))
            base=np.asarray(I.H.head_apply(params,I.jnp.asarray(anchor)))
            parity=float(np.linalg.norm(increment-direct)/max(1.,np.linalg.norm(base)))
            taylor=float(np.linalg.norm(increment-h*action)/(h*scale))
            local.append({'direction':name,'step':h,'absolute_increment_difference':float(np.linalg.norm(increment-direct)),
                'scale_relative_increment_parity':parity,'first_order_relative_remainder':taylor,
                'increment_norm':float(np.linalg.norm(increment))})
        rows.extend(local)
        passed=passed and all(r['scale_relative_increment_parity']<=1e-12 for r in local)
        passed=passed and local[-1]['first_order_relative_remainder']<=1e-7
        passed=passed and local[2]['first_order_relative_remainder']<=1e-3
        h=1e-5
        plus=np.asarray(I.head_increment(params,I.jnp.asarray(anchor),I.jnp.asarray(h*direction)))
        minus=np.asarray(I.head_increment(params,I.jnp.asarray(anchor),I.jnp.asarray(-h*direction)))
        central=float(np.linalg.norm((plus-minus)/(2*h)-action)/scale)
        passed=passed and central<=1e-6
        centrals.append({'direction':name,'relative_difference':central})
    return {'rows':rows,'central_jacobian_actions':centrals,'passed':bool(passed)}


@I.jax.jit
def activation_ranges(params,codes):
    a1=codes@params['W1']+params['b1']
    a2=I.jax.nn.silu(a1)@params['W2']+params['b2']
    return I.jnp.stack((I.jnp.min(a1,axis=1),I.jnp.max(a1,axis=1),
                       I.jnp.min(a2,axis=1),I.jnp.max(a2,axis=1)),axis=1)


def check_ranges(params,codes):
    ranges=np.asarray(activation_ranges(params,I.jnp.asarray(codes)))
    finite=bool(np.all(np.isfinite(ranges)))
    return {'columns':['layer1_min','layer1_max','layer2_min','layer2_max'],
            'per_endpoint':ranges,'max_absolute':float(np.max(np.abs(ranges))),
            'passed':finite and bool(np.max(np.abs(ranges))<600.),
            'range_limit':600.,'scope':'safe exponential arithmetic for independent host audit'}


def check_saturation():
    a=np.asarray([-1000.,-100.,-40.,-1.,0.,1.,40.,100.,1000.])
    records=[]
    for d in (-1000.,-.5-1e-12,-.5,-.5+1e-12,-1e-12,0.,1e-12,.5-1e-12,.5,.5+1e-12,1000.):
        got=np.asarray(I.silu_increment(I.jnp.asarray(a),I.jnp.full(a.shape,d)))
        direct=np.asarray(I.jax.nn.silu(I.jnp.asarray(a+d))-I.jax.nn.silu(I.jnp.asarray(a)))
        records.append({'delta':d,'finite':bool(np.all(np.isfinite(got))),
            'scale_relative_difference':float(np.max(np.abs(got-direct)/np.maximum(1.,np.abs(a)+abs(d))))})
    return {'rows':records,'passed':all(r['finite'] and r['scale_relative_difference']<=1e-12 for r in records)}
