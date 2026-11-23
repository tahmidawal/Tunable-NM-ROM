"""Complete-query adapter sharing the audited initializer with the phase path.

The inherited compiled evolution is unchanged. Host fit/audit overhead is part
of this implementation, and no new timing or speed qualification is claimed.
"""
import numpy as np
import support as S
import legacy_query as L
import fit_api as A
jax,jnp,H,SR=S.jax,S.jnp,S.H,S.SR


def make_query(*,n,rank,latent,dt,steps,output_every,budget=250):
    if steps%output_every:raise ValueError('output interval must divide steps')
    _,prepare,resjac=SR.make_frozen_step(dt,n,rank,1,1e-6)
    def coeff(w,params):return H.head_apply(params,w[:latent])
    def evaluate(w,previous,operators,params):
        a=coeff(w,params)
        residual,jac=resjac(jnp.r_[a,w[latent:]],previous,operators)
        return residual,jnp.concatenate((jac[:,:rank]@H.head_jacobian(params,w[:latent]),jac[:,rank:]),axis=1)
    lm=L.make_lm(evaluate,budget,True)

    @jax.jit
    def evolve(initial,center,viscosity,basis,opA,T,lam,Dd,params,gtol):
        operators=prepare(viscosity,opA,T,lam,Dd)
        def decode(w,c):return SR.shift_field((basis@coeff(w,params)).reshape(3,n,n,n),c*n)
        def step(carry,_):
            w,old,c=carry
            new,info=lm(w+(w-old),(coeff(w,params),operators,params),gtol)
            c=(c+new[-3:])%1.
            return (new,w,c),info
        def block(carry,_):
            carry,infos=jax.lax.scan(step,carry,None,length=output_every)
            return carry,(decode(carry[0],carry[2]),infos)
        _,(fields,infos)=jax.lax.scan(block,(initial,initial,center),None,length=steps//output_every)
        return jnp.concatenate((decode(initial,center)[None],fields)),infos.reshape(steps,6)

    def query(initial,viscosity,basis,opA,T,lam,Dd,params,codes,candidates,gtol,fit_gtol):
        fit_field,record=A.fit_initial_field(initial,basis,params,codes,candidates,float(fit_gtol))
        w=jnp.r_[jnp.asarray(record['code']),jnp.zeros(3)]
        fields,infos=evolve(w,jnp.asarray(record['center']),viscosity,basis,opA,T,lam,Dd,params,gtol)
        fields,infos=np.asarray(fields),np.asarray(infos)
        parity=float(np.linalg.norm(fields[0]-fit_field)/record['normalization'])
        record['query_initial_output_parity']=parity
        return {'fields':fields,'step_info':infos,'initializer':record,
                'qualified':bool(record['qualified'] and parity<=1e-12 and np.all(infos[:,0]==1)),
                'timing_claim':False}
    return query
