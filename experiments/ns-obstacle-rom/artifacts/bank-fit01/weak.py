"""Gradient-form steady mixed weak operators in FIXED dimensionless units.

Array axes: point, vector component, derivative coordinate, column. Volume points use
x/D, velocities u/U, pressure p/U**2, nu_hat=nu/(U*D). Pressure lift is nu_hat*p_lift.
Both learned-bank and POD arms must use these same routines and the same test spaces.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from spaces import checked_cholesky,positive_weights


@dataclass
class WeakRule:
    weights: np.ndarray
    lift: np.ndarray
    lift_gradient: np.ndarray
    pressure_lift_per_nu: np.ndarray
    velocity: np.ndarray
    velocity_gradient: np.ndarray
    pressure: np.ndarray
    tests: np.ndarray
    test_gradient: np.ndarray
    scalar_tests: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            setattr(self,name,np.asarray(getattr(self,name),dtype=np.float64))
        n,d=self.lift.shape
        nv=self.velocity.shape[-1]
        m=self.tests.shape[-1]
        shapes={"lift_gradient":(n,d,d),"pressure_lift_per_nu":(n,),
                "velocity":(n,d,nv),"velocity_gradient":(n,d,d,nv),
                "tests":(n,d,m),"test_gradient":(n,d,d,m)}
        for name,shape in shapes.items():
            if getattr(self,name).shape!=shape:
                raise ValueError(f"bad {name} shape: expected {shape}")
        for name in ("pressure","scalar_tests"):
            if getattr(self,name).ndim!=2 or len(getattr(self,name))!=n:
                raise ValueError(f"bad {name} shape")
        positive_weights(self.weights,n)
        if not all(np.isfinite(getattr(self,name)).all() for name in self.__dataclass_fields__):
            raise ValueError("nonfinite rule arrays")

    def direct(self,state,nu_hat):
        """Independent pointwise assembly; returns unwhitened momentum/continuity."""
        nv=self.velocity.shape[-1]
        a=np.asarray(state,dtype=np.float64)
        if a.shape!=(nv+self.pressure.shape[-1],):
            raise ValueError("state dimension mismatch")
        u=self.lift+np.einsum("ndr,r->nd",self.velocity,a[:nv])
        du=self.lift_gradient+np.einsum("ndjr,r->ndj",self.velocity_gradient,a[:nv])
        p=nu_hat*self.pressure_lift_per_nu+self.pressure@a[nv:]
        adv=np.einsum("nj,nij->ni",u,du)
        divt=np.einsum("niim->nm",self.test_gradient)
        rm=(np.einsum("n,ni,nim->m",self.weights,adv,self.tests)+
            nu_hat*np.einsum("n,nij,nijm->m",self.weights,du,self.test_gradient)-
            np.einsum("n,n,nm->m",self.weights,p,divt))
        rc=np.einsum("n,nq,n->q",self.weights,self.scalar_tests,np.trace(du,axis1=1,axis2=2))
        return rm,rc

    def tensor_moments(self,chunk=1024,formulation="weak_residual_least_squares"):
        if formulation=="mixed_Galerkin" and not (
                np.array_equal(self.tests,self.velocity) and
                np.array_equal(self.test_gradient,self.velocity_gradient) and
                np.array_equal(self.scalar_tests,self.pressure)):
            raise ValueError("Galerkin tests must equal the actual velocity/pressure trial modes")
        nv=self.velocity.shape[-1]
        ext=np.concatenate((self.lift[:,:,None],self.velocity),axis=2)
        dext=np.concatenate((self.lift_gradient[:,:,:,None],self.velocity_gradient),axis=3)
        m=self.tests.shape[-1]
        tensor=np.zeros((m,nv+1,nv+1),dtype=np.float64)
        for start in range(0,len(ext),chunk):
            s=slice(start,min(start+chunk,len(ext)))
            # Separate chunks avoid materializing point*test*trial*trial arrays.
            tensor+=np.einsum("n,nim,nja,nijb->mab",self.weights[s],self.tests[s],ext[s],dext[s],optimize=True)
        viscous=np.einsum("n,nija,nijm->ma",self.weights,dext,self.test_gradient,optimize=True)
        divt=np.einsum("niim->nm",self.test_gradient)
        pressure=-np.einsum("n,nk,nm->mk",self.weights,self.pressure,divt)
        plift=-np.einsum("n,n,nm->m",self.weights,self.pressure_lift_per_nu,divt)
        continuity=np.einsum("n,nq,niia->qa",self.weights,self.scalar_tests,dext)
        vg=(np.einsum("n,nim,nil->ml",self.weights,self.tests,self.tests)+
            np.einsum("n,nijm,nijl->ml",self.weights,self.test_gradient,self.test_gradient))
        qg=np.einsum("n,nq,nr->qr",self.weights,self.scalar_tests,self.scalar_tests)
        ug=np.einsum("n,nia,nib->ab",self.weights,self.velocity,self.velocity)
        pg=np.einsum("n,na,nb->ab",self.weights,self.pressure,self.pressure)
        return dict(convection=tensor,viscous=viscous,pressure=pressure,pressure_lift=plift,
                    continuity=continuity,velocity_test_gram=vg,scalar_test_gram=qg,
                    velocity_unknown_gram=ug,pressure_unknown_gram=pg)

    def tensor(self,chunk=1024,formulation="weak_residual_least_squares"):
        return WeakOperator(**self.tensor_moments(chunk,formulation),formulation=formulation)


@dataclass
class WeakOperator:
    convection: np.ndarray
    viscous: np.ndarray
    pressure: np.ndarray
    pressure_lift: np.ndarray
    continuity: np.ndarray
    velocity_test_gram: np.ndarray
    scalar_test_gram: np.ndarray
    velocity_unknown_gram: np.ndarray
    pressure_unknown_gram: np.ndarray
    formulation: str = "weak_residual_least_squares"

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name!="formulation":
                setattr(self,name,np.asarray(getattr(self,name),dtype=np.float64))
        self.lv=checked_cholesky(self.velocity_test_gram)
        self.lq=checked_cholesky(self.scalar_test_gram)
        a=checked_cholesky(self.velocity_unknown_gram)
        p=checked_cholesky(self.pressure_unknown_gram)
        self.unknown_cholesky=np.zeros((len(a)+len(p),len(a)+len(p)))
        self.unknown_cholesky[:len(a),:len(a)]=a
        self.unknown_cholesky[len(a):,len(a):]=p
        self.unknown_map=np.linalg.inv(self.unknown_cholesky.T)
        self.nv=len(a)
        self.np=len(p)
        if self.formulation=="weak_residual_least_squares":
            if len(self.lv)+len(self.lq)<=self.nv+self.np:
                raise ValueError("weak LS requires strictly more independent test rows than unknowns")
        elif self.formulation=="mixed_Galerkin":
            if (len(self.lv),len(self.lq))!=(self.nv,self.np):
                raise ValueError("mixed Galerkin needs matched velocity and pressure test dimensions")
        else:
            raise ValueError("unsupported reduced formulation")

    def terms(self,state,nu_hat):
        z=np.asarray(state,dtype=np.float64)
        if z.shape!=(self.nv+self.np,) or not np.isfinite(z).all() or not np.isfinite(nu_hat) or nu_hat<=0:
            raise ValueError("finite state and positive viscosity required")
        x=np.r_[1.,z[:self.nv]]
        return {"advection":np.einsum("mab,a,b->m",self.convection,x,x),
                "viscous":nu_hat*self.viscous@x,
                "pressure":self.pressure@z[self.nv:],
                "pressure_lifting":nu_hat*self.pressure_lift,
                "continuity":self.continuity@x}

    def residual(self,state,nu_hat,whiten=True):
        terms=self.terms(state,nu_hat)
        m=sum(terms[k] for k in ("advection","viscous","pressure","pressure_lifting"))
        c=terms["continuity"]
        if whiten:
            m=np.linalg.solve(self.lv,m)
            c=np.linalg.solve(self.lq,c)
        return np.r_[m,c]

    def jacobian_terms(self,state,nu_hat):
        z=np.asarray(state,dtype=np.float64)
        x=np.r_[1.,z[:self.nv]]
        sym=self.convection+self.convection.transpose(0,2,1)
        shape=(len(self.lv)+len(self.lq),self.nv+self.np)
        result={k:np.zeros(shape) for k in ("advection","viscous","pressure","continuity")}
        result["advection"][:len(self.lv),:self.nv]=np.einsum("mab,b->ma",sym,x)[:,1:]
        result["viscous"][:len(self.lv),:self.nv]=nu_hat*self.viscous[:,1:]
        result["pressure"][:len(self.lv),self.nv:]=self.pressure
        result["continuity"][len(self.lv):,:self.nv]=self.continuity[:,1:]
        return result

    def whiten_matrix(self,jac,columns=False):
        out=np.r_[np.linalg.solve(self.lv,jac[:len(self.lv)]),np.linalg.solve(self.lq,jac[len(self.lv):])]
        return out@self.unknown_map if columns else out

    def jacobian(self,state,nu_hat,whiten=True,columns=False):
        jac=sum(self.jacobian_terms(state,nu_hat).values())
        if whiten:
            return self.whiten_matrix(jac,columns)
        if columns:
            raise ValueError("column-only whitening is not a supported contract")
        return jac

    def condition(self,state,nu_hat):
        sv=np.linalg.svd(self.jacobian(state,nu_hat,columns=True),compute_uv=False)
        rank=int(np.sum(sv>1e-12*sv[0])) if len(sv) and sv[0]>0 else 0
        cond=float(sv[0]/sv[-1]) if rank==len(sv) else float("inf")
        return {"rank":rank,"columns":len(sv),"singular_values":sv.tolist(),"condition":cond,
                "passed":bool(rank==self.nv+self.np and cond<=1e10)}

    def quadrature_gap(self,other,state,nu_hat):
        """Full-rule normalization and whitening stay frozen; compare another raw operator."""
        base=self.terms(state,nu_hat)
        diff=self.residual(state,nu_hat,False)-other.residual(state,nu_hat,False)
        wr=np.r_[np.linalg.solve(self.lv,diff[:len(self.lv)]),np.linalg.solve(self.lq,diff[len(self.lv):])]
        denominator=sum(np.linalg.norm(np.linalg.solve(self.lv,base[k])) for k in
                        ("advection","viscous","pressure","pressure_lifting"))
        denominator+=np.linalg.norm(np.linalg.solve(self.lq,base["continuity"]))
        denominator=max(float(denominator),1e-12)
        jd=self.jacobian(state,nu_hat,False)-other.jacobian(state,nu_hat,False)
        jnum=float(np.linalg.norm(self.whiten_matrix(jd,columns=True)))
        jden=max(sum(np.linalg.norm(self.whiten_matrix(v,columns=True)) for v in
                     self.jacobian_terms(state,nu_hat).values()),1e-12)
        return {"operator_numerator":float(np.linalg.norm(wr)),"operator_denominator":denominator,
                "operator_relative":float(np.linalg.norm(wr))/denominator,
                "jacobian_numerator":jnum,"jacobian_denominator":float(jden),"jacobian_relative":jnum/jden}


def physical_to_dimensionless(weights,lift,lift_gradient,pressure_lift_per_nu,D,U,dimension):
    """FE exporter bridge. Test values are dimensionless; test gradients separately *D."""
    if not D>0 or not U>0 or dimension not in (2,3):
        raise ValueError("positive physical scales and dimension 2/3 required")
    return {"weights":np.asarray(weights)/D**dimension,"lift":np.asarray(lift)/U,
            "lift_gradient":np.asarray(lift_gradient)*D/U,
            "pressure_lift_per_nu":np.asarray(pressure_lift_per_nu)*D/U}


def force_moments(boundary_weights,n_body,velocity_gradient,pressure_basis,lift_gradient,pressure_lift_per_nu,reference_area):
    """Dimensionless force coefficient C=2*integral(sigma*n_body)/area, separately linear.

    n_body points from solid into fluid. In 2D reference_area=1 after scaling by D;
    in 3D sphere reference_area=pi/4. Geometry mapping/admission is the caller's gate.
    """
    w=positive_weights(boundary_weights,len(n_body))
    n=np.asarray(n_body,dtype=np.float64)
    if not np.allclose(np.linalg.norm(n,axis=1),1,rtol=1e-10,atol=1e-12) or reference_area<=0:
        raise ValueError("unit body normals and positive reference area required")
    g=np.asarray(velocity_gradient)
    gl=np.asarray(lift_gradient)
    factor=2/reference_area
    return {"velocity_per_nu":factor*np.einsum("n,nijr,nj->ir",w,g+g.transpose(0,2,1,3),n),
            "pressure":-factor*np.einsum("n,nr,ni->ir",w,pressure_basis,n),
            "lifting_per_nu":factor*(np.einsum("n,nij,nj->i",w,gl+gl.transpose(0,2,1),n)-
                                      np.einsum("n,n,ni->i",w,pressure_lift_per_nu,n))}
