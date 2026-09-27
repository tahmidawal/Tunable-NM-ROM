"""Independent NumPy/SciPy NS3D verification; does not import ns3d_fom.

Uses physical advective derivatives, separate FFT/projector code and RK4. The
production path uses rotational advection and CNAB2. Shared inputs are arrays
and the written mathematical specification, not numerical helper functions.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq


def setup(n):
    freq = fftfreq(n)*n
    wave = 2*np.pi*np.array(np.meshgrid(freq, freq, freq, indexing='ij'))
    square = (wave*wave).sum(axis=0)
    active = np.ones((n,n,n), dtype=bool)
    for d in range(3):
        active &= np.abs(wave[d]) < 2*np.pi*n/3
    active[0,0,0] = False
    return wave, square, active


def transform(u):
    return fftn(u, axes=(1,2,3), norm='forward')


def physical(v):
    return ifftn(v, axes=(1,2,3), norm='forward').real


def solenoidal(v, spec):
    wave, square, active = spec
    component = np.zeros_like(square, dtype=np.complex128)
    for d in range(3):
        component += wave[d]*v[d]
    factor = np.zeros_like(component)
    np.divide(component, square, out=factor, where=square != 0)
    result = np.empty_like(v)
    for d in range(3):
        result[d] = active*(v[d]-wave[d]*factor)
    return result


def advective(v, spec):
    v = solenoidal(v, spec)
    vel = physical(v)
    result = np.zeros_like(vel)
    for d in range(3):
        derivative = physical(1j*spec[0][d]*v)
        result -= vel[d]*derivative
    return solenoidal(transform(result), spec)


def solve(initial, viscosity, dt, nsteps, out_every):
    spec = setup(initial.shape[-1])
    v = solenoidal(transform(initial), spec)
    result = [physical(v)]
    def fun(q):
        return advective(q, spec)-viscosity*spec[1]*q
    for step in range(nsteps):
        a = fun(v)
        b = fun(v+.5*dt*a)
        c = fun(v+.5*dt*b)
        d = fun(v+dt*c)
        v = solenoidal(v+dt*(a+2*b+2*c+d)/6, spec)
        if (step+1) % out_every == 0:
            result.append(physical(v))
    return np.stack(result)


def manufactured(n, t):
    """Analytic u, du/dt, Laplacian and advective derivative for a mixed field."""
    xx = np.arange(n)/n
    x,y,z = np.meshgrid(xx, xx, xx, indexing='ij')
    s = 2*np.pi
    A = np.stack((np.sin(s*y), np.sin(s*z), np.sin(s*x)))
    B = np.stack((np.cos(s*(y+z)), np.cos(s*(x+z)), np.cos(s*(x+y))))
    a, b = 1+.2*np.sin(2*t), .4*np.cos(1.3*t)
    at, bt = .4*np.cos(2*t), -.52*np.sin(1.3*t)
    u = a*A+b*B
    lap = -s*s*(a*A+2*b*B)
    deriv = np.zeros((3,3,n,n,n))  # direction, velocity component
    deriv[1,0] = a*s*np.cos(s*y)
    deriv[2,1] = a*s*np.cos(s*z)
    deriv[0,2] = a*s*np.cos(s*x)
    for d, comp, angle in ((1,0,y+z),(2,0,y+z),(0,1,x+z),
                           (2,1,x+z),(0,2,x+y),(1,2,x+y)):
        deriv[d,comp] += -b*s*np.sin(s*angle)
    adv = sum(u[d]*deriv[d] for d in range(3))
    return u, at*A+bt*B, lap, adv
