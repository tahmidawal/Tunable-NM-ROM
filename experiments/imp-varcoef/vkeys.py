"""imp-varcoef competitor keys, programs, prepared inputs, validity (DESIGN §6, A1.12, A2.22, A3.14).

Keys:
  psi|N<N>|<dt>|<SBDF2|CNAB2>|<tol>[|Q<Q>]     spectral IMEX + warm CG
  etd|N<N>|<dt>|<max|mid>[|Q<Q>]               ETDRK4 constant-coefficient split
  fdmg|L<L>|<dt>|<MG|DST>|<tol>                FD + SBDF2 + PCG(MG or DST)
  ccv|L<L>|<dt>|t<level>                       E0 scheme (CN-R Newton-BiCGStab) at jcp-fom2d tolerance level
Timing keys append '@C<contrast>'; 'KERNEL|' / 'SETUP|' prefixes select the A3.14 decomposition programs.
Every program maps (prepared IC x0, kappa parameters kp) -> (fields (6, T+1, T+1), stats).
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import vcoef as VC
import vspec as VS
import vfd as VF
import refcore as FC
import fomrun2 as FR

UMAX_GUARD = 10.0


def split_key(k):
    pre = None
    if k.startswith(('KERNEL|', 'SETUP|')):
        pre, k = k.split('|', 1)
    C = None
    if '@C' in k:
        k, c = k.split('@C')
        C = float(c)
    return pre, k, C


def family(k):
    return split_key(k)[1].split('|')[0].upper()


def parse(k):
    p = split_key(k)[1].split('|')
    f = p[0]
    if f == 'psi':
        Q = int(p[5][1:]) if len(p) > 5 else None
        return dict(f=f, N=int(p[1][1:]), dt=float(p[2]), scheme=p[3], tol=float(p[4]), Q=Q)
    if f == 'etd':
        Q = int(p[4][1:]) if len(p) > 4 else None
        return dict(f=f, N=int(p[1][1:]), dt=float(p[2]), kbar=p[3], Q=Q)
    if f == 'fdmg':
        return dict(f=f, L=int(p[1][1:]), dt=float(p[2]), prec=p[3], tol=float(p[4]))
    if f == 'ccv':
        return dict(f=f, L=int(p[1][1:]), dt=float(p[2]), level=p[3][1:])
    raise KeyError(k)


def key(sp):
    f = sp['f']
    if f == 'psi':
        return f"psi|N{sp['N']}|{sp['dt']:g}|{sp['scheme']}|{sp['tol']:g}" + (f"|Q{sp['Q']}" if sp.get('Q') else '')
    if f == 'etd':
        return f"etd|N{sp['N']}|{sp['dt']:g}|{sp['kbar']}" + (f"|Q{sp['Q']}" if sp.get('Q') else '')
    if f == 'fdmg':
        return f"fdmg|L{sp['L']}|{sp['dt']:g}|{sp['prec']}|{sp['tol']:g}"
    return f"ccv|L{sp['L']}|{sp['dt']:g}|t{sp['level']}"


def resolution(sp):
    return sp.get('N') or sp.get('L')


class CCvProgram:
    """CCv in the Program interface: query = solve (AOT) then the output map (two dispatches in one timed call, as
    jcp-fom2d's FD); setup = face coefficients; kernel = query (CCv's setup is inside refcore's solve)."""

    def __init__(self, sp, out):
        self.q = VF.make_ccv(sp['L'], sp['dt'], sp['level'], out)
        self.name = key(sp)
        L = sp['L']
        self.setup_only = jax.jit(lambda kp: VF.faces(kp, L))

    def query(self, x0, kp):
        return self.q(x0, kp)

    kernel = None

    def __call__(self, x0, kp):
        return self.query(x0, kp)


def program(sp, out='lattice'):
    f = sp['f']
    if f == 'psi':
        return VS.make_psi(sp['N'], sp['dt'], sp['scheme'], sp['tol'], out=out, Q=sp.get('Q'))
    if f == 'etd':
        return VS.make_etd(sp['N'], sp['dt'], sp['kbar'], out=out, Q=sp.get('Q'))
    if f == 'fdmg':
        return VF.make_fdmg(sp['L'], sp['dt'], sp['tol'], sp['prec'], out=out)
    if f == 'ccv':
        return CCvProgram(sp, out)
    raise KeyError(f)


def prepare(sp, case):
    """Prepared IC (outside every timer, as C2): spectral sine coefficients or FD interior nodal samples."""
    t = time.perf_counter()
    x = VS.ic(case['desc'], sp['N']) if sp['f'] in ('psi', 'etd') else VF.fd_ic(case['desc'], sp['L'])
    x = jax.block_until_ready(jnp.asarray(x))
    return x, time.perf_counter() - t


def validity(sp, F, st):
    """A1.12 / A2.22 run validity from the program's own stats and outputs."""
    F = np.asarray(F)
    finite = bool(np.all(np.isfinite(F)))
    if sp['f'] == 'ccv':
        r = st['raw']
        ns = int(st['nexp'])
        res = np.asarray(r['res'])[:ns]
        complete = int(r['stages']) == ns
        res_ok = bool(len(res) == ns and np.all(np.isfinite(res)) and np.all(res <= 2 * st['ntol']))
        umax = float(r['umax'])
        ev = dict(stages=int(r['stages']), expected=ns, worst_res=float(np.nanmax(res)) if len(res) else None,
                  newton_total=int(np.asarray(r['nwt'])[:ns].sum()), umax=umax)
    else:
        st = jax.device_get(st)
        complete = int(st['nst']) == int(st['nst_expected'])
        res_ok = int(st['nbad']) == 0
        umax = float(st['umax'])
        ev = dict(stages=int(st['nst']), expected=int(st['nst_expected']), nbad=int(st['nbad']),
                  it_sum=int(st['it_sum']), it_max=int(st['it_max']), worst_res=float(st['res_max']), umax=umax)
    ok = bool(finite and complete and res_ok and np.isfinite(umax) and umax <= UMAX_GUARD)
    ev.update(finite=finite, complete=bool(complete), res_ok=bool(res_ok), valid=ok)
    return ok, ev


def lat_err(Flat, R, n0):
    """E0 metric: max_{k>=1} |F_k - R_k|_2 / |R(t0)|_2 on the 257^2 lattice."""
    Flat, R = np.asarray(Flat, np.float64), np.asarray(R, np.float64)
    return float(max(np.linalg.norm((Flat[k] - R[k]).ravel()) for k in range(1, 6)) / n0)
