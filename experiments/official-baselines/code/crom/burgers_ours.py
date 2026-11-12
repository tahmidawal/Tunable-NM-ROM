"""Our Burgers 2D problem for the official CROM online code (peterchencyc/CROM_online_deployment @ ccb8a66).

`Burger` follows online/Experiments/Burger.py method by method (the class keeps that name because the released
SamplePoint.initialize dispatches on the class name; 'full' sampling as in the released Burgers example). What differs
is the physics: the residual is our discretisation (burgers_fom.spatial, in torch: sign-upwind advection, five-point
Laplacian, 256 mesh, zero Dirichlet boundary), and the boundary value is imposed on the decoded state (the released
class imposes its inflow value with `sq[0,0,0] = self.mu1`); the boundary rows of the decoder Jacobian are zeroed to
match.

Explicit arm (CROM-E, as released): getResidual returns du/dt = -(adv(q) - nu*lap(q)); online loop target q + dt*f(q).
Implicit arm (CROM-I, connected): `implicit=True` makes getJacobianSample return the Jacobian of the backward-Euler
residual r(q) = q - q_prev + dt*(adv(q) - nu*lap(q)) w.r.t. the latent (chain rule through the stencil), and
`be_residual` is used by NonlinearSolverBE.computeLossAndR.
"""
import os, sys
import numpy as np
import torch

L = 256


def node_coords():
    x = np.arange(L + 1) / L
    return np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)


def _pad(u):          # u [..., L-1, L-1] interior -> [..., L+1, L+1] with zero boundary
    return torch.nn.functional.pad(u, (1, 1, 1, 1))


def spatial(u):
    """burgers_fom.spatial in torch, batched over leading dims. u [..., L-1, L-1] -> adv, lap [..., L-1, L-1]."""
    p = _pad(u)
    c, xm, xp, ym, yp = p[..., 1:-1, 1:-1], p[..., :-2, 1:-1], p[..., 2:, 1:-1], p[..., 1:-1, :-2], p[..., 1:-1, 2:]
    pos = c > 0
    adv = c * L * (torch.where(pos, c - xm, xp - c) + torch.where(pos, c - ym, yp - c))
    lap = L ** 2 * (xm + xp + ym + yp - 4 * c)
    return adv, lap


def spatial_lin(u, v):
    """Directional derivative of spatial(u) along v (v [k, L-1, L-1]); the sign switch is held fixed (as jax.jvp of where)."""
    p, w = _pad(u), _pad(v)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    dc, dxm, dxp, dym, dyp = w[:, 1:-1, 1:-1], w[:, :-2, 1:-1], w[:, 2:, 1:-1], w[:, 1:-1, :-2], w[:, 1:-1, 2:]
    pos = c > 0
    g = torch.where(pos, c - xm, xp - c) + torch.where(pos, c - ym, yp - c)
    dg = torch.where(pos, dc - dxm, dxp - dc) + torch.where(pos, dc - dym, dyp - dc)
    dadv = dc * L * g + c * L * dg
    dlap = L ** 2 * (dxm + dxp + dym + dyp - 4 * dc)
    return dadv, dlap


class Burger(object):
    def __init__(self, u0, nu, dt, dtype, device, implicit=False):
        self.dtype, self.device = dtype, device
        self.nu, self.dt, self.implicit = nu, dt, implicit
        self.x = torch.as_tensor(node_coords(), dtype=dtype, device=device).view(1, -1, 2)
        b = np.zeros((L + 1, L + 1), bool); b[[0, -1], :] = True; b[:, [0, -1]] = True
        self.bmask = torch.as_tensor(b.reshape(-1), device=device)
        self.q0_gt = torch.as_tensor(u0, dtype=dtype, device=device).view(1, -1, 1)
        self.q_prev_interior = None

    def getPhysicalInitialCondition(self):
        return self.q0_gt

    # --- as Experiments/Burger.py ------------------------------------------------------------------------------
    def updateState(self, xhat, decoder, index=None, vhat=None, xhat_prev=None, save_q=False):
        if index is None:
            index = np.arange(self.x.size(1))
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:, index, :]), 2)
        x = x.view(x.size(0) * x.size(1), x.size(2))
        q = decoder.forward(x)
        sq = q.view(1, q.size(0), q.size(1))
        sq[0, self.bmask[index], 0] = 0.          # ADAPTED: zero Dirichlet boundary (released: sq[0,0,0] = self.mu1)
        if save_q:
            self.q = sq
        return sq.detach()

    def interior(self, q):
        return q.view(L + 1, L + 1)[1:-1, 1:-1]

    def getResidual(self, xhat, decoder, index):
        q = self.updateState(xhat, decoder).view(1, -1, 1)
        adv, lap = spatial(self.interior(q))
        f = torch.zeros(L + 1, L + 1, dtype=q.dtype, device=q.device)
        f[1:-1, 1:-1] = -(adv - self.nu * lap)     # ADAPTED: our spatial operator (released: 1D Burgers flux)
        return f.view(1, -1, 1)

    def getJacobian(self, xhat, decoder, index=None):
        if index is None:
            index = np.arange(self.x.size(1))
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:, index, :]), 2)
        x = x.view(x.size(0) * x.size(1), x.size(2))
        jacobian_part = decoder.jacobianPart(x, lbllength, 'fir')
        jacobian_part = jacobian_part.view(-1, jacobian_part.size(2))
        jacobian_part[self.bmask[index]] = 0.     # ADAPTED: boundary values are imposed, not decoded
        return jacobian_part

    def getResidualSample(self, xhat, decoder, sample_point, step):
        return self.getResidual(xhat, decoder, sample_point.indices)

    def getJacobianSample(self, xhat, decoder, sample_point):
        J = self.getJacobian(xhat, decoder, sample_point.indices)
        if not self.implicit:
            return J
        # CROM-I: Jacobian of the backward-Euler residual (interior rows) = (I + dt (dadv - nu dlap)) J
        k = J.size(1)
        q = self.updateState(xhat, decoder, sample_point.indices).view(-1)
        V = J.t().reshape(k, L + 1, L + 1)[:, 1:-1, 1:-1]
        dadv, dlap = spatial_lin(self.interior(q), V)
        R = V + self.dt * (dadv - self.nu * dlap)
        return R.reshape(k, -1).t().contiguous()

    def be_residual(self, q):
        qi = self.interior(q)
        adv, lap = spatial(qi)
        return (qi - self.q_prev_interior + self.dt * (adv - self.nu * lap)).reshape(-1, 1)

    def updateStateSample(self, xhat, decoder, sample_point, vhat=None):
        self.q_sample = self.updateState(xhat, decoder, sample_point.indices, save_q=False)
        return self.q_sample

    def updateStateSampleAll(self, xhat, decoder, sample_point, v_hat=None):
        q_all = self.updateState(xhat, decoder, None, save_q=True)
        return q_all

    def updateSampleIndicesWithProblemSpeficifInfo(self, sample_point):
        pass

    def callBack(self, step, xhat, decoder, vhat=None, xhat_prev=None):
        return False
