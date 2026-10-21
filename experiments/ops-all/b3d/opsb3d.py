"""Burgers 3D (homogeneous Dirichlet, unit cube, n nodes per axis) contract for the four
operator families of ops3d.py (copied unchanged from experiments/ns3d-operators).

Grid embedding (DESIGN.md §3). The FOM state is the interior field on (n-2)^3 nodes
(flat, x slowest). The operators work on the G^3 grid, G = n-1 = 64/128/256, made of the
interior nodes plus the boundary plane x_i = 0 on each axis (value exactly 0). Its
periodic extension is the Dirichlet problem's own boundary (node G == node 0 == wall), so
the circular padding of U-Net / DeepONet / Transolver and the FFT of the FNO see a
continuous field with the wall in the right place; no extra padding is needed. The
boundary plane of every output is discarded (never scored).

Features (5 channels, float64 then cast): u0 / u_scale, standardised log(nu) as one
constant channel, and the node coordinates x, y, z = i/(n-1) (the problem is not
translation invariant; the 2D Dirichlet cells also fed coordinates).
Output: 5 channels = the five evolved output times t = 0.05 ... 0.25, times out_scale.
predict() returns (B, 6, (n-2)^3) float64 with t = 0 equal to u0 exactly.
Error: ||pred_t - truth_t|| / ||truth_0|| on every interior node (the lane's metric).
"""
from __future__ import annotations

import torch

import ops3d as O

CIN, COUT = 5, 5
O.CIN, O.COUT = CIN, COUT          # ops3d.make_model reads these module globals


class Embed:
    def __init__(self, n, device='cuda'):
        self.n, self.ni, self.G = n, n - 2, n - 1
        ax = torch.arange(self.G, dtype=torch.float64, device=device) / (n - 1)
        self.coords = torch.stack(torch.meshgrid(ax, ax, ax, indexing='ij'))[None]   # (1,3,G,G,G)

    def to_grid(self, u):                       # (B, ni^3) -> (B, G, G, G), wall plane 0
        b = u.shape[0]
        g = u.new_zeros((b, self.G, self.G, self.G))
        g[:, 1:, 1:, 1:] = u.reshape(b, self.ni, self.ni, self.ni)
        return g

    def from_grid(self, g):                     # (..., G, G, G) -> (..., ni^3)
        return g[..., 1:, 1:, 1:].reshape(*g.shape[:-3], -1)


def features(u0, nu, norm, emb):
    u_scale, mu, sd, _ = norm
    b = u0.shape[0]
    G = emb.G
    lognu = ((torch.log(nu) - mu) / sd)[:, None, None, None, None].expand(b, 1, G, G, G)
    return torch.cat((emb.to_grid(u0)[:, None] / u_scale, lognu, emb.coords.expand(b, 3, G, G, G)), dim=1)


def predict(model, u0, nu, norm, emb):
    """u0 (B, ni^3) float64, nu (B,) float64 -> (B, 6, ni^3) float64."""
    out = model(features(u0, nu, norm, emb)) * norm[3]
    return torch.cat((u0[:, None], emb.from_grid(out)), dim=1)


def initial_relative(pred, target, u0):
    num = (pred - target).square().sum(-1).sqrt()
    den = u0.square().sum(-1).sqrt()
    return num / den[:, None].clamp_min(1e-300)


def make_model(config):
    return O.make_model(config)


def load_checkpoint(path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    model = make_model(ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()
    norm = tuple(torch.tensor(v, dtype=torch.float64, device='cuda') for v in ck['normalization'])
    return model, norm, ck
