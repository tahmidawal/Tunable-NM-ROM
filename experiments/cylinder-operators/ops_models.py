"""Self-contained PyTorch models for the three arms (PROTOCOL.md "Arms") and the surrogate wrapper
that turns (first state, Re) into the 681 saves of a window in the native representation."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NX, NY = 1024, 192
LX, LY = 2.2, 0.41
CENTER, RADIUS = (0.2, 0.2), 0.05


def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- arm 1: Transolver-style node stepper
class PhysAttn(nn.Module):
    def __init__(self, C, H, M):
        super().__init__()
        self.H, self.dh = H, C // H
        self.in_x = nn.Linear(C, C); self.in_f = nn.Linear(C, C)
        self.slice = nn.Linear(self.dh, M)
        self.temp = nn.Parameter(torch.full((1, H, 1, 1), 0.5))
        self.q = nn.Linear(self.dh, self.dh, bias=False); self.k = nn.Linear(self.dh, self.dh, bias=False)
        self.v = nn.Linear(self.dh, self.dh, bias=False)
        self.out = nn.Linear(C, C)

    def forward(self, x):
        B, N, C = x.shape
        fx = self.in_f(x).view(B, N, self.H, self.dh).transpose(1, 2)
        xm = self.in_x(x).view(B, N, self.H, self.dh).transpose(1, 2)
        w = torch.softmax(self.slice(xm) / self.temp.clamp(0.1, 5.0), dim=-1)          # B,H,N,M
        tok = torch.einsum('bhnd,bhnm->bhmd', fx, w) / (w.sum(2)[..., None] + 1e-5)
        tok = F.scaled_dot_product_attention(self.q(tok), self.k(tok), self.v(tok))
        out = torch.einsum('bhmd,bhnm->bhnd', tok, w).transpose(1, 2).reshape(B, N, C)
        return self.out(out)


class Transolver(nn.Module):
    def __init__(self, n_in, n_out, C=128, L=6, H=8, M=32):
        super().__init__()
        self.embed = mlp([n_in, 2 * C, C])
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(nn.ModuleList([nn.LayerNorm(C), PhysAttn(C, H, M), nn.LayerNorm(C), mlp([C, 2 * C, C])]))
        self.head = nn.Sequential(nn.LayerNorm(C), mlp([C, C, n_out]))

    def forward(self, x):
        x = self.embed(x)
        for n1, at, n2, ff in self.blocks:
            x = x + at(n1(x))
            x = x + ff(n2(x))
        return self.head(x)


def node_static(xy):
    """Static per-node features: scaled coordinates, distance to the cylinder, a few Fourier features."""
    xy = torch.as_tensor(xy, dtype=torch.float32)
    x = xy[:, 0] / LX * 2 - 1; y = xy[:, 1] / LY * 2 - 1
    r = torch.hypot(xy[:, 0] - CENTER[0], xy[:, 1] - CENTER[1]) - RADIUS
    feats = [x, y, torch.tanh(r / 0.1), torch.exp(-r / 0.02)]
    for f in (1, 2, 4, 8, 16):
        feats += [torch.sin(math.pi * f * x), torch.cos(math.pi * f * x)]
    for f in (1, 2, 4):
        feats += [torch.sin(math.pi * f * y), torch.cos(math.pi * f * y)]
    return torch.stack(feats, dim=1)


# ---------------------------------------------------------------- arm 2: FNO / U-Net grid stepper
class SpecConv(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2, self.cout = m1, m2, cout
        s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')


class FNO(nn.Module):
    def __init__(self, n_in, n_out, W=40, L=4, m1=16, m2=64, pad=(16, 32)):
        super().__init__()
        self.pad = pad
        self.lift = nn.Sequential(nn.Conv2d(n_in, W, 1), nn.GELU(), nn.Conv2d(W, W, 1))
        self.spec = nn.ModuleList([SpecConv(W, W, m1, m2) for _ in range(L)])
        self.mix = nn.ModuleList([nn.Sequential(nn.Conv2d(W, W, 1), nn.GELU(), nn.Conv2d(W, W, 1)) for _ in range(L)])
        self.skip = nn.ModuleList([nn.Conv2d(W, W, 1) for _ in range(L)])
        self.proj = nn.Sequential(nn.Conv2d(W, 4 * W, 1), nn.GELU(), nn.Conv2d(4 * W, n_out, 1))

    def forward(self, x):
        x = self.lift(x)
        py, px = self.pad
        x = F.pad(x, (0, px, 0, py))
        for i, (sp, mx, sk) in enumerate(zip(self.spec, self.mix, self.skip)):
            y = mx(sp(x)) + sk(x)
            x = F.gelu(y) if i < len(self.spec) - 1 else y
        x = x[:, :, :x.shape[2] - py, :x.shape[3] - px]
        return self.proj(x)


class UNet(nn.Module):
    def __init__(self, n_in, n_out, base=32, levels=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(levels)]

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU())
        self.down = nn.ModuleList([block(n_in if i == 0 else ch[i - 1], ch[i]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(levels - 1, 0, -1)])
        self.dec = nn.ModuleList([block(2 * ch[i - 1], ch[i - 1]) for i in range(levels - 1, 0, -1)])
        self.out = nn.Conv2d(ch[0], n_out, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = d(torch.cat([u(x), s], dim=1))
        return self.out(x)


def grid_static(fluid):
    xc = (torch.arange(NX) + 0.5) * LX / NX; yc = (torch.arange(NY) + 0.5) * LY / NY
    Y, X = torch.meshgrid(yc, xc, indexing='ij')
    r = torch.hypot(X - CENTER[0], Y - CENTER[1]) - RADIUS
    m = torch.as_tensor(fluid, dtype=torch.float32)
    return torch.stack([X / LX * 2 - 1, Y / LY * 2 - 1, m, torch.tanh(r / 0.1), torch.exp(-r.clamp(min=0) / 0.02)])


# ---------------------------------------------------------------- arm 3: DeepONet
class DeepONet(nn.Module):
    def __init__(self, n_sens, p=128, width=512, depth=4, nf=96, nft=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        sig = torch.tensor([10.0, 40.0, 120.0]).repeat_interleave(nf // 3)
        self.register_buffer('Bxy', torch.randn(nf, 2, generator=g) * sig[:, None])
        self.register_buffer('Bt', torch.randn(nft, generator=g) * 20.0)
        self.p = p
        self.branch = mlp([3 * n_sens + 1] + [width] * 3 + [3 * p])
        self.trunk = mlp([2 * nf + 2 * nft + 3] + [width] * depth + [p], last_act=True)
        self.bias = nn.Parameter(torch.zeros(3))

    def coeffs(self, sens, re):                       # (B, S, 3), (B,) -> (B, 3, p)
        return self.branch(torch.cat([sens.flatten(1), re[:, None]], dim=1)).view(-1, 3, self.p)

    def basis(self, xy, tau):                         # (..., 2), (...,) -> (..., p)
        a = xy @ self.Bxy.T; b = tau[..., None] * self.Bt
        f = torch.cat([torch.sin(a), torch.cos(a), torch.sin(b), torch.cos(b), xy / LX * 2 - 1, tau[..., None] / 1.7], dim=-1)
        return self.trunk(f)

    def forward(self, sens, re, xy, tau):             # xy (B, P, 2), tau (B,) -> (B, P, 3)
        c = self.coeffs(sens, re)
        t = self.basis(xy, tau[:, None].expand(-1, xy.shape[1]))
        return torch.einsum('bcp,bqp->bqc', c, t) / math.sqrt(self.p) + self.bias


# ---------------------------------------------------------------- surrogate wrapper
def build(arm, cfg, meta):
    K = cfg.get('K', 1)
    if arm == 'node':
        return Transolver(3 + 1 + meta['n_static'], 3 * K, cfg['C'], cfg['L'], cfg['H'], cfg['M'])
    if arm == 'grid':
        if cfg['net'] == 'fno':
            return FNO(3 + 1 + 5, 3 * K, cfg['W'], cfg['L'], cfg['m1'], cfg['m2'])
        return UNet(3 + 1 + 5, 3 * K, cfg['base'], cfg['levels'])
    if arm == 'don':
        return DeepONet(meta['n_sens'], cfg['p'], cfg['width'], cfg['depth'])
    raise ValueError(arm)


class Surrogate:
    """Holds model + normalisation + static inputs; window(x0, Re) -> (saves, ...) in physical units."""

    def __init__(self, arm, cfg, meta, stats, static, device):
        self.arm, self.cfg, self.meta, self.dev = arm, cfg, meta, device
        self.model = build(arm, cfg, meta).to(device)
        self.mean = torch.as_tensor(stats['mean'], dtype=torch.float32, device=device)
        self.std = torch.as_tensor(stats['std'], dtype=torch.float32, device=device)
        self.dstd = torch.as_tensor(stats['dstd'], dtype=torch.float32, device=device) if 'dstd' in stats else None  # (K, 3)
        self.stats = stats
        self.static = {k: torch.as_tensor(v).to(device) for k, v in static.items()}
        self.K = cfg.get('K', 1)

    def re_feat(self, Re):
        return (torch.as_tensor(Re, dtype=torch.float32, device=self.dev) - 100.0) / 10.0

    # ---- shapes: node (B, N, 3); grid (B, 3, NY, NX)
    def norm(self, x):
        return (x - self.mean) / self.std if self.arm != 'grid' else (x - self.mean[:, None, None]) / self.std[:, None, None]

    def step_delta(self, xn, re, node_idx=None):
        """Normalised state -> normalised deltas for the next K saves: node (B, N, K, 3); grid (B, K, 3, NY, NX)."""
        B = xn.shape[0]
        if self.arm == 'node':
            st = self.static['feat'] if node_idx is None else self.static['feat'][node_idx]
            inp = torch.cat([xn, re.view(B, 1, 1).expand(B, xn.shape[1], 1), st[None].expand(B, -1, -1)], dim=-1)
            return self.model(inp).view(B, xn.shape[1], self.K, 3)
        st = self.static['feat']
        inp = torch.cat([xn, re.view(B, 1, 1, 1).expand(B, 1, NY, NX), st[None].expand(B, -1, -1, -1)], dim=1)
        return self.model(inp).view(B, self.K, 3, NY, NX)

    @torch.no_grad()
    def window(self, x0, Re, n_saves):
        self.model.eval()
        x0 = x0.to(self.dev, torch.float32)
        if self.arm == 'don':
            return self._window_don(x0, Re, n_saves)
        re = self.re_feat([Re])
        out = [x0]
        x = x0
        while len(out) < n_saves:
            d = self.step_delta(self.norm(x[None]), re)[0]
            if self.arm == 'node':
                nxt = x[:, None, :] + d * self.dstd[None] * self.std          # (N, K, 3)
                nxt = nxt.permute(1, 0, 2)
            else:
                nxt = x[None] + d * (self.dstd * self.std)[:, :, None, None]  # (K, 3, NY, NX)
                nxt[:, :2] = nxt[:, :2] * self.static['fluid']
            out += list(nxt)
            x = nxt[-1]
        return torch.stack(out[:n_saves])

    def _window_don(self, x0, Re, n_saves):
        sens = self.norm(x0[self.static['sens']])[None]
        c = self.model.coeffs(sens, self.re_feat([Re]))                         # (1, 3, p)
        xy = self.static['xy']; dt = self.meta['dt']
        out = []
        for j in range(n_saves):
            t = self.model.basis(xy, torch.full((xy.shape[0],), j * dt, device=self.dev))
            out.append((t @ c[0].T / math.sqrt(self.model.p) + self.model.bias) * self.std + self.mean)
        return torch.stack(out)

    def checkpoint(self):
        return {'arm': self.arm, 'cfg': self.cfg, 'meta': self.meta, 'stats': self.stats, 'state': self.model.state_dict()}


def n_params(model):
    return int(sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))
