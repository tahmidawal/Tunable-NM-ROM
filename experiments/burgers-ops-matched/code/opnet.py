"""Networks and the operator contract (float32 networks), shared by train.py and evaluate.py.

Contract = the paper's (ops/model.py @ 822f27166): features = [normalised u0, normalised nu broadcast, x, y];
the network emits five evolved fields, multiplied by the target scale and masked to the zero-Dirichlet boundary.
Network classes: neuralop FNO (as ops/model.py's float32 rung) and families.UNet2d / families.Transolver2d (verbatim copy).
"""
import torch
import families

ARMS = {
    'F0': dict(family='fno', width=64, modes=32, layers=4, lr=1e-3),
    'F1': dict(family='fno', width=64, modes=48, layers=4, lr=1e-3),
    'F2': dict(family='fno', width=96, modes=32, layers=4, lr=1e-3),
    'U0': dict(family='unet', base=48, groups=8, lr=3e-4),
    'U1': dict(family='unet', base=64, groups=8, lr=3e-4),
    'U2': dict(family='unet', base=48, groups=8, lr=1e-3),
    'T0': dict(family='transolver', dim=128, layers=8, heads=8, slices=64, mlp_ratio=2, patch=4, ref=8, lr=3e-4,
               warmup_steps=5000),
}
VALIDATION = {128: 16, 576: 64, 4608: 256}


def configure():
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


def make(cfg):
    cin, cout = 4, 5
    if cfg['family'] == 'fno':
        from neuralop.models import FNO
        net = FNO(n_modes=(cfg['modes'],) * 2, in_channels=cin, out_channels=cout, hidden_channels=cfg['width'],
                  n_layers=cfg['layers'], positional_embedding=None, fno_block_precision='full', norm=None)
    elif cfg['family'] == 'unet':
        net = families.UNet2d(cin, cout, base=cfg['base'], groups=cfg['groups'])
    elif cfg['family'] == 'transolver':
        net = families.Transolver2d(cin, cout, dim=cfg['dim'], layers=cfg['layers'], heads=cfg['heads'],
                                    slices=cfg['slices'], mlp_ratio=cfg['mlp_ratio'], patch=cfg['patch'], ref=cfg['ref'])
    else:
        raise ValueError(cfg['family'])
    return net.cuda()


def real_parameters(net):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())


def features(u0, nu, norm):
    """u0 (B,H,W), nu (B,), any float dtype -> (B,4,H,W) in u0's dtype."""
    b, h, w = u0.shape
    xx = torch.linspace(0, 1, h, device=u0.device, dtype=u0.dtype)
    yy = torch.linspace(0, 1, w, device=u0.device, dtype=u0.dtype)
    gx, gy = torch.meshgrid(xx, yy, indexing='ij')
    f = (u0 - norm['u_mean']) / norm['u_std']
    p = ((nu - norm['nu_mean']) / norm['nu_std'])[:, None, None].expand(b, h, w)
    return torch.stack((f, p, gx.expand(b, h, w), gy.expand(b, h, w)), dim=1)


def predict(net, u0, nu, norm):
    """Five evolved fields (B,5,H,W), in the dtype of u0; the network itself runs in float32."""
    out = net(features(u0, nu, norm).to(torch.float32)).to(u0.dtype) * norm['scale']
    out[..., 0, :] = 0
    out[..., -1, :] = 0
    out[..., :, 0] = 0
    out[..., :, -1] = 0
    return out


def relative_errors(pred, target, u0):
    """(B,5): ||pred(t_k) - target(t_k)||_2 / ||u0||_2."""
    num = (pred - target).square().sum(dim=(-2, -1)).sqrt()
    return num / u0.square().sum(dim=(-2, -1)).sqrt()[:, None]
