"""Reconstruct verified CROM checkpoints for the native predictor/projector path.

Callers must hash-verify the checkpoint and pinned native sources before passing
objects here. This does not turn method-native collocation into the common weak
PDE decoder comparison, and confers no accuracy or runtime qualification.
"""
import torch
from torch import nn
from baselines.crom_boundary import BoundaryDecoder


class AnalyticalFunction(nn.Module):
    def __init__(self, decoder):
        super().__init__();self.decoder=decoder
    def forward(self, inputs):
        return self.decoder.computeJacobianFullAnalytical(inputs)


def load_native_online(state, native_model_class, native_decoder_class, *, device='cpu', legacy_latent=None):
    """Return model, pinned native Decoder adapter, and classification metadata.

    Expanded checkpoints bind their boundary flag. Older unconstrained controls
    require an explicit legacy latent dimension; wrapped keys never imply a flag.
    Native setup initializes normalization before wrapping; strict loading follows.
    """
    cfg=state.get('config',{});bound='boundary_envelope' in cfg
    if bound:
        if type(cfg['boundary_envelope']) is not bool:raise ValueError('Boolean boundary flag required')
        boundary=cfg['boundary_envelope'];latent=cfg['latent']
    else:
        if legacy_latent is None:raise ValueError('Unbound legacy control requires explicit legacy_latent')
        boundary=False;latent=cfg.get('latent',legacy_latent)
        if latent!=legacy_latent:raise ValueError('Legacy latent override conflicts with checkpoint')
    if not isinstance(latent,int) or latent<=0:raise ValueError('Invalid latent dimension')
    wrapped_keys=any(k.startswith('decoder.native.') for k in state['model'])
    if wrapped_keys!=boundary:raise ValueError('Boundary configuration and decoder state layout disagree')
    fmt,stats=state['format'],state['statistics']
    if fmt['i_dim']!=2 or fmt['o_dim']!=1:raise ValueError('Scalar2D port required')
    if cfg.get('width',128)!=128:raise ValueError('Unsupported native decoder width')
    model=native_model_class(fmt,stats,torch.zeros((1,fmt['npoints'],3),dtype=torch.float64),
        cfg.get('base_lr',1e-4),4,latent,64,6,4,False,True,.3,30,False,
        cfg.get('lr_multipliers',[1.,.5]),cfg.get('scheduler_epochs',[100,100]),'explicit')
    model.setup('fit')
    if boundary:model.decoder=BoundaryDecoder(model.decoder,latent)
    model=model.to(device=device,dtype=torch.float64)
    model.load_state_dict(state['model'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if any(p.dtype!=torch.float64 for p in model.parameters()):raise ValueError('Float64 parameters required')
    if any(b.is_floating_point() and b.dtype!=torch.float64 for b in model.buffers()):raise ValueError('Float64 buffers required')
    decoder=native_decoder_class(model.decoder,None,AnalyticalFunction(model.decoder))
    record=dict(latent=latent,boundary_envelope=boundary,config_bound=bound,
        normalization='Native physical denormalization inside envelope' if boundary else 'Native physical denormalization',
        classification='Method-native CROM predictor/projector adapted to scalar2D Burgers; '+
            ('continuous-boundary new-PDE port, not unchanged reproduction' if boundary else 'unconstrained native-decoder control'),
        comparison_track='Method-native explicit fine-stencil collocation, distinct from common weak-PDE decoder comparison',
        pde_comparator_admitted=False)
    return model,decoder,record
