"""Continuous homogeneous-Dirichlet port of the physical CROM decoder.

This is a new-PDE boundary-conforming port, not the unchanged native network.
The wrapped decoder must already return physical, denormalized values. There
is no target division by the envelope and no endpoint-only output masking.
"""
import torch
from torch import nn


class BoundaryDecoder(nn.Module):
    def __init__(self, decoder, latent):
        super().__init__()
        if not isinstance(latent,int) or latent<=0:raise ValueError('Positive latent dimension required')
        self.native=decoder;self.latent=latent

    def envelope(self, inputs):
        if inputs.ndim!=2 or inputs.shape[1]!=self.latent+2:
            raise ValueError('Expected batch of [latent code, physical x, physical y]')
        x,y=inputs[:,-2],inputs[:,-1]
        b=16*x*(1-x)*y*(1-y)
        db=torch.stack([16*(1-2*x)*y*(1-y),16*x*(1-x)*(1-2*y)],dim=-1)
        return b[:,None],db

    def forward(self, inputs):
        b,_=self.envelope(inputs)
        raw=self.native(inputs)
        if raw.shape!=(len(inputs),1):raise ValueError('Scalar physical decoder required')
        return b*raw

    def computeJacobianFullAnalytical(self, inputs):
        b,db=self.envelope(inputs)
        jac,raw=self.native.computeJacobianFullAnalytical(inputs)
        if jac.shape!=(len(inputs),1,self.latent+2) or raw.shape!=(len(inputs),1):
            raise ValueError('Expected scalar native decoder and full physical-input Jacobian')
        gradient=torch.cat([torch.zeros_like(inputs[:,:self.latent]),db],dim=-1)
        return b[:,:,None]*jac+raw[:,:,None]*gradient[:,None,:],b*raw
