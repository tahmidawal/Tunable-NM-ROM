"""Synthetic checkpoint/native-adapter fixture only; no trained model evaluation."""
import importlib.util,hashlib,json,io
from pathlib import Path
import torch
from baselines.crom_boundary import BoundaryDecoder
from baselines.crom_loader import load_native_online
from baselines.crom_scalar2d import ScalarBurgers2D

torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
path=Path(__file__).parent/'results/b6native/output/sources/crom_online/online/Decoder.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5cbc11c2ea15a206a3de5efc676e78b2f6267979e91b5728133c4435b9a7c385'
spec=importlib.util.spec_from_file_location('fixture_native_decoder',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Affine(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.2,-.1,.3,.4]]));self.bias=torch.nn.Parameter(torch.tensor([.2]))
        self.register_buffer('mean',torch.tensor([.37]));self.register_buffer('std',torch.tensor([.23]))
    def forward(self,x):return self.mean+self.std*(x@self.weight.T+self.bias)
    def computeJacobianFullAnalytical(self,x):return (self.std[:,None]*self.weight)[None].expand(len(x),-1,-1),self(x)
class SyntheticModel(torch.nn.Module):
    def __init__(self,fmt,stats,*args):super().__init__();self.decoder=Affine();self.stats=stats
    def setup(self,stage):
        self.decoder.mean.copy_(torch.as_tensor(self.stats['mean_q']));self.decoder.std.copy_(torch.as_tensor(self.stats['std_q']))
fmt=dict(i_dim=2,o_dim=1,npoints=8);stats=dict(mean_q=[.37],std_q=[.23]);source=SyntheticModel(fmt,stats)
source.decoder=BoundaryDecoder(source.decoder,2)
state=dict(model=source.state_dict(),config=dict(latent=2,boundary_envelope=True,width=128),format=fmt,statistics=stats)
buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0);loaded=torch.load(buffer,weights_only=False)
model,decoder,record=load_native_online(loaded,SyntheticModel,module.Decoder)
z=torch.tensor([.6,-.2]);errors=[];rhs_errors=[];rhs_j_errors=[];strip_sizes=[]
for L in [4096,8192,16384]:
 h=1/L
 points=torch.tensor([[0.,.4],[h,.4],[2*h,.4],[1.,.4],[1-h,.4],[1-2*h,.4],
                      [.4,0.],[.4,h],[.4,2*h],[.4,1.],[.4,1-h],[.4,1-2*h]])
 inp=torch.cat([z[None].expand(len(points),-1),points],1)
 raw=.37+.23*(.2*z[0]-.1*z[1]+.3*points[:,0]+.4*points[:,1]+.2)
 B=16*points[:,0]*(1-points[:,0])*points[:,1]*(1-points[:,1]);expected=B*raw
 y=decoder.forward(inp)[:,0];assert torch.equal(y[[0,3,6,9]],torch.zeros(4))
 assert float((y-expected).abs().max())<1e-15
 strip_sizes.append(float(y[[1,4,7,10]].abs().max()))
 J,_=model.decoder.computeJacobianFullAnalytical(inp)
 auto=torch.func.vmap(torch.func.jacrev(lambda x:model.decoder(x[None])[0]))(inp)
 errors.append(float((J-auto).abs().max()))
 assert torch.equal(decoder.jacobianPart(inp,2,'fir'),J[:,:,:2])
 assert torch.equal(decoder.jacobianPart(inp,2,'sec'),J[:,:,2:])
 nodes=torch.tensor([[1,L//2],[2,L//2],[L-1,L//2],[L-2,L//2],[L//2,1],[L//2,2]],dtype=torch.int64)
 initial=lambda p:model.decoder(torch.cat([z[None].expand(len(p),-1),p],1))
 problem=ScalarBurgers2D(decoder,2,initial,nodes,intervals=L)
 # Independent polynomial stencil; no full L-by-L grid is allocated.
 stencil=nodes[:,None,:]+torch.tensor([[0,0],[-1,0],[1,0],[0,-1],[0,1]])[None]
 xy=stencil.to(torch.float64)/L;x,y=xy[...,0],xy[...,1]
 b=16*x*(1-x)*y*(1-y);r=.37+.23*(.2*z[0]-.1*z[1]+.3*x+.4*y+.2);u=b*r
 c,xm,xp,ym,yp=u.unbind(1);du=L*((c-xm)+(c-ym));lap=L*L*(xm+xp+ym+yp-4*c)
 rhs=.005*lap-c*du
 actual=problem.getResidualSample(z)[:,0];rhs_errors.append(float((actual-rhs).abs().max()))
 # Exact discrete code product rule, independently from adapter implementation.
 j=b[...,None]*torch.tensor([.23*.2,-.23*.1]);jc,jxm,jxp,jym,jyp=j.unbind(1)
 expected_j=.005*L*L*(jxm+jxp+jym+jyp-4*jc)-jc*du[:,None]-c[:,None]*L*(2*jc-jxm-jym)
 rhs_j_errors.append(float((problem.rhs_code_jacobian(z)-expected_j).abs().max()))
assert max(errors)<1e-13 and max(rhs_errors)<1e-10 and max(rhs_j_errors)<1e-10
assert strip_sizes[2]<.26*strip_sizes[0] and strip_sizes[1]<.51*strip_sizes[0]
# Native unconstrained control stays physical/unwrapped; mismatched flags reject.
control=SyntheticModel(fmt,stats);cs=dict(state,model=control.state_dict(),config=dict(latent=2,boundary_envelope=False))
cm,cd,cr=load_native_online(cs,SyntheticModel,module.Decoder)
assert not isinstance(cm.decoder,BoundaryDecoder) and not cr['boundary_envelope']
try:load_native_online(dict(state,config=dict(latent=2,boundary_envelope=False)),SyntheticModel,module.Decoder)
except ValueError:pass
else:raise AssertionError('Mismatched boundary config accepted')
print(json.dumps(dict(loaded_synthetic_checkpoint=True,physical_normalization=True,exact_boundary=True,
    strip_magnitudes=strip_sizes,jacobian_errors=errors,fine_stencil_rhs_errors=rhs_errors,
    fine_stencil_rhs_code_jacobian_errors=rhs_j_errors,native_control_preserved=True,incompatible_flag_rejected=True)))
