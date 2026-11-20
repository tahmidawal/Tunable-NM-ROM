"""Tiny random/affine CPU fixture; no trained model or scientific fields."""
import torch,json
from baselines.crom_boundary import BoundaryDecoder

torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
torch.manual_seed(907)
class PhysicalAffine(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(1,5));self.bias=torch.nn.Parameter(torch.tensor([.37]))
        self.register_buffer('std',torch.tensor([.23]));self.register_buffer('mean',torch.tensor([.41]))
    def forward(self,x):return self.mean+self.std*(x@self.weight.T+self.bias)
    def computeJacobianFullAnalytical(self,x):
        return (self.std[:,None]*self.weight)[None].expand(len(x),-1,-1),self(x)

native=PhysicalAffine();model=BoundaryDecoder(native,3)
x=torch.rand(17,5);x[:,:3]=torch.randn(17,3)
x[:8,-2:]=torch.tensor([[0.,.2],[1.,.7],[.3,0.],[.8,1.],[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
j,y=model.computeJacobianFullAnalytical(x)
auto=torch.func.vmap(torch.func.jacrev(lambda row:model(row[None])[0]))(x)
error=float((j-auto).abs().max().detach());assert error<1e-13
assert torch.equal(y[:8],torch.zeros_like(y[:8]))
assert torch.equal(j[:8,:,:3],torch.zeros_like(j[:8,:,:3]))
# Boundary spatial derivative need not vanish: product-rule contribution matters.
assert abs(float(j[0,0,-2].detach()))>1e-6
b,_=model.envelope(x);assert torch.equal(y,b*native(x))
assert float((y-b*(native(x)-native.mean)).abs().max().detach())>1e-3
# Parameters remain the exact native objects; gradient follows the wrapped loss.
assert list(model.parameters())[0] is native.weight
target=torch.randn_like(y);loss=1000*torch.mean((model(x)-target)**2)
g=torch.autograd.grad(loss,tuple(model.parameters()))
manual=1000*torch.mean((b*native(x)-target)**2);gm=torch.autograd.grad(manual,tuple(native.parameters()))
assert all(torch.equal(a,b) for a,b in zip(g,gm))
clone=BoundaryDecoder(PhysicalAffine(),3);clone.load_state_dict(model.state_dict());assert torch.equal(clone(x),model(x))
# Spatial finite differences, including the boundary polynomial extension.
eps=1e-6;fd=[]
for k in [-2,-1]:
 d=torch.zeros_like(x);d[:,k]=eps
 fd.append((model(x+d)-model(x-d))/(2*eps))
fd=torch.stack(fd,dim=-1);fd_error=float((fd-j[:,:,-2:]).abs().max().detach());assert fd_error<1e-9
print(json.dumps(dict(analytic_autodiff_max_error=error,spatial_finite_difference_max_error=fd_error,
    exact_boundary_field=True,exact_boundary_code_jacobian=True,spatial_product_rule=True,
    physical_denormalization_inside_envelope=True,wrapped_objective_gradient_parity=True,state_roundtrip=True)))
