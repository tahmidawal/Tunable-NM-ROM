"""Complete dyadic leaf partitions for local correction banks."""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True,order=True)
class Cell:
 depth:int
 i:int
 j:int
 def patch(self):return (self.i/2**self.depth,self.j/2**self.depth,1/2**self.depth)
 def children(self):return tuple(Cell(self.depth+1,2*self.i+i,2*self.j+j) for i in (0,1) for j in (0,1))
def leaves(refined):
 out=[]
 for c in Cell(0,0,0).children():out.extend(c.children() if 2*c.i+c.j in refined else [c])
 return tuple(sorted(out))
def initial_order(initial):
 u=np.asarray(initial)[::16,::16];gx,gy=np.gradient(u);e=(gx*gx+gy*gy).reshape(2,128,2,128).sum((1,3));s=e.copy();s[1:]+=e[:-1]/2;s[:,1:]+=e[:,:-1]/2
 return tuple(int(i) for i in np.argsort(-s.ravel(),kind='stable'))
def validate(cells):
 occupancy=np.zeros((4,4),int)
 for c in cells:
  assert 0<=c.depth<=2 and 0<=c.i<2**c.depth and 0<=c.j<2**c.depth
  w=2**(2-c.depth);occupancy[c.i*w:(c.i+1)*w,c.j*w:(c.j+1)*w]+=1
 assert np.all(occupancy==1)
 return True
