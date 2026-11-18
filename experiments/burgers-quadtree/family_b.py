"""Registered boundary-compatible localized family; every rejected center is retained."""
import numpy as np
def draw(seed,count):
 r=np.random.default_rng(seed);rows=[]
 for case in range(count):
  J=int(r.integers(1,3));background=float(r.uniform(.1,.3));blobs=[];rejected=[]
  for j in range(J):
   amp=float(r.uniform(.5,1.5));c=r.uniform(.2,.8,2)
   if j:
    while not .08<=np.linalg.norm(c-np.asarray(blobs[0]['center']))<=.35:
     rejected.append(c.tolist());c=r.uniform(.2,.8,2)
   widths=np.exp(r.uniform(np.log(.03),np.log(.12),2));theta=float(r.uniform(0,np.pi))
   blobs.append(dict(amplitude=amp,center=c.tolist(),widths=widths.tolist(),theta=theta))
  rows.append(dict(J=J,background=background,blobs=blobs,nu=float(np.exp(r.uniform(np.log(.0025),np.log(.02)))),rejected_centers=rejected))
 return rows

def initial(L,p):
 x=np.arange(L+1)/L;X,Y=np.meshgrid(x,x,indexing='ij');u=np.full_like(X,p['background'])
 for b in p['blobs']:
  dx=X-b['center'][0];dy=Y-b['center'][1];ct=np.cos(b['theta']);st=np.sin(b['theta'])
  xi=ct*dx+st*dy;eta=-st*dx+ct*dy
  u+=b['amplitude']*np.exp(-.5*((xi/b['widths'][0])**2+(eta/b['widths'][1])**2))
 u*=np.sin(np.pi*X)**2*np.sin(np.pi*Y)**2;u[[0,-1],:]=0;u[:,[0,-1]]=0
 return u
