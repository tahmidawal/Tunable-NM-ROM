"""Initial-field-only geometry; no generative descriptor or future state access."""
import numpy as np
def geometry(initial,nu,horizon=.25):
 u=np.asarray(initial)[::16,::16];x=np.arange(u.shape[0])*16/(initial.shape[0]-1);mass=u.sum();assert mass>0
 mx=float(u.sum(1)@x/mass);my=float(u.sum(0)@x/mass);variance=float((u.sum(1)@(x-mx)**2+u.sum(0)@(x-my)**2)/(2*mass));peak=float(u.max())
 width=4*np.sqrt(variance+2*nu*horizon)+peak*horizon/4
 center=np.array([mx,my])+peak*horizon/8
 return (float(center[0]-width/2),float(center[1]-width/2),float(width))
