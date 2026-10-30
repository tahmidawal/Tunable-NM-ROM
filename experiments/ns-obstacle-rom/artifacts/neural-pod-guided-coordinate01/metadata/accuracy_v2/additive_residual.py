"""Prospective capacity helper only: four existing columns plus four corrections.

No trainer, parent/model selection, serialization contract or execution protocol.
The correction uses the existing features/envelope, not a new frequency policy.
Both values and derivatives retain the original physical-coordinate convention.
"""
import numpy as np
from accuracy_v2.spatial_bank import initialize,evaluate_layers


def initialize_correction(width=128,depth=3,seed=0):
    bank=initialize(rank=4,width=width,depth=depth,seed=seed)
    # Nonzero hidden features permit a first output-layer update; exactly zero
    # final weights and bias preserve the parent function and derivative.
    w,b=bank.layers[-1];bank.layers[-1]=(np.zeros_like(w),np.zeros_like(b))
    return bank.layers


def evaluate_correction_sum(parent_values,parent_physical_gradients,layers,points,xp=np):
    values,gradients=evaluate_layers(layers,points,xp)
    if values.shape!=(len(points),2,4) or gradients.shape!=(len(points),2,2,4):raise ValueError('exactly four correction columns required')
    if parent_values.shape!=values.shape or parent_physical_gradients.shape!=gradients.shape:raise ValueError('same original four-column parent layout required')
    # A future trainer supplies frozen parent arrays; this formula does not
    # concatenate online columns or modify pressure/supremizer/lift machinery.
    return parent_values+values,parent_physical_gradients+gradients
