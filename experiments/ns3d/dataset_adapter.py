"""NumPy-only contract for shared f64 FNO3D/U-Net3D training on vector NS.

Input information is u0 and viscosity only. The five geometric family parameters
in the archive are provenance, never predictor features. No wall mask is valid.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def load(path):
    with np.load(Path(path),allow_pickle=False) as archive:
        states=np.asarray(archive['states'])
        parameters=np.asarray(archive['parameters'])
    if states.dtype != np.float64 or parameters.dtype != np.float64:
        raise ValueError('dataset must retain native f64 fields and parameters')
    if states.ndim != 6 or states.shape[1:3] != (6,3):
        raise ValueError('expected (cases,6,3,N,N,N) velocity states')
    if len(set(states.shape[-3:])) != 1 or parameters.shape != (len(states),6):
        raise ValueError('cube or parameter manifest mismatch')
    if not np.all(np.isfinite(states)):
        raise ValueError('dataset contains nonfinite velocity')
    initial=np.ascontiguousarray(states[:,0])
    viscosity=np.ascontiguousarray(parameters[:,-1:])
    return dict(initial=initial,viscosity=viscosity,targets=np.ascontiguousarray(states[:,1:]),
                states=states,metadata=dict(n=states.shape[-1],boundary='periodic',
                    endpoint_included=False,output_times=np.linspace(0,.2,6).tolist(),
                    field='three-component velocity',initial_sha256=digest(initial),
                    states_sha256=digest(states),parameter_sha256=digest(parameters),
                    predictor_parameters=['viscosity'],input_channels=4,output_channels=15))


def raw_inputs(initial,viscosity):
    initial=np.asarray(initial)
    viscosity=np.asarray(viscosity)
    if initial.dtype != np.float64 or viscosity.dtype != np.float64:
        raise ValueError('f64 predictor input required')
    if initial.ndim != 5 or initial.shape[1] != 3 or viscosity.shape != (len(initial),1):
        raise ValueError('velocity/viscosity shape mismatch')
    channel=np.broadcast_to(viscosity[:,:,None,None,None],(len(initial),1,*initial.shape[-3:]))
    return np.ascontiguousarray(np.concatenate((initial,channel),axis=1))


def training_statistics(dataset):
    inputs=raw_inputs(dataset['initial'],dataset['viscosity'])
    mean=np.mean(inputs,axis=(0,2,3,4),keepdims=True)
    std=np.std(inputs,axis=(0,2,3,4),keepdims=True)
    std=np.maximum(std,1e-12)
    output_scale=float(np.sqrt(np.mean(dataset['targets']**2)))
    return dict(input_mean=mean,input_std=std,output_scale=max(output_scale,1e-12))


def targets_to_channels(target):
    if target.ndim != 6 or target.shape[1:3] != (5,3):
        raise ValueError('expected five evolved vector frames')
    return np.ascontiguousarray(target.reshape(len(target),15,*target.shape[-3:]))


def channels_to_trajectories(channels,initial):
    if channels.ndim != 5 or channels.shape[1] != 15:
        raise ValueError('expected time-major 15 output channels')
    evolved=channels.reshape(len(channels),5,3,*channels.shape[-3:])
    return np.concatenate((initial[:,None],evolved),axis=1)


def relative_errors(prediction,target,initial):
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise ValueError('trajectory shapes differ')
    error=np.sqrt(np.sum((prediction-target)**2,axis=(2,3,4,5)))
    norm=np.sqrt(np.sum(initial**2,axis=(1,2,3,4)))
    return error/np.maximum(norm[:,None],1e-300)


def self_check():
    rng=np.random.default_rng(21)
    states=rng.normal(size=(2,6,3,4,4,4)).astype(np.float64)
    target=states[:,1:]
    reconstructed=channels_to_trajectories(targets_to_channels(target),states[:,0])
    assert np.array_equal(reconstructed,states)
    errors=relative_errors(reconstructed,states,states[:,0])
    assert np.all(errors == 0)
    perturbed=reconstructed.copy();perturbed[:,1]=0
    expected=np.linalg.norm(states[:,1].reshape(2,-1),axis=1)/np.linalg.norm(states[:,0].reshape(2,-1),axis=1)
    assert np.allclose(relative_errors(perturbed,states,states[:,0])[:,1],expected)
    inputs=raw_inputs(states[:,0],np.asarray([[.002],[.01]]))
    assert inputs.shape == (2,4,4,4,4)
    assert np.all(inputs[0,3] == .002)
    return dict(passed=True,checks=['channel/frame roundtrip','initial-norm velocity metric',
                                   'viscosity-only parameter input'])


if __name__=='__main__':
    print(self_check())
