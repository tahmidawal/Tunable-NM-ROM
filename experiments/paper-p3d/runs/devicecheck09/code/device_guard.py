"""Untimed CUDA-driver physical-device identity guards for paired timings."""
import ctypes
import os
import subprocess
import uuid
import jax


class Guard:
    def __init__(self):
        self.cuda=ctypes.CDLL('libcuda.so.1')
        assert self.cuda.cuInit(0)==0
        self.env={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_JOB_ID']}
        self.initial=self.check()
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id','--format=csv,noheader'],text=True)
        assert self.initial in inventory
        self.record=dict(physical_uuid=self.initial,environment=self.env,nvidia_smi_inventory=inventory,
            jax_devices=[str(x) for x in jax.devices()],cuda_visible_device_count=1,
            checks_outside_timing=True,policy='CUDA driver UUID before and after every timed invocation; exactly one CUDA/JAX visible GPU.')
        print('DEVICE_GUARD '+str(self.record),flush=True)

    def check(self):
        devices=jax.devices();assert len(devices)==1 and devices[0].platform=='gpu'
        assert len(jax.local_devices())==1
        if hasattr(self,'env'):assert self.env=={k:os.environ.get(k) for k in self.env}
        count=ctypes.c_int();assert self.cuda.cuDeviceGetCount(ctypes.byref(count))==0 and count.value==1
        device=ctypes.c_int();assert self.cuda.cuDeviceGet(ctypes.byref(device),0)==0
        raw=(ctypes.c_ubyte*16)();fn=getattr(self.cuda,'cuDeviceGetUuid_v2',self.cuda.cuDeviceGetUuid)
        assert fn(ctypes.byref(raw),device)==0
        value='GPU-'+str(uuid.UUID(bytes=bytes(raw)))
        if hasattr(self,'initial'):assert value==self.initial
        return value


def audit(record):
    g=record['device_guard'];u=g['physical_uuid'];assert u.startswith('GPU-')
    assert g['cuda_visible_device_count']==1 and len(g['jax_devices'])==1
    assert u in g['nvidia_smi_inventory'] and g['checks_outside_timing']
    assert record['device_guard_final_uuid']==u
    for row in record['invocations']:
        assert row['physical_uuid_before']==u==row['physical_uuid_after']
    return dict(passed=True,physical_uuid=u,checked_timed_invocations=len(record['invocations']),
        environment=g['environment'],scope='Independent record consistency; physical UUID queried through CUDA driver outside every timed invocation.')
