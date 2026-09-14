"""Bounded plumbing/fit smoke; no production accuracy or latency claims."""
from pathlib import Path
from dataset import preflight, calibrate, generate, HERE
from diagnose import run
core,jax,info=preflight(True)
out=HERE/'checks/diagnosis-smoke'
calibrate(core,jax,info,out/'calibration',1,[32],64,128,.1)
generate(core,jax,info,out/'validation','validation',1,32,out/'calibration/calibration.json',True)
run(out/'validation/index.json',out/'diagnosis',True,1)
