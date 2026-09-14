from dataset import preflight,calibrate,generate,HERE
from train_matched import run
core,jax,info=preflight(True)
out=HERE/'checks/matched-training-smoke'
calibrate(core,jax,info,out/'calibration',1,[32],64,128,.1)
generate(core,jax,info,out/'train','train',32,32,out/'calibration/calibration.json')
run(out/'train/index.json',out/'r128',128,True)
