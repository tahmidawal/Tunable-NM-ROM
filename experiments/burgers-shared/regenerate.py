"""Guard the retained streaming extractor's exact dependencies and initial-code ordering."""
import hashlib
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/separable-decoder'))
import sep_coeff_extract as X


def main():
    assert hashlib.sha256(Path(X.CKPT).read_bytes()).hexdigest()=='18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'
    assert (X.N,X.K,X.R,X.MAX_SNAPS,X.T_EARLY,X.N_TEST,X.SEED0,X.EXTRA_SEED,X.EXTRA_TRAJ,X.LOOSE)==(256,16,512,131072,5,8,0,1000,4032,0)
    assert X.bc.SEED==0 and X.bc.TEST_SEED==1 and X.bc.NUM_STEPS==50 and X.bc.DT==.005
    assert (X.bc.bf.NEWTON_ITERS, X.bc.bf.LIN_TOL, X.bc.bf.LIN_MAXITER) == (8, 1e-10, 2000)
    for module in (X,X.sc,X.bc,X.bc.bf,X.bc.mp,X.bc.nda):
        assert Path(module.__file__).resolve().is_relative_to(ROOT),module.__file__
    # Compare the actual historical JOB manifest, including its staged solver patch.
    legacy = dict(line.split('  ', 1)[::-1] for line in
                  (ROOT/'experiments/burgers-shared/checks/legacy-data-manifest.sha256').read_text().splitlines())
    source_paths = ((X, 'code/sep_coeff_extract.py'), (X.sc, 'code/sep_common.py'),
        (X.bc, 'code/deps/burgers2d-rom-latent-stepping/blat_common.py'),
        (X.bc.bf, 'code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py'),
        (X.bc.mp, 'code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'),
        (X.bc.nda, 'code/deps/nonlinear-decoder-architecture/nda_arch.py'))
    for module, name in source_paths:
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == legacy[name], name
    _,z,cfg=X.sc.load_pkl(X.CKPT)
    n=(X.bc.bf.N_TRAIN+X.bc.bf.N_VAL+X.EXTRA_TRAJ)*(X.bc.NUM_STEPS+1)
    rng=np.random.default_rng(X.SEED0);t=np.arange(n)%(X.bc.NUM_STEPS+1)
    early=np.flatnonzero(t<=X.T_EARLY);rest=np.flatnonzero(t>X.T_EARLY)
    pick=np.sort(np.concatenate((early,rng.choice(rest,X.MAX_SNAPS-len(early),replace=False))))
    assert z.shape==(131072,16) and np.array_equal(pick,np.asarray(cfg['hfit_pick']))
    print('EXTRACT_SOURCE_AND_PICK_PASS historical_generator=c521b36a NEWTON_ITERS=8 LIN_TOL=1e-10 LIN_MAXITER=2000',flush=True)
    X.main()


if __name__=='__main__':main()
