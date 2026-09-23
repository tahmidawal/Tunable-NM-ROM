# local import path (the cluster stages the same files flat)
E=/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-poisson-bank-knob-3d/experiments
export PYTHONPATH_LSHAPE=$E/poisson-bank-knob-3d:$E/hires-poisson/lshape:$E/separable-decoder:$E/head-ablation:$E/mr-burgers2d
export PYTHONPATH_CUBE=$E/poisson-bank-knob-3d:$E/paper-p3d
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
