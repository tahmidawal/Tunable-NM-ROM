# local import path for the flat parent modules (the cluster stages them flat)
E=/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-poisson-bank-knob/experiments
export PYTHONPATH=$E/poisson-bank-knob:$E/hires-poisson:$E/multiresolution-poisson:$E/separable-decoder:$E/cost-to-tolerance:$E/wave2d-rom-latent-stepping/deps/multistage-precision:$E/mr-burgers2d:$E/head-ablation:$E/p-bank-head:$E/p-linear
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
