# Source this (or prefix commands with `env $(cat gpu_env.sh | xargs)`) to run the study on the
# AMD Radeon 8060S (gfx1151) through the ROCm JAX plugin in ~/Venv/jax.
#
# Why it was not working out of the box:
#   * the study's own .venv has jax[cpu]; the ROCm build lives in ~/Venv/jax (jax 0.9.2 + jax-rocm7 plugin)
#   * that plugin needs ROCm 7.12+ shared libraries (libroctracer64.so.4, librocblas.so.5, libhipblaslt.so.1,
#     libMIOpen.so.1, libhipfft.so.0, librccl.so.1); the system /opt/rocm is 7.2.4 and lacks them, but the
#     python ROCm SDK copied under ~/Venv/rocm-7.12 has them -> put those on LD_LIBRARY_PATH
#   * the integrated GPU reports the whole 121 GB unified memory as its pool; JAX preallocates 75% of it and
#     the kernel OOM-killer kills the process (exit 137) on the first compute -> disable preallocation
export LD_LIBRARY_PATH=$HOME/Venv/rocm-7.12/_rocm_sdk_core/lib:$HOME/Venv/rocm-7.12/_rocm_sdk_libraries_gfx1151/lib:$HOME/Venv/rocm-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYGPU=$HOME/Venv/jax/bin/python
