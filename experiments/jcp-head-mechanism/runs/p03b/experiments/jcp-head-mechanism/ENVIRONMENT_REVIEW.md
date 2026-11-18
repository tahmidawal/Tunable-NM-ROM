# GPU pilot environment review

The repaired pilot is admitted for numerical accuracy after independent review. Its host CPU-affinity limitation prevents using diagnostic elapsed times as a speed claim.

Upstream attributes this error to Slurm CPU affinity restrictions; GPU execution continues with OS thread scheduling. [Primary upstream report](https://github.com/openxla/xla/issues/45484).

```text
E1118 16:44:41.030427  966387 numa_hwloc.cc:121] Call to hwloc_set_cpubind() failed: Invalid argument [22]
E1118 16:44:48.306492  966441 numa_hwloc.cc:121] Call to hwloc_set_cpubind() failed: Invalid argument [22]
E1118 16:45:05.311647  966778 numa_hwloc.cc:121] Call to hwloc_set_cpubind() failed: Invalid argument [22]
```

The recorded execution backend was `gpu`, array precision `float64`, and matrix-product precision `highest`. Source, operator, derivative, initial-state, stationarity, trajectory and same-grid reference checks passed. Raw logs and checksums remain in the retained run.

## Glossary

CPU affinity is a request to pin a host thread to particular processor cores. The GPU backend is where the numerical array calculations execute. Stationarity is the remaining local least-squares gradient. A same-grid reference uses the same spatial discretization with a smaller time step.
