# Phase-10-D resource estimate

The sole diagnostic requests one H200, 8 CPUs, 96 GiB host memory, and 16
hours. It performs no model/optimizer update. The dominant work is one start,
35,904 snapshots, and at most 40 bounded-q attempts, plus independently repeated
initial/terminal Cox/K3 full-train verification.

P8-D completed 5,712 snapshots, two starts, and 40 attempts in 33m35s on an
H200. Phase10-D has 3.14 times as many snapshot-attempts. A conservative 5x
multiplier plus two hours of regeneration, compression, and independent audit
is below 12 hours. The 16-hour request preserves further margin without using
cross-job time as scientific evidence. Trace storage is below 1 GiB uncompressed,
well inside 96 GiB host memory; explicit 8-row device batches bound accelerator
memory.
