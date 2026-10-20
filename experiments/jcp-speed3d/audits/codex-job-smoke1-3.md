**HOLD.** Reviewed HEAD `059af2a227`. The previous blocker is only partially resolved.

1. **Host-OOM isolation remains incomplete.** [smoke.sh:15](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke.sh:15) launches `smkmesh384` as a separate Python process, but gives it no separate memory limit protecting the supervisor. It still shares the batch step’s 200G limit. An OOM can kill other processes or the entire step, preventing acceptance and evidence collection; this depends on cluster configuration. [Slurm documentation](https://slurm.schedmd.com/cgroup.conf.html) explicitly describes these behaviours. The round-2 requirement for a resource boundary preserving the supervisor remains unmet.

2. **New honest-path false-PASS risk.** [smoke_accept.py:119](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:119) treats any nonzero exit as resource termination if an OOM string appears **anywhere** in the log. An earlier recoverable allocation failure followed by an unrelated fatal error therefore drops all pilot gates. A CPU evaluation of the actual predicate confirmed this. Exit 137 alone likewise establishes SIGKILL, not its cause. Resource classification needs evidence tied to the terminating event; unexpected failures must remain failures.

The timing-attribution fix resolves the previous false declaration of ROM infeasibility: failures now belong to panels. The recorded subject list identifies candidate FOM subjects, not the particular failing call.

**Staged-job checks pass:**

- One H200, `gpu` partition, eight CPUs, 200G, 24 hours.
- GPU preflight exits 42 on CPU; cluster venv, x64, highest precision, E0 paths and paralab output/cache/tmp configured.
- All **45 manifest entries** verify; **41 staged repository files** match HEAD byte-for-byte; `COMMIT.txt` matches.
- Both shell syntax checks pass. Sequence includes both ROM tests, FOM test, grid/FOM/profile/mesh modes, isolated-process `smkmesh384`, then acceptance.

**Runtime:** 24 hours remains plausible but unverified. L=384 still schedules **16 full FOM solves** when both families qualify and both panels run: six calibration, two accuracy, eight warm/timed calls. Average solve costs of 5/15/30 minutes imply **1.33/4/8 hours**, plus ROM tables, ADT, compilation and earlier steps. Splitting the process adds repeated initialization and G5 checks; it does not reduce that solve count.

No repository or git changes, GPU execution, or job submission.