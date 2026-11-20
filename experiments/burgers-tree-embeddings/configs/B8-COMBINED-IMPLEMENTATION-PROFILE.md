# B8 sequential implementation profiles

Run profile_residual_implementation.py in one fresh owned job of at most15 allocated GPU-minutes, within B8 and the maximumtwo pending/running limit. It executes the previously registered B8-JACOBIAN-PROFILE.md and B8-OUTPUT-PROFILE.md diagnostics sequentially in independent Python subprocesses. Each inherits verified source/GPU settings and runs its own preflight. Full PDE trajectory or method/comparator qualification is not inferred from these profiles. Retain each result/log and all costs, including a failed profile.

Same retained b8childprimary checkpoint and b8treepde trajectory; unchanged equations, precision, physical spacing and stopping criteria. Local and remote retention of the source trajectory is verified before launch. Prospective follow-up requires exact-equivalence gates and demonstrated same-invocation cost reduction before a fresh complete trajectory replay.
