SUBMIT — staged q3r8 passes; submission-time guards remain required.

Paths below are relative to `experiments/jcp-dims/`.

- **Membership:** Exact ordered match of four exposed, untested candidates from the 436-candidate universe; predecessor labels agree with `analysis-3d.json`. All q+/q++ links and the control match registration. `configs/q3r8.json:78`, `make_round.py:36`.
- **Population:** Seed 923801, 64 cases for q/q+, cases 0–7 for q++; control retained. `configs/q3r8.json:5`, `d3run.py:528`.
- **Certificate:** Reproduced without writes: **0.5154107918 h**; 1.5× = **0.7731161876 h**, rounded to **1 h**. Config hash matches. `checks/resources-q3r8.json:6`, `resources.py:233`.
- **Staging:** All **178 manifest entries** verified; **174 provenance files** match HEAD `9b6a0a2f3`. H200, GPU partition, exit-42 preflight, f64/highest, isolated paralab paths, and shell syntax pass. `runs/q3r8/PROVENANCE.json:2`, `runs/q3r8/run.sbatch:3`.
- **Budget:** Validation passes; pending **1.03 ≥ 0.5154 h**. Reservation calculation: **15.9648 + 26.5 + 1 + 4 = 47.4648 ≤ 55 GPU-h**. `runs/GPU-LEDGER.json:215`.
- **Queue:** Successful SSH snapshot showed **2/3 lane, 3/8 account**, superseding the handoff’s 3/3. A second SSH check stalled, so remote-directory absence remains unverified. The wrapper must recheck capacity and directory absence. `cluster/submit.sh:18`, `cluster/submit.sh:29`.

No files changed, submissions made, arrays copied, or LAB-LOG writes.