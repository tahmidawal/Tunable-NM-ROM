# Fixed-obstacle incompressible Navier–Stokes campaign

Status: discovery and pre-production design; no reference or ROM results exist yet.

The authoritative request is saved in USER-REQUEST.txt. The canonical project log remains
`/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md`. Experimental fork is consolidated
`93f399faad67f94ea2b56fe25f13003143e64694`; mirror parent
`5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f`. No existing sealed cohort may be opened.

## Ownership

| Lane | Owner | Worktree suffix | Cluster namespace |
|---|---|---|---|
| design, integration, evidence | coordinator | 2026-10-27-ns-obstacle | ns-obstacle-20261027 |
| cylinder reference | reference_2d | 2026-10-27-ns-obstacle-2d | ns-obstacle-2d-20261027 |
| sphere reference and feasibility | reference_3d | 2026-10-27-ns-obstacle-3d | ns-obstacle-3d-20261027 |
| coordinate bank, quadrature and POD | rom_baselines | 2026-10-27-ns-obstacle-rom | ns-obstacle-rom-20261027 |

Each namespace is below `/cluster/tufts/paralab/tawal01/`. Independent reviewers are
read-only. Discovery precedes design review; reference/data admission precedes training.

## Initial resource ceiling

Proposed first envelope, to be frozen in reviewed DESIGN.md before any run: twelve total
allocated GPU-hours, including allocations running CPU FE work, with eight for reference
verification and four reserved for a conditional small ROM/POD pilot. At most two campaign
allocations simultaneously, one GPU and at most sixteen CPU cores per allocation; first
reference smoke allocation no more than twenty minutes per dimension. Initial local/remote
unique-data cap sixty/fifty GiB respectively. Individual jobs enforce wall time and memory.
This is a ceiling, not a commitment to spend it. No broad dependency changes or paid service.

Full transient/unsteady and larger 3D stages remain conditional on numerical and resource
gates. Any stage failing its declared stop rule is recorded, not repeatedly optimized away.

## Review and current release

`DESIGN.md` with A1/A2 and `campaign.json` are the authoritative prospective contract.
Discovery proposals are preserved as history and do not override these files. Independent
Codex numerical and physics reviews permit reference/control implementation; code, exact
job configuration and required Claude design review remain necessary before numerical
submission. No production data, bank fitting or performance claim is released.
