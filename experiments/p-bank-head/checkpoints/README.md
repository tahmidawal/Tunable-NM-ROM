# Trained checkpoints from `pbh01`

Every checkpoint this cell trained, Git-tracked so the branch can be rerun and not only read.
`sc.load_pkl(path)` returns `(params, Z_tr, cfg)`; `cfg` carries `layer`, `rank` (R), `sources`
(S), `K` and, for head arms, `beta_weak` / `beta_smooth` and the bank they were fitted on.

- `bank_R{R}_S{S}.pkl` — the six bank-sweep arms, each with its own jointly trained K=16 head.
- `head_K{K}_w{beta_weak}_s{beta_smooth}.pkl` — the twelve head-sweep arms, all on the frozen
  `bank_R512_S192` bank, which is the arm the **withdrawn** selection rule chose. The rule in
  force (DESIGN.md amendment 4) selects `bank_R512_S3072`; the head sweep on that bank is in
  `pbh02`.
- `primary_K{K}-basis.npz` — the 32-direction training-residual correction bases of the two
  `pbh01` head primaries.

Checksums and provenance for all of them are in `../runs/pbh01/archive/output/result.json`
under `checkpoints`, and the whole raw job is preserved in `../artifacts/pbh01/`.
