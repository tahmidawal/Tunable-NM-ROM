# final256b (job 4175066) — FAILED after 4 min, no numbers

`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 39.62GiB [executable_name='jit_query']` on an H200 (143 GB).
Cause: the blocked encode/decode sliced the single 42 GB device bank (`bank[s:e]`), and each slice is a COPY — with 8 blocks live
inside one jit the bank was duplicated (42 GB + ~40 GB + fields).

Fix (committed, not yet run on the cluster): `bank_at` now STORES the bank as row blocks (~2M rows each), so no slice copy ever
exists; `bank_project` / `bank_expand` iterate the stored blocks; `tsqr_r`, `weak_matrix`, `sep2d_directions`, `run.py` bank_bytes and
`train_vp.floor_at` updated. At <= 128^3 there is exactly one block (unchanged path).
Verified locally against the committed final01 panel-A numbers at 64^3 (`diagnostics/block_storage_parity.py`): max |error difference|
7.5e-15 with 1 block and 7.5e-15 with 7 forced blocks — i.e. block splitting adds nothing beyond run-to-run noise.
Remote dir removed. Budget: the 2 user-authorised extra jobs (4171513, 4175066) are spent; 256^3 needs one more job.
