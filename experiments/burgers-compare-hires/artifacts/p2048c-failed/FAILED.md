# p2048c (job 4215837, H200 pax008) -- FAILED, no numbers used

Ran truth, Phase F, the robust-rule and bank-span arms, snapshots, POD fits and POD-16/64/256 blocks, then died silently
during the POD-512 quick run (last log line 10:57 EDT, job end 11:05, state FAILED exit 1, empty stderr, and not even the
`|| echo "CMP FAILED"` line of the batch script ran). sacct MaxRSS 182 GB (sampled) of a 240 GB request; the node
(pax008) reported only ~182 GB free physical memory shortly after. Most likely a host-memory kill; not proven.
Retry p2048d with --mem 400G. Only result.json (incomplete) and the logs are kept.
