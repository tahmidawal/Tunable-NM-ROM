# Full raw q-ridge archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including every numerical array, the immutable staged source, logs and manifests.

From this directory, verify and restore into a fresh destination:

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored
tar -xzf collection.tar.gz -C restored
cd restored
sha256sum -c OUTPUTS.sha256
sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads.
The whole-archive checksum is recorded in `archive.json`.

`output/bank_G.npz` is deliberately NOT in these chunks; see
`EXCLUDED-FROM-ARCHIVE.txt` and `BANK-SHA256.txt`.
