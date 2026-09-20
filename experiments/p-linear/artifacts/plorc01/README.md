# Full raw p-linear archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including every numerical array (all output fields), the immutable staged source, logs
and manifests.

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

The complete raw `output/result.json` is retained inside the bounded archive; its
local expanded convenience copy is not separately tracked. `audit.json` is the
strengthened independent local NumPy audit, including every retained multistart
objective and selected coefficient feasibility. The original cluster audit remains
immutable inside the archive. `restore-audit.json` confirms actual reconstruction
and extraction of these chunks and both source/output manifests. The whole-archive
checksum is in `archive.json`. The generated compact tables are in
`../../reports/oracle-confirmation.json`.
