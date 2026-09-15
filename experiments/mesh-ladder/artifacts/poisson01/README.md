# poisson01

Checksum-verified collection of cluster job `3711389` in namespace
`/cluster/tufts/paralab/tawal01/mrladder_20260914`. The remote attempt directory was removed after this archive was
verified. `result.json` beside these chunks is the driver record, copied out so
the report generator does not need to unpack the archive.

Restore:

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored/poisson01/out && sha256sum -c ../OUTPUTS.sha256
```
