"""Build exact shared reuse from retained Re20 and immutable-cache archives."""
from pathlib import Path
import json
from delta_retention import archive_catalog,add_warm_catalog,sha,dump
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];out=B/'g2/population-rest01/SHARED-REUSE-INDEX.json'
def descriptor(name):
 a=B/'artifacts'/name;cold=a/'GIT-COLD-ARCHIVE.json';retention=json.loads((a/'RETENTION.json').read_text())
 return {'job_name':name,'sha256':retention['archive_sha256'],'bytes':retention['archive_bytes'],'cold_manifest_path':str(cold.relative_to(REPO)),'cold_manifest_sha256':sha(cold)}
first='g2-population-re20-02';d=descriptor(first);index=archive_catalog(B/'runs'/(first+'-collection.tar.gz'),'re20_complete',d['sha256'],d,B/'artifacts'/first,first)
add_warm_catalog(index,B/'artifacts/g2-cache-primary01','results/cache_primary_control/caches','cache_primary',descriptor('g2-cache-primary01'))
for entry in index['entries'].values():
 if 'warm_source'in entry:entry['warm_source']=str(Path(entry['warm_source']).relative_to(REPO))
index['source_policy']='Every locator is backed by an immutable complete archive and original checksum inventory. Filename similarity never permits reuse; SHA256 and length must both match. Warm paths are optional exact-byte copies; historical Git archive recovery is mandatory fallback.'
index['recovery_helper_sha256']=sha(B/'g2/git_archive_recovery.py')
dump(out,index);print(json.dumps({'catalog_sha256':sha(out),'unique_content_entries':len(index['entries']),'base_archives':list(index['archives'])}))
