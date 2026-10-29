"""Actual1218-file recovery proof, including new scientific bytes and Git fallback."""
from pathlib import Path
import json,sys,shutil,subprocess
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'g2'))
from delta_retention import read,dump,sha,key,write_delta,recover_delta
root=B/'artifacts/g2-population-re20-02';index=read(B/'g2/population-rest01/SHARED-REUSE-INDEX.json');work=B/'staging/delta-retention-proof02';work.mkdir(exist_ok=False)
forced_new=['results/common/p4_g4_h3_baseline.npz','results/states/p3_g3_h3_solver_100x/jacobian_arrays.npz']
for relative in forced_new:
 p=root/relative;del index['entries'][key(sha(p),p.stat().st_size)]
p=root/'results/states/p4_g4_h1_solver_100x/config.json';forced_cold=key(sha(p),p.stat().st_size);index['entries'][forced_cold].pop('warm_source',None)
index_path=work/'PROOF-REUSE-INDEX.json';dump(index_path,index);archive=work/'proof.delta.tar.gz';receipt=write_delta(root,index_path,archive,root/'COLLECTION-SHA256SUMS',B/'runs/g2-population-re20-02');dump(work/'WRITE-RECEIPT.json',receipt)
command=[sys.executable,'-B',str(B/'g2/delta_retention.py'),'recover','--archive',str(archive),'--sha256',receipt['archive_sha256'],'--index',str(index_path),'--target',str(work/'restored'),'--repo',str(B.parents[1]),'--receipt',str(work/'CLI-RECOVERY.json')]
completed=subprocess.run(command,capture_output=True,text=True);(work/'CLI-RECOVERY.stdout').write_text(completed.stdout);(work/'CLI-RECOVERY.stderr').write_text(completed.stderr);completed.check_returncode();recovery=read(work/'CLI-RECOVERY.json');expected={line.split('  ',1)[1].removeprefix('./'):line.split('  ',1)[0]for line in(root/'COLLECTION-SHA256SUMS').read_text().splitlines()}
if len(expected)!=1218 or any(sha(work/'restored'/name)!=digest for name,digest in expected.items()):raise ValueError('full original inventory differs')
try:recover_delta(archive,receipt['archive_sha256'],index_path,work/'restored',repo=B.parents[1])
except FileExistsError:no_overwrite=True
else:raise AssertionError('overwrite was permitted')
record={'passed':True,'actual_original_file_inventory_sha256':sha(root/'COLLECTION-SHA256SUMS'),'source_complete_archive_sha256':'33c56e3dedaef0ce896ab9022bd4e7a27952f440aa3639f403403d96d0df44bc','writer_reader_source_sha256':sha(B/'g2/delta_retention.py'),'forced_new_actual_scientific_members':forced_new,'forced_historical_Git_recovery_key':forced_cold,'write':receipt,'recovery':recovery,'all1218_original_file_hashes_reproduced':True,'no_overwrite_negative_passed':no_overwrite,'scope':'Representation/recovery proof only; self-reuse ratios are not future-population savings estimates.'};dump(work/'ROUNDTRIP.json',record)
shutil.rmtree(work/'restored');archive.unlink();dump(work/'DERIVED-PROOF-CLEANUP.json',{'only_derived_recovery_and_delta_archive_removed_after_complete_verification':True,'originals_unchanged':True});print(json.dumps(record))
