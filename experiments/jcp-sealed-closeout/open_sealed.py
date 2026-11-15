"""A3.8 one-time, resumable publication of the original in-family E0 reference cases."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/cluster/tufts/paralab/tawal01/jcp-ref')
STORE = '3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0'
REGISTRY = ARCHIVE.parent/'jcp-opening-records'/STORE
COHORTS = {2: ('sealed128',128), 3: ('sealed64',64)}
ALLOW = {f'{d}d/{co}/{k}/c{i:03d}.{ext}' for d,(co,n) in COHORTS.items()
         for i in range(n) for k in ('S','ST') for ext in ('npz','json')}

def sha(b):
    return hashlib.sha256(b).hexdigest()

def verify_file(path, expected):
    data=path.read_bytes()
    if sha(data)!=expected:
        raise ValueError(f'Content mismatch at {path}')
    return data

def packed(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()

def immutable(path,data):
    """Publish atomically without clobber; a retry must reproduce exactly the same bytes."""
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.exists():
        verify_file(path,sha(data))
        return
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.part-')
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp,0o400)
        os.link(tmp,path)
        dfd=os.open(path.parent,os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        os.unlink(tmp)

def verify_approval(source):
    freeze_bytes=(source/'F1.json').read_bytes()
    freeze=json.loads(freeze_bytes)
    review=(source/'F1-REVIEW.json').read_bytes()
    if sha(review)!=os.environ.get('JCP_APPROVED_REVIEW_SHA256'):
        raise ValueError('Review fingerprint disagrees with independently approved staged Git record')
    receipt=json.loads(review)
    frozen_hash=sha(freeze_bytes)
    if (receipt.get('decision')!='ACCEPT' or receipt.get('freeze_sha256')!=frozen_hash
            or not receipt.get('selections_at_git_verified') is True
            or not receipt.get('freeze_commit') or not receipt.get('reviewer')
            or receipt.get('roster_sha256')!=freeze.get('files',{}).get('roster.json')
            or not receipt.get('roster_sha256')):
        raise ValueError('Missing independent ACCEPT for the precise freeze')
    if freeze.get('store_version')!=STORE or freeze.get('cohorts')!={'2':'sealed128','3':'sealed64'}:
        raise ValueError('Unregistered store or population')
    for name, digest in freeze['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unconfined source path')
        verify_file(source/path,digest)
    return dict(freeze_sha256=frozen_hash,review_sha256=sha((source/'F1-REVIEW.json').read_bytes()),store_version=STORE)

def decrypt(data,key):
    proc=subprocess.run(['openssl','enc','-aes-256-cbc','-pbkdf2','-md','sha256','-salt','-d','-pass',f'file:{key}'],
                        input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if proc.returncode:
        raise ValueError('Original reference decryption failed')
    return proc.stdout

def opening(source,archive,registry):
    binding=verify_approval(source)  # before any attempt to inspect protected inputs
    manifest=json.loads((source/'original_archive.json').read_bytes())
    if manifest['store_version']!=STORE:
        raise ValueError('Archive commitment changed')
    verify_file(archive/'store'/'MANIFEST.sha256',STORE)
    expected_status=json.loads((source/'F1.json').read_bytes())['reference_status_sha256']
    verify_file(archive/'store'/'STATUS.json',expected_status)
    registry.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (registry/'lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(registry/'INTENT.json',packed(binding))
        if (registry/'COMPLETE.json').is_file():
            complete=json.loads((registry/'COMPLETE.json').read_bytes())
            if any(complete.get(k)!=v for k,v in binding.items()):
                raise ValueError('Foreign completion')
            verify_file(registry/'published.json',complete['published_sha256'])
            return complete
        # Prepare a resumable private publication. The consumer refuses access until COMPLETE.
        key=archive/'custody'/'sealed.key'
        output=registry/'payload'
        seed_prov=manifest['seed_provenance']
        expected_files=set()
        for batch in manifest['original_batches']:
            dim,co=batch['dim'],batch['cohort']
            if dim not in COHORTS or co!=COHORTS[dim][0]:
                raise ValueError('Attempt included an OOD cohort')
            root=archive/'sealed'/f'{dim}d'/co/batch['attempt']
            verify_file(root/'result.json',batch['result_sha256'])
            verify_file(root/'manifest.json.enc',batch['manifest_cipher_sha256'])
            for ent in batch['ciphertexts']:
                rel=ent['file'].removesuffix('.enc')
                if rel in expected_files or rel not in ALLOW or ent['file']!=rel+'.enc':
                    raise ValueError('Unexpected public encrypted inventory')
                verify_file(root/ent['file'],ent['cipher'])
                expected_files.add(rel)
        if expected_files!=ALLOW:
            raise ValueError('Incomplete committed inventory')
        seeds_data=decrypt(verify_file(archive/'sealed'/'SEEDS.json.enc',seed_prov['seeds_cipher_sha256']),key)
        if sha(seeds_data)!=seed_prov['seeds_plain_sha256']:
            raise ValueError('Seed payload integrity failure')
        # The original archive used one encrypted seed JSON. Release only the two in-family keys.
        seeds=json.loads(seeds_data)
        inrange={f'{dim}d/{co}':seeds[f'{dim}d/{co}'] for dim,(co,_) in COHORTS.items()}
        del seeds_data,seeds
        seed_bytes=packed(inrange)
        immutable(output/'inrange-seeds.json',seed_bytes)
        published={}
        for batch in manifest['original_batches']:
            dim,co=batch['dim'],batch['cohort']
            if dim not in COHORTS or co!=COHORTS[dim][0]:
                raise ValueError('Attempt included an OOD cohort')
            root=archive/'sealed'/f'{dim}d'/co/batch['attempt']
            verify_file(root/'result.json',batch['result_sha256'])
            encoded=verify_file(root/'manifest.json.enc',batch['manifest_cipher_sha256'])
            inner=decrypt(encoded,key)
            if sha(inner)!=batch['manifest_plain_sha256']:
                raise ValueError('Plain manifest mismatch')
            entries=json.loads(inner)
            cipher_list={v['file']:v['cipher'] for v in batch['ciphertexts']}
            if len(entries)!=len(cipher_list) or set(cipher_list)!={v['file'] for v in entries}:
                raise ValueError('Incomplete original encrypted inventory')
            for ent in entries:
                encrypted=ent['file']
                rel=encrypted.removesuffix('.enc')
                if rel not in ALLOW or rel in published or encrypted!=rel+'.enc' or cipher_list[encrypted]!=ent['cipher']:
                    raise ValueError('Unexpected or duplicate original case file')
                fn=output/rel
                if fn.exists():
                    verify_file(fn,ent['plain'])
                else:
                    plain=decrypt(verify_file(root/encrypted,ent['cipher']),key)
                    if sha(plain)!=ent['plain']:
                        raise ValueError('Decrypted file failed original commitment')
                    immutable(fn,plain)
                published[rel]=ent['plain']
        if set(published)!=ALLOW:
            raise ValueError('Publication missing cases or kinds')
        public_bytes=packed(dict(**binding,files=published,seed_sha256=sha(seed_bytes)))
        immutable(registry/'published.json',public_bytes)
        complete=dict(**binding,published_sha256=sha(public_bytes),file_count=len(published))
        immutable(registry/'COMPLETE.json',packed(complete))
        return complete

if __name__=='__main__':
    print(json.dumps(opening(HERE,ARCHIVE,REGISTRY),sort_keys=True))
