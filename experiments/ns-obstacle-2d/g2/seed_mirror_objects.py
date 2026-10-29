"""Seed oversized mirror uploads through bounded additive code-only histories.

The final mirror commit/tree is unchanged. Seed refs stay reachable; no force push,
history rewrite, object pruning, or scientific-file alteration occurs.
"""
import os,subprocess,sys,tempfile
target,parent=sys.argv[1:3]
def git(*args,input=None,env=None):return subprocess.check_output(['git',*args],input=input,env=env).decode().strip()
tree=git('rev-parse',target+'^{tree}');ref='refs/heads/codeonly/seed/nsobs2d-'+tree[:16]
advertised=git('ls-remote','origin',ref);seed=advertised.split()[0] if advertised else parent
audit_target=git('commit-tree',tree,'-p',seed,'-m','Local mirror payload audit') if seed!=parent else target
objects=git('rev-list','--objects',audit_target,'^'+parent,'^'+seed)
ids=[line.split()[0] for line in objects.splitlines()]
if not ids:sys.exit(0)
metadata=git('cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)',input=('\n'.join(ids)+'\n').encode())
blobs=[(oid,int(size)) for oid,kind,size in (line.split() for line in metadata.splitlines()) if kind=='blob']
paths={}
for entry in subprocess.check_output(['git','ls-tree','-r','-z',target]).split(b'\0'):
    if not entry:continue
    metadata,path=entry.split(b'\t',1);mode,kind,oid=metadata.decode().split()
    if kind=='blob':paths.setdefault(oid,[]).append((mode,path.decode()))
limit=900*1024**2
if sum(size for _,size in blobs)<=limit:sys.exit(0)
if any(size>limit for _,size in blobs):raise ValueError('one blob exceeds bounded seed payload')
groups=[];group=[];total=0
for oid,size in sorted(blobs):
    if group and total+size>limit:groups.append(group);group=[];total=0
    group.append((oid,size));total+=size
if group:groups.append(group)
with tempfile.TemporaryDirectory(prefix='mirror-seed-') as temporary:
    env={**os.environ,'GIT_INDEX_FILE':temporary+'/index'}
    for index,group in enumerate(groups):
        git('read-tree',seed+'^{tree}',env=env)
        for oid,_ in group:
            for mode,path in paths[oid]:git('update-index','--add','--cacheinfo',mode+','+oid+','+path,env=env)
        seed_tree=git('write-tree',env=env)
        new=git('commit-tree',seed_tree,'-p',seed,'-m',f'Additive bounded mirror object seed for {tree}; batch {index+1}/{len(groups)}')
        print(f'seeding {len(group)} blobs / {sum(size for _,size in group)} bytes to {ref}',flush=True)
        subprocess.run(['git','push','origin',new+':'+ref],check=True)
        if git('ls-remote','origin',ref).split()[0]!=new:raise RuntimeError('seed receipt uncertain; inspect before retry')
        seed=new
print('bounded seeds retained at '+ref+' '+seed,flush=True)
