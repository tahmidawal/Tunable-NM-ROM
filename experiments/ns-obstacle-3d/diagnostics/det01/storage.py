"""Sequential-process output accounting and bounded compressed array chunks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import resource
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


class Budget:
    def __init__(self, root, policy):
        self.root=Path(root); self.policy=policy; self.ledger=self.root/'budget.json'
        if not self.ledger.exists():
            self.ledger.write_bytes(encode({'stored_point_records':0}))

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def guard(self, addition=0, points=0, replacing=None):
        self.state=json.loads(self.ledger.read_text())
        if self.state['stored_point_records']+points>self.policy['max_combined_point_records']:
            raise MemoryError('combined stored-point cap exceeded')
        old=Path(replacing).stat().st_size if replacing and Path(replacing).exists() else 0
        if self.used()-old+addition>self.policy['max_output_bytes']-1048576:
            raise MemoryError('combined output cap exceeded; terminal metadata reserve retained')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if rss>self.policy['max_rss_bytes']:
            raise MemoryError('observed process RSS cap exceeded')

    def finish(self,points=0):
        self.state['stored_point_records']+=points
        self.state['last_observed_output_bytes']=self.used()
        self.state['last_process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        self.ledger.write_bytes(encode(self.state))
        self.guard()

    def json(self,path,value):
        path=Path(path);data=encode(value);self.guard(len(data),replacing=path)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);self.finish()

    def arrays(self,path,points=0,**arrays):
        path=Path(path)
        bound=int(sum(a.nbytes for a in arrays.values())*1.01)+16384
        self.guard(bound,points,replacing=path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays)
        self.finish(points)


class Chunks:
    def __init__(self,budget,folder,dtype):
        self.budget=budget;self.folder=Path(folder);self.folder.mkdir()
        if dtype.itemsize>512:raise ValueError('record-size projection violated')
        self.buffer=np.empty(budget.policy['chunk_rows'],dtype=dtype)
        self.fill=0;self.total=0;self.files=[]

    def append(self,rows):
        start=self.total
        while len(rows):
            take=min(len(rows),len(self.buffer)-self.fill)
            self.buffer[self.fill:self.fill+take]=rows[:take]
            self.fill+=take;self.total+=take;rows=rows[take:]
            if self.fill==len(self.buffer):self.flush()
        return start

    def flush(self):
        if not self.fill:return
        path=self.folder/f'part-{len(self.files):04d}.npz'
        self.budget.arrays(path,points=self.fill,records=self.buffer[:self.fill])
        self.files.append({'path':path.name,'rows':self.fill,'sha256':sha(path)})
        self.fill=0
        self.budget.json(self.folder/'INDEX.json',{'records':self.total,'files':self.files})

    def __enter__(self):return self
    def __exit__(self,*unused):self.flush()
