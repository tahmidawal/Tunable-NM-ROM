"""CPU finite-difference control for the independent MLP geometry audit."""
import unittest
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from restore_archive import main as restore
import numpy as np
from audit_dynamics import geometry

class AuditGeometryTests(unittest.TestCase):
    def test_numpy_chain_rule_with_independent_finite_differences(self):
        rng=np.random.default_rng(7090722)
        p={'p/linear':rng.normal(size=(7,3))*.2,'p/bias':rng.normal(size=7)*.1,'frozen/output_scale':np.array(.3)}
        for name,nin,nout in [('l1',3,8),('l2',8,8),('out',8,7)]:
            p[f'p/{name}/w']=rng.normal(size=(nin,nout))*.2;p[f'p/{name}/b']=rng.normal(size=nout)*.1
        transform=rng.normal(size=(7,7));z=rng.normal(size=3);w=rng.normal(size=3)
        def forward(zz):
            h=zz
            for layer in ('l1','l2'):
                h=h@p[f'p/{layer}/w']+p[f'p/{layer}/b'];h=h/(1+np.exp(-h))
            return transform@(p['p/bias']+p['p/linear']@zz+.3*(h@p['p/out/w']+p['p/out/b']))
        a,b,j,curve=geometry(p,transform,z,w)
        np.testing.assert_allclose(a,forward(z),atol=1e-14)
        eps=1e-5
        fd=np.column_stack([(forward(z+eps*e)-forward(z-eps*e))/(2*eps) for e in np.eye(3)])
        np.testing.assert_allclose(j,fd,atol=1e-10)
        eps=1e-3
        second=(forward(z+eps*w)-2*forward(z)+forward(z-eps*w))/(eps*eps)
        np.testing.assert_allclose(curve,second,atol=1e-8)
        np.testing.assert_allclose(b,j@w,atol=1e-14)

class ArchiveRestoreTests(unittest.TestCase):
    def test_missing_restore_and_existing_mismatch_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            record=Path(temp);files={'cluster/existing.txt':b'existing data','cluster/missing.txt':b'restored data'}
            checks=''.join(hashlib.sha256(value).hexdigest()+'  '+name.removeprefix('cluster/')+'\n' for name,value in files.items()).encode()
            for name in ('PULL.sha256','MANIFEST.sha256','RESULTS.sha256'):files['cluster/'+name]=checks
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
                for name,value in files.items():
                    info=tarfile.TarInfo(name);info.size=len(value);tar.addfile(info,io.BytesIO(value))
            payload=buffer.getvalue();parts=[payload[:len(payload)//2],payload[len(payload)//2:]]
            names=[]
            for i,value in enumerate(parts):
                name=f'archive.part-{i:03d}';(record/name).write_bytes(value);names.append(name)
            (record/'ARCHIVE.json').write_text(json.dumps(dict(ordered_parts=names,original_sha256=hashlib.sha256(payload).hexdigest())))
            (record/'ARCHIVE.sha256').write_text(''.join(hashlib.sha256(value).hexdigest()+'  '+name+'\n' for name,value in zip(names,parts)))
            (record/'cluster').mkdir();(record/'cluster/existing.txt').write_bytes(files['cluster/existing.txt'])
            restore(record)
            self.assertEqual((record/'cluster/missing.txt').read_bytes(),files['cluster/missing.txt'])
            (record/'cluster/existing.txt').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):restore(record)
            self.assertEqual((record/'cluster/existing.txt').read_bytes(),b'changed')

if __name__=='__main__':unittest.main()

