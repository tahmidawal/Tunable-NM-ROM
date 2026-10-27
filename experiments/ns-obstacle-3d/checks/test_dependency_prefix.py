"""Metadata-only prefix guard tests; no package installation or native import."""
from pathlib import Path
import base64
import csv
import hashlib
import importlib.util
import io
import os
import sysconfig
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lane_dependencies',ROOT/'install_deps.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PrefixGuard(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'checks')
        self.job=Path(self.tmp.name);self.prefix=self.job/'deps'
        self.site=Path(sysconfig.get_path('purelib',vars={'base':str(self.prefix),'platbase':str(self.prefix)}))
        self.site.mkdir(parents=True);(self.job/'results').mkdir()
        self.meta={'candidates':{'netgen-occt':{'version':'7.8.1'}}}
        self.dist=self.site/'netgen_occt-7.8.1.dist-info';self.dist.mkdir()
        self.tk=self.prefix/'lib/libTKernel.so.7.8.1';self.tk.write_bytes(b'fixture native bytes, never loaded')
        self.blas=self.site/'ngsolve_openblas/libopenblas.so';self.blas.parent.mkdir();self.blas.write_bytes(b'fixture BLAS bytes')
        (self.dist/'METADATA').write_text('Metadata-Version: 2.1\nName: netgen-occt\nVersion: 7.8.1\n')
        rows=[]
        for path in [self.tk,self.blas,self.dist/'METADATA']:
            value=path.read_bytes();digest=base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode()
            rows.append([os.path.relpath(path,self.site),'sha256='+digest,str(len(value))])
        rows.append([os.path.relpath(self.dist/'RECORD',self.site),'',''])
        buf=io.StringIO();csv.writer(buf).writerows(rows);(self.dist/'RECORD').write_text(buf.getvalue())
        self.wheel=self.job/'netgen_occt-fixture.whl'
        with zipfile.ZipFile(self.wheel,'w') as z:z.writestr('netgen_occt-7.8.1.data/data/lib/libTKernel.so.7.8.1',self.tk.read_bytes())
    def tearDown(self):self.tmp.cleanup()
    def verify(self):module.verify_prefix(self.prefix,self.meta,[str(self.wheel)])
    def test_valid_metadata_only_layout(self):self.verify()
    def test_missing_library_caught_from_raw_record(self):
        self.tk.unlink()
        with self.assertRaises(FileNotFoundError):self.verify()
    def test_corrupted_library_caught(self):
        b=self.tk.read_bytes();self.tk.write_bytes(b'X'+b[1:])
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):self.verify()
    def test_record_cannot_escape_prefix(self):
        with (self.dist/'RECORD').open('a') as f:f.write('../../../../outside,,\n')
        with self.assertRaisesRegex(RuntimeError,'escaped prefix'):self.verify()

if __name__=='__main__':unittest.main()
