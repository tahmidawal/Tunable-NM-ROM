"""Offline refusal/resumption/identity tests. Every protected input is a generated fixture."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import open_sealed as op
import published as pub

class OpeningTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.src=self.root/'source'
        self.arc=self.root/'archive'
        self.reg=self.root/'opening'
        self.src.mkdir()
        self.put(self.arc/'store'/'MANIFEST.sha256',b'fixture manifest')
        self.put(self.arc/'store'/'STATUS.json',b'{}')
        self.store=op.sha(b'fixture manifest')
        self.key=self.arc/'custody'/'sealed.key'
        self.put(self.key,b'0'*64+b'\n')
        seed=op.packed({'2d/sealed128':101,'3d/sealed64':102,'2d/ood32':103,'3d/ood16':104})
        cseed=self.enc(seed)
        self.put(self.arc/'sealed'/'SEEDS.json.enc',cseed)
        one=b'fixture reference'
        sealed=self.enc(one)
        runs=[]
        self.paths=[]
        for d,(co,n) in op.COHORTS.items():
            dest=self.arc/'sealed'/f'{d}d'/co/'fake'
            self.put(dest/'result.json',b'public proof')
            entries=[]
            for i in range(n):
                for kind in ('S','ST'):
                    for ext in ('npz','json'):
                        rel=f'{d}d/{co}/{kind}/c{i:03d}.{ext}.enc'
                        self.put(dest/rel,sealed)
                        self.paths.append(dest/rel)
                        entries.append(dict(file=rel,cipher=op.sha(sealed),plain=op.sha(one)))
            manifest=op.packed(entries)
            ciphertext=self.enc(manifest)
            self.put(dest/'manifest.json.enc',ciphertext)
            runs.append(dict(dim=d,cohort=co,attempt='fake',result_sha256=op.sha(b'public proof'),
                             manifest_cipher_sha256=op.sha(ciphertext),manifest_plain_sha256=op.sha(manifest),
                             ciphertexts=[{'file':r['file'],'cipher':r['cipher']} for r in entries]))
        proof=op.packed(dict(store_version=self.store,original_batches=runs,
            seed_provenance=dict(seeds_cipher_sha256=op.sha(cseed),seeds_plain_sha256=op.sha(seed))))
        self.put(self.src/'original_archive.json',proof)
        self.put(self.src/'configs'/'p.json',b'{}')
        self.put(self.src/'roster.json',b'fixture-roster')
        f=op.packed(dict(store_version=self.store,cohorts={'2':'sealed128','3':'sealed64'},
                         reference_status_sha256=op.sha(b'{}'),files={'roster.json':op.sha(b'fixture-roster'),'original_archive.json':op.sha(proof),'configs/p.json':op.sha(b'{}')}))
        self.put(self.src/'F1.json',f)
        self.receipt=op.packed({'decision':'ACCEPT','freeze_sha256':op.sha(f),'selections_at_git_verified':True,'freeze_commit':'fixture','reviewer':'fixture','roster_sha256':op.sha(b'fixture-roster')})
        patched=patch.dict(os.environ,{'JCP_APPROVED_REVIEW_SHA256':op.sha(self.receipt)})
        patched.start()
        self.addCleanup(patched.stop)
        for module in (op,pub):
            m=patch.object(module,'STORE',self.store)
            m.start()
            self.addCleanup(m.stop)

    def put(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)

    def enc(self,data):
        return subprocess.run(['openssl','enc','-aes-256-cbc','-pbkdf2','-md','sha256','-salt','-pass',f'file:{self.key}'],
                              input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout

    def test_wrong_freeze_fails_before_reading_key(self):
        with self.assertRaises(FileNotFoundError):
            op.opening(self.src,self.arc,self.reg)
        self.put(self.src/'F1-REVIEW.json',self.receipt)
        self.put(self.src/'configs'/'p.json',b'changed')
        with self.assertRaises(ValueError):
            op.opening(self.src,self.arc,self.reg)
        self.assertFalse(self.reg.exists())

    def test_resume_replay_and_tamper(self):
        self.put(self.src/'F1-REVIEW.json',self.receipt)
        victim=self.paths[4]
        raw=victim.read_bytes()
        victim.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            op.opening(self.src,self.arc,self.reg)
        self.assertTrue((self.reg/'INTENT.json').is_file())
        self.assertFalse((self.reg/'COMPLETE.json').exists())
        with patch.object(pub,'REGISTRY',self.reg),patch.dict(os.environ,{'JCP_OPEN_REGISTRY':str(self.reg)}):
            with self.assertRaises(FileNotFoundError):
                pub.Opened(self.src,self.src/'configs'/'p.json')
            victim.write_bytes(raw)
            out=op.opening(self.src,self.arc,self.reg)
            self.assertEqual(out['file_count'],len(op.ALLOW))
            opened=pub.Opened(self.src,self.src/'configs'/'p.json')
            self.assertEqual(opened.seed(2),101)
            self.assertEqual(set(json.loads((self.reg/'payload'/'inrange-seeds.json').read_text())),{'2d/sealed128','3d/sealed64'})
            with self.assertRaises(ValueError):
                opened.bytes('2d/ood32/ST/c000.npz')
            self.key.unlink()
            self.assertEqual(op.opening(self.src,self.arc,self.reg),out)
            (self.reg/'published.json').chmod(0o600)
            (self.reg/'published.json').write_text('{}')
            with self.assertRaises(ValueError):
                pub.Opened(self.src,self.src/'configs'/'p.json')
if __name__=='__main__':
    unittest.main()
