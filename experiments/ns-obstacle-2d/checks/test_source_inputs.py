"""Metadata-only regression for explicit archive input and hardlink closure."""
import hashlib,io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'g2'))
from source_inputs import SourceInputs

class ArchiveInputs(unittest.TestCase):
    def test_hardlink_and_multiple_members_share_one_archive_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'fixture.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                for name,value in [('job/a',b'first'),('job/b',b'second')]:
                    member=tarfile.TarInfo(name);member.size=len(value);tar.addfile(member,io.BytesIO(value))
                member=tarfile.TarInfo('job/alias');member.type=tarfile.LNKTYPE;member.linkname='job/a';tar.addfile(member)
            manifest=json.dumps({'fixture_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}).encode()
            def recover(args):args.output.write_bytes(archive.read_bytes())
            rows=[]
            for name,value in [('b',b'second'),('alias',b'first'),('a',b'first')]:
                rows.append({'source':'cold/'+name,'destination':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest(),'archive_source':{'manifest_source':'recovery.json','manifest_sha256':hashlib.sha256(manifest).hexdigest(),'member':'job/'+name}})
            destination=root/'output';destination.mkdir()
            with SourceInputs(root,'unused','fixture') as reader,patch('source_inputs.git_archive_recovery.recover',side_effect=recover) as recovery:
                reader.committed=lambda path:manifest
                reader.materialize(rows,destination)
                self.assertEqual(recovery.call_count,1)
                self.assertEqual((destination/'alias').read_bytes(),b'first')
                bad={**rows[0],'sha256':'0'*64}
                with self.assertRaises(ValueError):reader.read(bad)
                bad={**rows[0],'archive_source':{**rows[0]['archive_source'],'member':'../escape'}}
                with self.assertRaises(ValueError):reader.read(bad)

if __name__=='__main__':unittest.main()
