"""Independent returned-artifact identity checks; no fitting or PDE calculation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();root=Path(a.output)
    execution=json.loads((root/'execution.json').read_text())
    phase=execution['phase'];cohort,count=('dev4',4) if phase=='dev' else ('val64',64)
    rows=json.loads((root/'all_scores.json').read_text());lookup={}
    for row in rows:
        key=(row['Lc'],row['source'],row['case'],row['method'])
        assert key not in lookup,('duplicate row',key)
        assert row['cohort']==cohort
        lookup[key]=row
    checked_coefficients=checked_timed_invocations=checked_fields=0
    for lc in (16,32):
        setup=json.loads((root/f'L{lc}/plans/setup.json').read_text())
        expected=set(setup['plans'])|set(setup['failures'])
        for source in ('A','B'):
            for case in range(count):
                case_dir=root/f'L{lc}/{cohort}/c{case:03d}'
                record=json.loads((case_dir/f'{source}_scores.json').read_text())
                assert len(record)==len(expected) and {r['method'] for r in record}==expected
                arrays=np.load(case_dir/f'{source}_coefficients.npz')
                expected_keys=set()
                for row in record:
                    method=row['method'];assert row==lookup[(lc,source,case,method)]
                    if row['status']=='setup-failure':continue
                    for ti in range(6):
                        key=f'{method}__t{ti}';expected_keys.add(key);values=arrays[key]
                        if row['e_t'][ti] is None:
                            assert values.shape==(0,0);continue
                        expected_hashes=row['coefficient_sha256'][ti]
                        same=row['repetition_deduplicated'][ti]
                        assert len(values)==(1 if same else len(expected_hashes))
                        for rep,want in enumerate(expected_hashes):
                            c=values[0 if same else rep]
                            assert c.dtype==np.float64 and np.isfinite(c).all()
                            assert digest(c)==want,(key,'coefficient hash')
                            checked_coefficients+=1
                            if row['repetitions'][ti]:
                                assert hashlib.sha256(c.tobytes()).hexdigest()==row['repetitions'][ti][rep]['coefficient_sha256']
                                checked_timed_invocations+=1
                        assert len(row['scores'][ti])==len(expected_hashes)
                        assert row['e_t'][ti]==row['scores'][ti][-1]['error']
                        assert all(np.isfinite(v) for s in row['scores'][ti] for v in s.values())
                    if row['status']=='complete':assert row['e']==max(row['e_t'][1:])
                assert set(arrays.files)==expected_keys,('unexpected or missing coefficient keys',case_dir,source)
                if phase=='val' and case==0:
                    fields=np.load(case_dir/f'{source}_selected_fields.npz')
                    for key in fields.files:
                        method,t=key.rsplit('__t',1);ti=int(t)
                        row=lookup[(lc,source,case,method)]
                        same=row['repetition_deduplicated'][ti];values=fields[key]
                        for rep,want in enumerate(row['field_sha256'][ti]):
                            field=values[0 if same else rep]
                            assert digest(field)==want,(key,'actual field hash')
                            if row['repetitions'][ti]:
                                assert hashlib.sha256(field.tobytes()).hexdigest()==row['repetitions'][ti][rep]['field_sha256']
                            checked_fields+=1
    assert len(rows)==sum((len(json.loads((root/f'L{lc}/plans/setup.json').read_text())[part])
                          for lc in (16,32) for part in ('plans','failures')))*count*2
    result=dict(status='pass',phase=phase,rows=len(rows),coefficient_invocations_checked=checked_coefficients,
        timed_invocations_checked=checked_timed_invocations,representative_field_invocations_checked=checked_fields,
        scope='Artifact identities and stored score consistency; independent numerical replay is separate.')
    (root/'ARTIFACT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
