"""Post-freeze rendering fix validated on public development runs before sealed results.

Full-field and decode queries only have a confirmation block. Preserve both measured
blocks, explicitly record missing pairs, and enforce the original gate wherever it can
be evaluated. This file does not change the frozen scientific choices or classifiers.
"""
from collections import defaultdict
import math
from statistics import median
import summarize_comparators as source

def timing(rep,cfg):
    panel=rep['timing']
    fixed=cfg['timing']
    cases=rep['cases'][:fixed['cases']]
    if panel['cases']!=cases:
        raise ValueError('Incorrect timing cases')
    def phase(key,n):
        groups=defaultdict(dict)
        for row in panel[key]:
            subject=row['kind']+'|'+row['name']
            k=(row['case'],row['rep'])
            t=row['seconds']
            if k in groups[subject] or not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0:
                raise ValueError('Invalid timing repetition')
            groups[subject][k]=t
        result={}
        for subject,values in groups.items():
            if set(values)!={(i,r) for i in cases for r in range(n)}:
                raise ValueError('Missing timing values')
            result[subject]={i:median(values[i,r] for r in range(n)) for i in cases}
        return result
    main=phase('invocations',fixed['reps'])
    confirm=phase('confirm_invocations',fixed['confirm_reps'])
    if set(main)!=set(panel['summary']) or set(confirm)!=set(panel['confirm_summary']):
        raise ValueError('Inconsistent per-subject timing inventory')
    fold={}
    admitted={}
    for k in sorted(set(main)&set(confirm)):
        m=panel['summary'][k]['median_ms']
        c=panel['confirm_summary'][k]['median_ms']
        fold[k]=max(c/m,m/c)
        admitted[k]=bool(rep['gates']['G_time']['outputs_pass'] and rep['gates']['G_time']['drift_pass'] and fold[k]<=1.10)
    return dict(raw_main_repetitions=len(panel['invocations']),raw_confirm_repetitions=len(panel['confirm_invocations']),
                gates=rep['gates']['G_time'],main_confirm_ratio_fold=fold,dual_block_subject_qualified=admitted,
                main_only_subjects=sorted(set(main)-set(confirm)),confirmation_only_subjects=sorted(set(confirm)-set(main)),
                main_per_case_median_seconds=main,confirm_per_case_median_seconds=confirm)

if __name__=='__main__':
    source.timing=timing
    source.main()
