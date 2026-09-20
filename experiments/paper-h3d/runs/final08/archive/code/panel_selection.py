"""Apply only prospectively frozen per-mesh control/transfer selections.

Development configurations omit these maps and retain the complete sweep.
"""

def filter_methods(methods,metadata,n,cfg):
    methods=dict(methods);metadata=dict(metadata)
    if 'operator_methods_by_mesh' in cfg:
        selected=cfg['operator_methods_by_mesh'][str(n)]
        assert len(selected)==len(set(selected))==len(cfg['frozen_operators'])
        available={name for name,meta in metadata.items() if meta['kind']=='neural_operator'}
        assert set(selected)<=available
        for base in cfg['frozen_operators']:
            assert sum(name==base or name.startswith(base+'_') for name in selected)==1
        if n==cfg['train_intervals']:assert set(selected)==set(cfg['frozen_operators'])
        for name in available-set(selected):methods.pop(name);metadata.pop(name)
    if 'cg_methods_by_mesh' in cfg and cfg.get('include_linear_controls',True):
        selected=cfg['cg_methods_by_mesh'][str(n)]
        available={name for name in metadata if name.startswith('fom_cn_cg_')}
        assert selected and len(selected)==len(set(selected)) and set(selected)<=available
        for name in available-set(selected):methods.pop(name);metadata.pop(name)
        assert 'dst_exact' in methods,'the independent direct control remains mandatory'
    assert set(methods)==set(metadata)
    return methods,metadata


def validate_against_development(cfg,record):
    """A final selection cannot name an unmeasured development variant."""
    assert not record['final_cohort_opened'] and record['complete']
    for n in cfg['evaluation_intervals']:
        mesh=next(mesh for mesh in record['meshes'] if mesh['intervals']==n)
        dummy={name:None for name in mesh['methods']}
        filter_methods(dummy,mesh['methods'],n,cfg)
