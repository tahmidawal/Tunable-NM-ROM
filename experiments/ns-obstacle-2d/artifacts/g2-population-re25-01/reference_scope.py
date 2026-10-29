"""Explicit user-authorized scope; this module never admits a reference."""
FD1_GATE='G2_FIXED_FINITE_CHANNEL_V1'
FD1_CONTRACT_SHA256='31ec75738dc47f8d960acf2f51dd355caf497a2681a3d499cc00faa138b28f76'
FD1_SCOPE={'scope_contract_sha256':FD1_CONTRACT_SHA256,'amendment_id':'FD1-2026-10-28',
 'original_G2_status':'not_admitted_domain_sensitivity_failed','domain_sensitivity_status':'failed',
 'domain_independence_admitted':False,'required_scope':'prescribed original fixed finite DFG channel and boundary conditions',
 'geometry_population_and_numerical_thresholds_unchanged':True}

def validate_scope(scope):
    if not isinstance(scope,dict) or any(type(scope.get(k)) is not type(v) or scope.get(k)!=v for k,v in FD1_SCOPE.items()):
        raise ValueError('exact FD1 qualification scope required; original domain failure remains')
    return dict(scope)

def validate_geometry(cfg):
    fixed={'kind':'dfg','outlet':'gradient','xmin':0.,'xmax':2.2,'H':.41,'center':[.2,.2],'radius':.05,'U':.2,'D':.1}
    if cfg.get('scope_contract_sha256')!=FD1_CONTRACT_SHA256 or any(cfg.get(k)!=v for k,v in fixed.items()):
        raise ValueError('FD1 is restricted to the original prescribed finite channel')
