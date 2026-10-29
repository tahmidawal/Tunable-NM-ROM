"""Array-only calibration selection/timing/readout mechanics; no native evidence."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import numpy as np
from fom_calibrate2d import choose
from fom_readout2d import readout_control
from timing import calibration_cpu
from pilot2d import DEV_RE


def main():
    candidates=[{'id':'A'},{'id':'B'}];rows=[]
    for c,t in [('A',2.),('B',1.)]:
        for re in DEV_RE:rows.append({'candidate':c,'Re':re,'contract':'fixed_probes','eligible':True,'elapsed_seconds':[t]*7})
    assert choose(candidates,rows,'fixed_probes')[0]=='B'
    rows[-1]['eligible']=False;assert choose(candidates,rows,'fixed_probes')[0]=='A'
    assert choose(candidates,rows[2:],'fixed_probes')[0] is None
    count=0
    def call():
        nonlocal count
        count+=1
        if count==4:
            exc=RuntimeError('synthetic failure');exc.invocation={'output_contract':'fixed_probes','outputs':None,'outputs_available':False,
                'work':{'status':'failed'},'evidence':{'state':np.array([count])}};raise exc
        return {'output_contract':'fixed_probes','outputs':{'u':np.array([count])},'work':{'status':'root_converged'}}
    timed=calibration_cpu(call,'fixed_probes',1,burn_seconds=0)
    assert len(timed['actual_invocation_outputs'])==7
    assert sum(row['invocation'].get('query_failed',False) for row in timed['actual_invocation_outputs'])==1
    points=np.array([[1.,1.],[2.,1.]]);gradient=np.array([[1.,2.],[3.,4.]])
    u=lambda mip:np.asarray(mip);p=lambda mip:float(mip[0]-mip[1])
    coeff=np.zeros(2);gf=NS(vec=NS(FV=lambda:NS(NumPy=lambda:coeff)),components=(u,p))
    ng=NS(GridFunction=lambda X:gf,CF=lambda expression:lambda mip:np.asarray(mip),x=0,y=1,
          Grad=lambda function:lambda mip:gradient,BoundaryFromVolumeCF=lambda function:function)
    fields=NS(native_owners=(None,[(lambda ip:ip,list(points),None),points,(lambda ip:ip,list(points),None),points],list(points),points,points))
    native=NS(ng=ng,X=None,cfg={'U':1.,'D':1.},evaluators={'full_velocity_pressure_fields':fields},
        diagnostics_from_evidence=lambda evidence:{'boundary_labels':np.array(['obstacle','obstacle']),
            'fluid_normals':np.array([[1.,0.],[1.,0.]]),'boundary_weights':np.array([.5,.5]),
            'velocity_gradient':np.repeat(gradient[None],2,axis=0),'boundary_velocity':points,
            'boundary_velocity_gradient':np.repeat(gradient[None],2,axis=0),'boundary_pressure':np.array([0.,1.])})
    candidate=NS(native=native,points=points,release={'field_rule_mode':'native_exact'})
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for index,(contract,outputs) in enumerate([
            ('full_velocity_pressure_fields',{'velocity':points,'pressure':np.array([0.,1.])}),
            ('fixed_probes',{'velocity':points,'pressure':np.array([0.,1.])}),
            ('integrated_forces',{'force_coefficient':np.array([.8,-.5])})]):
            out=root/str(index);out.mkdir()
            invocation={'output_contract':contract,'outputs':outputs,'evidence':{'state':np.zeros(2),'Re':20.}}
            result=readout_control(candidate,invocation,root,out)
            assert result['passed'],result
        candidate.release['field_rule_mode']='certified_requested_points'
        fields.native_owners=(None,[(lambda ip:ip,list(points[::-1]),None),points,(lambda ip:ip,list(points),None),points],list(points),points,points)
        np.savez_compressed(root/'normal_orientation.mapping.batch.npz',native_to_requested=np.array([1,0]))
        out=root/'permutation';out.mkdir()
        invocation={'output_contract':'full_velocity_pressure_fields','outputs':{'velocity':points,'pressure':np.array([0.,1.])},'evidence':{'state':np.zeros(2),'Re':20.}}
        assert readout_control(candidate,invocation,root,out)['passed']
        out=root/'failed';out.mkdir();invocation['outputs']['pressure']=np.array([1.,2.])
        assert not readout_control(candidate,invocation,root,out)['passed']
        assert (out/'scalar_readout.npz').is_file()
        out=root/'nonfinite-gap';out.mkdir();invocation['outputs']['pressure']=np.array([1e308,1e308])
        with np.errstate(over='ignore',invalid='ignore'):
            assert not readout_control(candidate,invocation,root,out)['passed']
        assert (out/'scalar_readout.json').is_file()
    print('PASS dev-only selection rejects partial/failure; seven actual calibration records; scalar readout/permutation/force/failure retention; synthetic only')


if __name__=='__main__':main()
