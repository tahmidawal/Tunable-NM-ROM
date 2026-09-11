"""Read-only outgoing-wave moment diagnostic from archived fields/operators.

No reference solver, model, training data or online query is modified or rerun.
The exact discrete moment is mass integral(v) + c*boundary integral(u), where the
boundary quadrature equals mass times the recorded unit damping ratio.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_dynamics import mass,write,digest


def main(record):
    native=record/'cluster/out/pilot';out=record/'analysis';data=json.loads((native/'result.json').read_text());cfg=data['config']
    rows=[];banks=[];reference=[]
    for n in cfg['meshes']:
        with np.load(native/f'mesh_absorbing_{n}.npz') as f:mesh={k:f[k] for k in f.files}
        g=mesh['g'];m=mass(n,'absorbing');ratio=np.zeros((n+1,n+1));ratio[[0,-1],:]+=2*n;ratio[:,[0,-1]]+=2*n
        boundary=m*ratio
        # The one-dimensional summation identity implies its tensor-sum identity.
        h=1/n;one_mass=np.full(n+1,h);one_mass[[0,-1]]*=.5
        edge=np.diff(np.eye(n+1),axis=0)/np.sqrt(h)
        one_laplacian=(edge.T@edge)/one_mass[:,None]
        full_moment_row=one_mass@one_laplacian
        np.testing.assert_allclose(full_moment_row,0,atol=1e-12,rtol=0)
        np.testing.assert_allclose(g.T@(m.ravel()[:,None]*g),np.eye(g.shape[1]),atol=1e-11,rtol=0)
        np.testing.assert_allclose(mesh['mass'].reshape(m.shape),m,atol=0,rtol=0)
        np.testing.assert_allclose(g.T@((boundary.ravel())[:,None]*g),mesh['damping'],atol=2e-10)
        ell=g.T@m.ravel();delta=g.T@boundary.ravel();constant_projection=g@ell
        defect=np.sqrt(np.sum(m.ravel()*(1-constant_projection)**2))/np.sqrt(m.sum())
        stiffness_row=ell@mesh['stiffness'];damping_row=delta-ell@mesh['damping']
        banks.append(dict(intervals=n,mass_area=float(m.sum()),boundary_measure=float(boundary.sum()),constant_test_projection_error=float(defect),
            one_dimensional_full_operator_moment_row_max=float(np.max(abs(full_moment_row))),
            exact_constant_direction_at_1e_minus10=bool(defect<1e-10),stiffness_moment_row_norm=float(np.linalg.norm(stiffness_row)),
            damping_moment_row_norm=float(np.linalg.norm(damping_row))))
        np.savez_compressed(out/f'moment_bank_{n}.npz',mass_moment=ell,boundary_moment=delta,stiffness_moment_row=stiffness_row,damping_moment_row=damping_row)
        for ci in cfg['validation_indices']:
            selected=[r for r in data['invocations'] if r['boundary']=='absorbing' and r['intervals']==n and r['case']==ci and r['repetition']==0 and (r['method'] in ('affine16','affine32','full64') or r['method']=='rom' and r['setting']==cfg['rom_dts'][0])]
            c=selected[0]['parameters'][5]
            with np.load(native/f'samegrid_absorbing_{n}_{ci}.npz') as f:u,v,u0,v0=f['u'],f['v'],f['u0'],f['v0']
            ref=np.sum(m*v+c*boundary*u,axis=(-2,-1));input_moment=float(np.sum(m*v0+c*boundary*u0))
            ap=g.T@(m.ravel()*u0.ravel());bp=g.T@(m.ravel()*v0.ravel());projected=float(ell@bp+c*delta@ap)
            scale=selected[0]['same_grid_discrepancy']['energy_state']['initial_scale']*np.sqrt(m.sum())
            reference.append(dict(intervals=n,case=ci,speed=c,input_moment=input_moment,reference_initial=float(ref[0]),
                reference_max_drift=float(np.max(abs(ref-ref[0]))),unrestricted_projected_initial=projected,
                projection_initial_defect=projected-input_moment,fixed_scale=scale))
            traces=dict(reference_moment=ref,reference_mean_displacement=np.sum(m*u,axis=(-2,-1))/m.sum(),times=np.arange(49)*cfg['observation_dt'])
            for row in selected:
                with np.load(native/(row['invocation_id']+'.npz')) as f:a,b=f['coefficients'],f['velocity_coefficients']
                moment=b@ell+c*(a@delta);mean=a@ell/m.sum()
                direct=[]
                for ti in (0,len(a)-1):
                    uf,vf=(g@a[ti]).reshape(m.shape),(g@b[ti]).reshape(m.shape)
                    direct.append(abs(float(np.sum(m*vf+c*boundary*uf))-moment[ti]))
                maximum_difference=max(direct);assert maximum_difference<1e-12
                method=row['method'];traces[method+'_moment']=moment;traces[method+'_mean_displacement']=mean
                rows.append(dict(intervals=n,case=ci,method=method,setting=row['setting'],
                    initial_moment=float(moment[0]),initial_error=float(moment[0]-input_moment),
                    head_vs_unrestricted_projection_initial_difference=float(moment[0]-projected),
                    maximum_drift_from_own_initial=float(np.max(abs(moment-moment[0]))),final_drift=float(moment[-1]-moment[0]),
                    final_error=float(moment[-1]-ref[-1]),maximum_error=float(np.max(abs(moment-ref))),
                    maximum_error_fixed_scaled=float(np.max(abs(moment-ref))/scale),final_error_fixed_scaled=float((moment[-1]-ref[-1])/scale),
                    mean_displacement_final=float(mean[-1]),reference_mean_displacement_final=float(traces['reference_mean_displacement'][-1]),
                    direct_full_field_moment_difference=float(maximum_difference)))
            np.savez_compressed(out/f'moment_traces_{n}_{ci}.npz',**traces)
    result=dict(status='Read-only postprocessing of completed audited dynamics02 artifacts; mechanism diagnostic, not a causal intervention.',
        native_sha256=digest(native/'result.json'),source_commit=data['provenance']['source_commit'],
        operator_definition='u_dot=v; v_dot=-c^2 L_N u-c D_unit v; M is tensor-product trapezoidal mass on [0,1]^2; D_unit adds2/h on each outgoing edge, including both contributions at corners; B=M D_unit; 1^T M L_N=0.',
        moment_definition='I = sum(mass*v) + c*sum(boundary_weights*u); boundary_weights = mass*unit_damping_ratio; edges use trapezoidal weights and corner contributions from both edges.',
        scale_definition='sqrt(initial phase energy times2) times sqrt(domain mass area), fixed per case; no division by the near-zero invariant.',
        bank_diagnostics=banks,references=reference,methods=rows,
        interpretation='Initial projection/fit error and subsequent moment drift are reported separately. Lack of the constant test direction and nonzero generator rows establish a missing discrete invariance property. Their correlation with late physical errors does not establish that they cause all of those errors. No constant-mode or moment-correction control was run.')
    write(out/'absorbing-moment.json',result)
    lines=['# Absorbing-wave global moment diagnostic','',
        'This generated note analyzes the previously archived dynamics pilot without changing a model or rerunning a solver. It is a provisional mechanism diagnostic; no causal correction experiment is included.','',
        'For the homogeneous outgoing boundary, the actual discrete operator satisfies', '',
        '$$I(t)=\\langle 1,v(t)\\rangle_M+c\\langle 1,u(t)\\rangle_B,\\qquad dI/dt=0.$$', '',
        'On the unit square, $h=1/N$ and $M$ is the tensor product of trapezoidal weights. The actual equations are $\\dot u=v$, $\\dot v=-c^2L_Nu-cD_0v$. Each outgoing edge contributes $2/h$ to the diagonal of $D_0$; corners receive both contributions, and $B=MD_0$. The positive Laplacian has one-dimensional endpoint rows $2(u_0-u_1)/h^2$ and $2(u_N-u_{N-1})/h^2$, with the usual centered interior rows. Therefore $1^TML_N=0$, and the speed factor in the boundary term is exactly $c$. The one-dimensional summation identity is checked independently, and its tensor sum gives the two-dimensional identity.','',
        'The boundary weights are the mass weights times the unit-speed damping ratio, including both corner contributions. The reference moment, bank-projected initial moment, head-fitted initial moment and subsequent autonomous drift are distinct measurements.','',
        '| Intervals | Constant test projection error | Contains constant to stated tolerance | Stiffness moment row norm | Damping moment row norm |','|---|---:|---|---:|---:|']
    for r in banks:lines.append(f"| {r['intervals']} | {r['constant_test_projection_error']:.12g} | {r['exact_constant_direction_at_1e_minus10']} | {r['stiffness_moment_row_norm']:.12g} | {r['damping_moment_row_norm']:.12g} |")
    lines += ['', 'The exact-constant diagnostic tolerance is $10^{-10}$. The retained bank obeys $G^TMG=I$. Let $\\ell=G^TM1$, $d=G^TB1$, $K=G^TML_NG$, and $D=G^TBG$. The full-bank equations are $\\dot a=b$, $\\dot b=-c^2Ka-cDb$. Thus $\\dot I=-c^2\\ell^TKa+c(d^T-\\ell^TD)b$. The table gives Euclidean coefficient-row norms before the recorded case speed factors are applied. Nonzero rows mean that this reduced generator does not preserve the original discrete moment for general states. This is an algebraic property, not proof that it explains every observed prediction error.','',
        '| Intervals | Case | Input moment | Reference maximum drift | Unrestricted bank-projected initial moment |','|---|---:|---:|---:|---:|']
    for r in reference:lines.append(f"| {r['intervals']} | {r['case']} | {r['input_moment']:.12g} | {r['reference_max_drift']:.12g} | {r['unrestricted_projected_initial']:.12g} |")
    lines += ['', '| Intervals | Case | Method | Fitted initial error | Maximum drift from its own initial moment | Final moment error | Maximum error / fixed scale |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['intervals']} | {r['case']} | {r['method']} | {r['initial_error']:.12g} | {r['maximum_drift_from_own_initial']:.12g} | {r['final_error']:.12g} | {r['maximum_error_fixed_scaled']:.12g} |")
    lines += ['', 'Every coefficient-based initial/final moment is checked against reconstructed full-grid fields. The fixed denominator is $\\sqrt{2E(0)}\\sqrt{\\langle1,1\\rangle_M}$, not the almost-zero moment itself. Raw traces, moment operators and displacement means are preserved beside this note. The original physical error metrics and runtime claims are unchanged.','',
        'A future control may separate adding the exact constant test direction from correcting the fitted initial moment. Neither control has been run here, and the current larger-head pilot is unchanged. A corrected moment is not by itself a guarantee of local field or energy-state accuracy.','',
        '## Plain-language glossary','',
        '- **Moment / invariant:** a global weighted combination of the fields / a quantity the exact discrete equations keep constant.',
        '- **Mass / boundary weights:** area integration weights / outgoing-boundary integration weights, including corners.',
        '- **Bank / constant test / projection:** learned spatial span / testing against the spatial constant function / closest field in that span under the mass norm.',
        '- **Generator row / drift / fixed scale:** algebraic derivative of the moment under reduced dynamics / change from its own starting value / a nonvanishing initial normalization.',
        '- **Unrestricted / head fitted / reference:** any coefficient in the learned span / coefficient constrained by the decoder / retained full solver trajectory.',
        '- **Causal control / artifact / hash:** an intervention that isolates a mechanism / stored numerical output / content fingerprint.','']
    (out/'ABSORBING-MOMENT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(bank=banks,references=reference,methods=rows),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);main(ap.parse_args().record)
