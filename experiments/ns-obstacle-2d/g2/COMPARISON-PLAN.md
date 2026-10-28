# Common physical-volume field comparison candidate

`compare_fields.py` is unexecuted source for the eventual reference ladder. It is separate from the frozen native01 producer inputs and creates no admission record.

For the h ladder, use the finest candidate's complete positive mapped volume rule for both successive differences. Evaluate each actual binary FE solution at every requested physical coordinate, retaining requested and reconstructed coordinates and a validity mask. An invalid point aborts the entire comparison after saving evidence. No point is dropped, moved, replaced with a nearest node or assigned an extrapolated field. Thus a geometric overlap failure remains a failure requiring an independently reviewed remedy. Every candidate must first have a valid source-bound binary fresh-process reload certificate.

The saved arrays support the fixed velocity and pressure metrics

$$e_u=\frac{\sqrt{\sum_q w_q\|u_q-u_{q,ref}\|^2}}{\sqrt{\sum_q w_q\|u_{q,ref}\|^2}},\qquad e_p=\frac{\sqrt{\sum_q w_q(p_q-p_{q,ref})^2}}{U^2\sqrt{\sum_qw_q}}.$$

Each numerator, denominator and common-rule volume is retained. Pressure is never shifted. For upstream/downstream extensions the full original-domain rule is used, and pressure field differences are labelled ineligible for the domain gate; valid exact common-pole pressure differences are compared instead. A different outlet stress model remains a separately named problem. All existing array/point caps remain enforced, so this source cannot silently run the proposed quarter-size ladder before its resource review.

Future explicit comparison rows will distinguish the two h differences, the independent p/geometry control, solver tightening, quadrature tightening and each domain/outlet case. The final admission aggregator must combine these field arrays with all unchanged per-case physical gates and force/probe diagnostics. It must retain every uncertainty constituent, reject nondecreasing/unresolved spatial differences, and use the declared conditional uncertainty recipe; a successful field evaluation alone does not satisfy G2. The precise common rule and source-bound candidate manifests must freeze before execution. No comparison plan with data paths has been created or executed yet.

## Glossary

FE: finite element. h ladder: successively finer meshes. p control: a comparison at higher polynomial degree. Common rule: one set of physical points and positive volume weights used by every compared field. Validity mask: whether each original requested point was located inside that candidate's actual FE domain. Binary reload certificate: measured agreement after loading exact archived mesh and solution bytes in a new process. Conditional uncertainty: a refinement-based estimate, not a rigorous bound. G2: physical reference admission after every required control and independent review.
