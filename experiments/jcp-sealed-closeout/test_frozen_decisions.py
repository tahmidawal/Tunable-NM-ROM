import copy
import itertools
import unittest

import frozen_decisions as fc


def panel(d=2):
    cohort, n = fc.POPULATION[d]
    cases = list(range(n))
    fib = [("fib610","fibonacci"), ("fib987","fibonacci"), ("fib1597","fibonacci"), ("fib2584","fibonacci")]
    gl = [("gauss45","gauss"), ("gauss60","gauss"), ("gauss80","gauss")]
    if d == 3:
        fib = [("lat3072","lattice"), ("lat4096","lattice"), ("lat6144","lattice"), ("lat8192","lattice")]
        gl = [("gl20","gauss"), ("gl22","gauss"), ("gl24","gauss")]
    cfg = dict(dim=d, mesh=fc.MESHES[d][0], cohort=cohort, cases=cases,
               arms=[dict(name=n,family=f,rule=n) for n,f in fib+gl], controls=[],
               foms=[dict(name="classical")], conv_arm=gl[-1][0], anchor_check_arm=fib[-1][0],
               expected_store_version=fc.STORE)
    spec,_,sizes = fc.sizes_and_families(cfg)
    def scores():
        return {k:dict(e=1.,e_t=[1.]*5,eligible=True) for k in fc.KINDS}
    rows = []
    for name,c in itertools.product(spec,cases):
        hist = (dict(exits=dict(budget=0,tol=5,tiny_step=0,damping_limit=0,stationary=0))
                if d==2 else dict(reasons={"0":0,"1":5,"2":0,"3":0,"4":0}))
        rows.append(dict(arm=name, family=spec[name]["family"], rule=name, case=c, m=sizes.get(name),
                         finite=True, hard_fail=False, steps=5, nonconv_steps=0, dist_conv=0., **hist, **scores()))
    foms = [dict(fom="classical",case=c,converged=True,**scores()) for c in cases]
    refs = {f"{c}|{k}": dict(outcome="accepted",eligible=True,eps_ref=0.,n0=1.) for c,k in itertools.product(cases,fc.KINDS)}
    gates={k:dict(passed=True) for k in ("G_R","G_coh","G_ref","G_lat","G_score_vs_refstore")}
    gates["G_ref"].update(loaded=2*n,store_version=fc.STORE)
    for k in ("G1_derivative_vs_fd","G2_meshnodes_offmesh_vs_tensor","G3_jacobian_vs_jacfwd","G3_adv_half_Jc"):
        gates[k]=0.
    gates["G4_solver_vs_vendor"]=dict(its_equal=True,fields=0.,coefs=0.)
    digest="2"*64
    rep=dict(config=copy.deepcopy(cfg),mesh=cfg["mesh"],cases=cases,precision="highest",x64=True,complete=True,
             backend="gpu",store_version=fc.STORE,job_id="11111",gpu="synthetic",gates=gates,rows=rows,
             fom_rows=foms,references=refs)
    audit=dict(passed=True, dim=d, mesh=cfg["mesh"], job_id="11111", result_sha256=digest,
               checks=dict(G_audit=dict(passed=True),fault_injections=dict(passed=True)))
    return rep,cfg,audit,digest


def result(bundle,kind="ST",name="fib987"):
    return fc.classify(*bundle)["kinds"][kind]["rules"][name]["decision"]


def exclude(rep,c,kind):
    rep["references"][f"{c}|{kind}"].update(outcome="order-fail",eligible=False,eps_ref=None)
    for r in rep["rows"]+rep["fom_rows"]:
        if r["case"]==c:
            r[kind]["eligible"]=False


def setscore(row,kind,value):
    row[kind].update(e=value,e_t=[value]*5)


class Decisions(unittest.TestCase):
    def test_all_pass_independent_dimensions(self):
        for d,names in fc.RULES.items():
            for name in names:
                self.assertEqual(result(panel(d),name=name),"PASS")

    def test_interval_uncertainty_failure_both_sides_including_equality(self):
        self.assertEqual(fc.band_interval(1.045,1.055,1.,1.),"UNRESOLVED")
        self.assertEqual(fc.band_interval(.94,.949,1.,1.),"FAIL")
        self.assertEqual(fc.band_interval(1.051,1.055,1.,1.),"FAIL")
        self.assertEqual(fc.band_interval(.95,1.05,1.,1.),"PASS")
        self.assertEqual(fc.band_interval(1.,1.,0.,1.),"UNRESOLVED")

    def test_higher_suffix_failure_or_lower_size_pass(self):
        b=panel()
        # Four rungs: change intermediate, preserve the two valid large-rung tails
        # by inserting a new rung below the two unchanged valid top sizes.
        cfg=b[1]
        extra=dict(name="fib1000",family="fibonacci",rule="fib1000")
        cfg["arms"].append(extra)
        b[0]["config"]=copy.deepcopy(cfg)
        higher=[copy.deepcopy(r) for r in b[0]["rows"] if r["arm"]=="fib987"]
        for r in higher:
            r.update(arm="fib1000",rule="fib1000",m=1000)
            setscore(r,"ST",1.07)
        b[0]["rows"].extend(higher)
        self.assertEqual(result(b),"FAIL")
        for r in higher:
            setscore(r,"ST",1.)
        self.assertEqual(result(b),"PASS") # fib610 passes too

    def test_case_with_bad_reference_still_invalidates_rollout(self):
        b=panel()
        exclude(b[0],0,"ST")
        r=next(r for r in b[0]["rows"] if r["arm"]=="fib987" and r["case"]==0)
        r["finite"]=False
        setscore(r,"ST",None)
        self.assertEqual(result(b),"FAIL")

    def test_uncertain_not_failed_and_independent_secondary(self):
        b=panel()
        for k,v in b[0]["references"].items():
            if k.endswith("|ST"):
                v["eps_ref"]=0.026
        self.assertEqual(result(b),"UNRESOLVED")
        self.assertEqual(result(b,"S"),"PASS")

    def test_bad_anchor_or_own_tail_withholds(self):
        for defect in ("anchor","tail"):
            b=panel()
            if defect=="anchor":
                next(r for r in b[0]["rows"] if r["arm"]=="fib2584")["dist_conv"]=.01
            else:
                for r in b[0]["rows"]:
                    if r["arm"]=="fib1597":
                        setscore(r,"ST",1.08)
            self.assertEqual(result(b),"UNRESOLVED")

    def test_eligible_epsilon_missing_no_cross_kind_suppression(self):
        b=panel()
        b[0]["references"]["0|ST"]["eps_ref"]=None
        self.assertEqual(result(b),"UNRESOLVED")
        self.assertEqual(result(b,"S"),"PASS")

    def test_empty_eligible_and_dimension_denominators(self):
        b=panel(3)
        for c in b[1]["cases"]:
            exclude(b[0],c,"ST")
        p=fc.classify(*b)
        self.assertEqual(p["kinds"]["ST"]["reason"],"no eligible references")
        self.assertEqual(p["kinds"]["S"]["rules"]["gl20"]["decision"],"PASS")

    def test_exact_reference_resolution_and_eligibility(self):
        b=panel()
        for r in b[0]["rows"]:
            if r["arm"]=="fib987" and r["case"]==0:
                setscore(r,"ST",1.25)
        b[0]["references"]["0|ST"]["eps_ref"]=.125
        x=fc.classify(*b)["kinds"]["ST"]["rules"]["fib987"]["paired_vs_conv"]
        self.assertEqual(x["counts"]["uncertainty_overlap"],128)
        self.assertEqual(x["counts"]["higher_error"],0)

    def test_incomplete_coverage_duplicates_evidence_binding_gates(self):
        for issue in ("duplicate","missing","audit","gate","wrongcases","noFOM","falseeligibility"):
            b=panel()
            if issue=="duplicate":
                b[0]["rows"].append(b[0]["rows"][0])
            if issue=="missing":
                b[0]["rows"].pop()
            if issue=="audit":
                b[2]["result_sha256"]="1"*64
            if issue=="gate":
                b[0]["gates"]["G_R"]["passed"]=False
            if issue=="wrongcases":
                b[0]["cases"]=[1,2]
            if issue=="noFOM":
                b[0]["fom_rows"].pop()
            if issue=="falseeligibility":
                b[0]["references"]["0|ST"]["eligible"]=False
            self.assertEqual(result(b),"UNRESOLVED",issue)

    def test_three_d_hard_failure_or_nonconvergence_threshold(self):
        b=panel(3)
        r=next(r for r in b[0]["rows"] if r["arm"]=="lat4096")
        r["hard_fail"]=True
        r["reasons"].update({"3":1,"1":4})
        r["nonconv_steps"]=1
        self.assertEqual(result(b,name="lat4096"),"FAIL")
        r["hard_fail"]=False
        r["reasons"].update({"3":0,"2":3,"1":2})
        r["nonconv_steps"]=3
        self.assertEqual(result(b,name="lat4096"),"PASS") # 3/320 <= 0.01
        r["reasons"].update({"2":4,"1":1})
        r["nonconv_steps"]=4
        self.assertEqual(result(b,name="lat4096"),"FAIL")

    def test_aggregate_fail_dominates_missing_requires_every_mesh(self):
        results=[]
        for d, meshes in fc.MESHES.items():
            p=fc.classify(*panel(d))
            for m in meshes:
                results.append(dict(p,mesh=m))
        combined=fc.aggregate(results)
        self.assertTrue(all(k["decision"]=="PASS" for n in combined.values() for k in n.values()))
        self.assertEqual(fc.aggregate(results[:-1])["gl20"]["ST"]["decision"],"UNRESOLVED")
        results[-2]["kinds"]["ST"]["rules"]["gl20"]["decision"]="FAIL"
        self.assertEqual(fc.aggregate(results[:-1])["gl20"]["ST"]["decision"],"FAIL")
        with self.assertRaises(ValueError):
            fc.aggregate(results+[results[0]])


if __name__ == "__main__":
    unittest.main()
