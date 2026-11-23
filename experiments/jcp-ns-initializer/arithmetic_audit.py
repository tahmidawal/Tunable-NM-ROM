"""Independent cluster-host arithmetic audit, never a GPU/physics solver.

All inputs are exact stored f64 values. Extra precision only diagnoses objective
rankings; it does not replace either required f64 stationarity check.
"""
from decimal import Decimal,localcontext
import time
import numpy as np

PRECISION=50
EPS64=np.finfo(np.float64).eps


class NetworkAudit:
    def __init__(self,params):
        self.long={k:np.asarray(v,dtype=np.longdouble) for k,v in params.items()}
        self.decimal=None
        self.params={k:np.asarray(v,dtype=np.float64) for k,v in params.items()}
        if not all(np.all(np.isfinite(v)) for v in self.params.values()):
            raise RuntimeError('independent arithmetic audit received nonfinite frozen parameters')
        if np.finfo(np.longdouble).nmant<63:
            raise RuntimeError('independent longdouble audit requires at least63 explicit mantissa bits')

    def long_objective(self,code,target):
        p=self.long;z=np.asarray(code,dtype=np.longdouble)
        x=z@p['W1']+p['b1'];x=x/(1+np.exp(-x))
        x=x@p['W2']+p['b2'];x=x/(1+np.exp(-x))
        r=x@p['W3']+p['b3']+z@p['Ws']-np.asarray(target,dtype=np.longdouble)
        return np.dot(r,r)/2

    def decimal_objective(self,code,target):
        if self.decimal is None:
            self.decimal={k:np.asarray([Decimal.from_float(float(x)) for x in v.ravel()],dtype=object).reshape(v.shape)
                          for k,v in self.params.items()}
        with localcontext() as ctx:
            ctx.prec=PRECISION
            p=self.decimal;z=np.asarray([Decimal.from_float(float(x)) for x in code],dtype=object)
            x=z@p['W1']+p['b1'];x=np.asarray([v/(1+(-v).exp()) for v in x],dtype=object)
            x=x@p['W2']+p['b2'];x=np.asarray([v/(1+(-v).exp()) for v in x],dtype=object)
            r=x@p['W3']+p['b3']+z@p['Ws']-np.asarray([Decimal.from_float(float(x)) for x in target],dtype=object)
            return sum((v*v for v in r),Decimal(0))/2

    def compare(self,controls,returned,target,direct_guard):
        begin=time.perf_counter()
        codes=[np.asarray(c['code']) for c in controls]+[np.asarray(returned)]
        if not np.all(np.isfinite(target)) or not all(np.all(np.isfinite(z)) for z in codes):
            return {'passed':False,'executed':False,'reason':'nonfinite target or endpoint'}
        costs=[self.long_objective(z,target) for z in codes]
        old=min(costs[:-1]);band=np.longdouble(1024)*np.longdouble(EPS64)*max(old,np.longdouble(1e-300))
        delta=costs[-1]-old
        long_pass=bool(delta<=band)
        # Trigger is fixed before outcomes and solely numerical, not physical accuracy.
        decimal_required=bool(not direct_guard or abs(delta-band)<=.1*band)
        result={'precision_scope':'independent cluster-host arithmetic audit; primary solves remain f64 GPU',
            'longdouble_explicit_mantissa_bits':np.finfo(np.longdouble).nmant,
            'longdouble_epsilon':str(np.finfo(np.longdouble).eps),
            'conversion':'all parameters/codes/targets start as exact stored f64; longdouble casts exact; Decimal.from_float exact',
            'control_order':[{'seed_index':c['seed_index'],'kind':c['kind']} for c in controls],
            'longdouble_objective_strings':[str(x) for x in costs],
            'longdouble_delta':str(delta),'longdouble_original_band':str(band),
            'longdouble_pass':long_pass,'decimal_required':decimal_required,
            'decimal_trigger':'ordinary-direct objective guard fails OR longdouble delta lies within0.1band of threshold',
            'ordinary_direct_guard':bool(direct_guard)}
        if decimal_required:
            dc=[self.decimal_objective(z,target) for z in codes]
            with localcontext() as ctx:
                ctx.prec=PRECISION
                dmin=min(dc[:-1]);dband=Decimal(1024)*Decimal.from_float(EPS64)*max(dmin,Decimal.from_float(1e-300))
                ddelta=dc[-1]-dmin
                # Compare differences, so a common absolute objective offset cancels.
                discrepancies=[abs(Decimal(str(costs[i]-costs[0]))-(dc[i]-dc[0])) for i in range(len(costs))]
                delta_agreement=max(discrepancies)<=dband*Decimal('.01')
                result.update(decimal_digits=PRECISION,decimal_objective_strings=[str(x) for x in dc],
                    decimal_delta=str(ddelta),decimal_original_band=str(dband),decimal_pass=bool(ddelta<=dband),
                    longdouble_decimal_delta_max_discrepancy=str(max(discrepancies)),
                    longdouble_decimal_delta_agreement=bool(delta_agreement))
        result['passed']=bool(long_pass and (not decimal_required or
            (result['decimal_pass'] and result['longdouble_decimal_delta_agreement'])))
        result['wall_seconds']=time.perf_counter()-begin
        return result
