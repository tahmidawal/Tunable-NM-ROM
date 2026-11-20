"""Scalar 2D Burgers problem adapter for CROM's explicit predictor/projector.

The supplied decoder and nonlinear solver are the pinned authors' components.
This new problem adapter is not the authors' original forced 1D Burgers example.
Spatial sample density never changes the fine physical difference spacing.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SamplePoints:
    indices: torch.Tensor


def structured_nodes(q, intervals=16384, *, device='cpu', include_boundary=False):
    """Q×Q fixed sample centers lying on the fine grid (not a coarse PDE grid)."""
    assert isinstance(q, int) and isinstance(intervals, int)
    assert 2 <= q <= intervals-1 and intervals >= 4
    lo, hi = (0, intervals) if include_boundary else (1, intervals-1)
    axis = torch.linspace(lo, hi, q, dtype=torch.float64, device=device).round().to(torch.int64)
    assert len(torch.unique(axis)) == q
    return torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)


class ScalarBurgers2D:
    """Only initial callable, viscosity, geometry and decoder enter this adapter.

    ``decoder`` implements the native ``forward`` and ``jacobianPart`` APIs,
    returning physical field values and physical code derivatives (including
    the training-only normalization inverse when the trained model needs it).
    All methods expose the sample/update API used by native NonlinearSolver.
    Decoder weights are fixed; only the independent latent code evolves.
    """

    def __init__(self, decoder, latent, initial_callable, sample_nodes, *,
                 viscosity=.005, intervals=16384, initial_points=None):
        assert isinstance(intervals, int) and intervals >= 4
        assert isinstance(latent, int) and latent > 0 and viscosity > 0
        nodes = torch.as_tensor(sample_nodes)
        assert nodes.dtype == torch.int64 and nodes.ndim == 2 and nodes.shape[1] == 2
        assert len(nodes) > latent and bool(torch.all((nodes >= 0) & (nodes <= intervals)))
        assert len(torch.unique(nodes, dim=0)) == len(nodes)
        self.decoder = decoder; self.latent = latent; self.initial_callable = initial_callable
        self.nu = float(viscosity); self.intervals = intervals; self.h = 1./intervals
        self.nodes = nodes; self.points = nodes.to(torch.float64)/intervals
        self.x = self.points[None]
        self.sample_point = SamplePoints(torch.arange(len(nodes), device=nodes.device))
        self.initial_points = self.points if initial_points is None else torch.as_tensor(
            initial_points, dtype=torch.float64, device=nodes.device)
        assert self.initial_points.ndim == 2 and self.initial_points.shape[1] == 2
        offsets = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
                               dtype=torch.int64, device=nodes.device)
        stencil = nodes[:, None, :]+offsets[None]
        unique, inverse = torch.unique(stencil.reshape(-1, 2), dim=0, return_inverse=True)
        self.stencil_points = unique.to(torch.float64)/intervals
        self.stencil_inverse = inverse.reshape(len(nodes), 5)
        self.q_sample = None
        self.counters = dict(decoder_calls=0, decoded_points=0, jacobian_calls=0,
                             jacobian_points=0, rhs_calls=0, sample_refreshes=0)

    @staticmethod
    def interior(points):
        return torch.all((points > 0.) & (points < 1.), dim=-1)

    def _code(self, z):
        assert z.numel() == self.latent and z.dtype == torch.float64
        assert z.device == self.points.device and bool(torch.isfinite(z).all())
        return z.reshape(1, self.latent)

    def _indices(self, sample_point):
        indices = self.sample_point.indices if sample_point is None else sample_point.indices
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.points.device)
        assert indices.ndim == 1 and bool(torch.all((indices >= 0) & (indices < len(self.points))))
        return indices

    def field(self, z, points, decoder=None):
        """Return exact zero at/on/outside the Dirichlet boundary."""
        decoder = self.decoder if decoder is None else decoder
        points = torch.as_tensor(points, dtype=torch.float64, device=self.points.device)
        code = self._code(z).expand(len(points), -1)
        values = decoder.forward(torch.cat([code, points], 1)).reshape(-1, 1)
        assert values.shape == (len(points), 1) and values.dtype == torch.float64
        assert bool(torch.isfinite(values).all())
        self.counters['decoder_calls'] += 1; self.counters['decoded_points'] += len(points)
        return torch.where(self.interior(points)[:, None], values, torch.zeros_like(values)).detach()

    def code_jacobian(self, z, points, decoder=None):
        decoder = self.decoder if decoder is None else decoder
        points = torch.as_tensor(points, dtype=torch.float64, device=self.points.device)
        code = self._code(z).expand(len(points), -1)
        # The native autodiff fallback needs grad enabled even when a caller uses no_grad.
        with torch.enable_grad():
            jac = decoder.jacobianPart(torch.cat([code, points], 1), self.latent, 'fir')
        jac = jac.reshape(len(points), self.latent)
        assert jac.dtype == torch.float64 and bool(torch.isfinite(jac).all())
        self.counters['jacobian_calls'] += 1; self.counters['jacobian_points'] += len(points)
        return torch.where(self.interior(points)[:, None], jac, torch.zeros_like(jac)).detach()

    def getPhysicalInitialCondition(self):
        # This callable is the only initial-field input. There is no truth/parameter lookup.
        values = torch.as_tensor(self.initial_callable(self.initial_points), dtype=torch.float64,
                                 device=self.points.device).reshape(-1, 1)
        assert values.shape == (len(self.initial_points), 1) and bool(torch.isfinite(values).all())
        values = torch.where(self.interior(self.initial_points)[:, None], values, torch.zeros_like(values))
        return values.detach()[None]

    def updateState(self, z, decoder=None, index=None, vhat=None, xhat_prev=None, save_q=False):
        points = self.points if index is None else self.points[index]
        values = self.field(z, points, decoder)[None]
        if save_q:
            self.q = values
        return values

    def updateStateSample(self, z, decoder=None, sample_point=None, vhat=None):
        self.q_sample = self.updateState(z, decoder, self._indices(sample_point))
        self.counters['sample_refreshes'] += 1
        return self.q_sample

    def updateStateSampleAll(self, z, decoder=None, sample_point=None, v_hat=None):
        # Explicit refresh avoids dependence on the native GN solver's final-loss
        # side effect. The original Burger.updateStateSampleAll only updates q.
        # Full requested output is obtained separately via field().
        return self.updateStateSample(z, decoder, sample_point)

    def getJacobianSample(self, z, decoder=None, sample_point=None):
        return self.code_jacobian(z, self.points[self._indices(sample_point)], decoder)

    getJacobianSampleProj = getJacobianSample

    def getResidualSample(self, z, decoder=None, sample_point=None, step=None):
        indices = self._indices(sample_point)
        # Decode the deduplicated union once. Subsets may reuse this union; counters
        # report actual evaluations, never an idealized reduced operation count.
        all_values = self.field(z, self.stencil_points, decoder)[:, 0]
        values = all_values[self.stencil_inverse[indices]]
        center, xm, xp, ym, yp = values.unbind(1)
        dx = torch.where(center > 0., center-xm, xp-center)/self.h
        dy = torch.where(center > 0., center-ym, yp-center)/self.h
        lap = (xm+xp+ym+yp-4*center)/(self.h*self.h)
        rhs = self.nu*lap-center*(dx+dy)
        rhs = torch.where(self.interior(self.points[indices]), rhs, torch.zeros_like(rhs))
        self.q_sample = center.reshape(1, -1, 1).detach()
        self.counters['sample_refreshes'] += 1; self.counters['rhs_calls'] += 1
        return rhs[:, None].detach()

    def rhs_code_jacobian(self, z, decoder=None, sample_point=None):
        """Piecewise code derivative for projected-stability diagnostics, away from sign switches."""
        indices = self._indices(sample_point); mapping = self.stencil_inverse[indices]
        values = self.field(z, self.stencil_points, decoder)[:, 0][mapping]
        jac = self.code_jacobian(z, self.stencil_points, decoder)[mapping]
        c, xm, xp, ym, yp = values.unbind(1)
        jc, jxm, jxp, jym, jyp = jac.unbind(1)
        dx = torch.where(c > 0., c-xm, xp-c)/self.h
        dy = torch.where(c > 0., c-ym, yp-c)/self.h
        jdx = torch.where((c > 0.)[:, None], jc-jxm, jxp-jc)/self.h
        jdy = torch.where((c > 0.)[:, None], jc-jym, jyp-jc)/self.h
        jlap = (jxm+jxp+jym+jyp-4*jc)/(self.h*self.h)
        answer = self.nu*jlap-jc*(dx+dy)[:, None]-c[:, None]*(jdx+jdy)
        return torch.where(self.interior(self.points[indices])[:, None], answer, torch.zeros_like(answer))

    def explicit_target(self, z, dt, sample_point=None):
        assert dt > 0
        rhs = self.getResidualSample(z, self.decoder, sample_point)
        # getResidualSample refreshed q_sample from this exact code, never a previous step.
        return self.q_sample.reshape_as(rhs)+float(dt)*rhs

    def projection_diagnostics(self, z, target, previous=None, sample_point=None):
        points = self.points[self._indices(sample_point)]
        q = self.field(z, points); jac = self.code_jacobian(z, points)
        target = target.reshape_as(q); residual = q-target
        norm = torch.linalg.vector_norm(target).clamp(min=torch.finfo(torch.float64).tiny)
        scale = torch.linalg.vector_norm(jac, dim=0).clamp(min=1e-14)
        record = dict(relative_projection_residual=float(torch.linalg.vector_norm(residual)/norm),
                      scaled_stationarity=float(torch.max(torch.abs((jac.T@residual)[:, 0]/scale))/norm),
                      convention='Unweighted native sample objective; not independent physical accuracy')
        if previous is not None:
            old = self.field(previous, points)
            record['relative_pde_field_update'] = float(torch.linalg.vector_norm(q-old)/torch.linalg.vector_norm(old).clamp(min=1e-30))
            record['update_scope'] = 'Whole PDE step, not the native solver final-iteration stopping increment'
        return record

    def advance(self, z, dt, native_solver, *, max_iters=50, sample_point=None):
        """Native physical Euler target + supplied author GN; no weak-step replacement."""
        sample_point = self.sample_point if sample_point is None else sample_point
        assert native_solver.problem is self and max_iters > 0
        target = self.explicit_target(z, dt, sample_point)
        next_code = native_solver.solve(z, target, sample_point, step_size=1., max_iters=max_iters, print_info=False)
        diagnostics = self.projection_diagnostics(next_code, target, previous=z, sample_point=sample_point)
        self.updateStateSampleAll(next_code, self.decoder, sample_point)
        return next_code, diagnostics

    def callBack(self, step, z, decoder=None, vhat=None, xhat_prev=None):
        return False

    def updateSampleIndicesWithProblemSpeficifInfo(self, sample_point):
        return None
