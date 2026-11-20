"""Tiny CPU audit using the pinned authors' Decoder and NonlinearSolver directly."""
import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np
import torch
from baselines.crom_scalar2d import ScalarBurgers2D, structured_nodes


def native(name, expected):
    path = Path(__file__).resolve().parent/'results/b6native/output/sources/crom_online/online'/f'{name}.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('b8_native_'+name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return getattr(module, name)


def dense_rhs(field, viscosity, intervals):
    c = field[1:-1, 1:-1]; xm, xp = field[:-2, 1:-1], field[2:, 1:-1]
    ym, yp = field[1:-1, :-2], field[1:-1, 2:]
    dx = torch.where(c > 0, c-xm, xp-c)*intervals
    dy = torch.where(c > 0, c-ym, yp-c)*intervals
    return viscosity*intervals**2*(xm+xp+ym+yp-4*c)-c*(dx+dy)


def main():
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    assert not torch.cuda.is_initialized(), 'This fixture must remain CPU-only'
    Decoder = native('Decoder', '5cbc11c2ea15a206a3de5efc676e78b2f6267979e91b5728133c4435b9a7c385')
    Solver = native('NonlinearSolver', '48a9f6f385ecb7fd0d2ff66ade315e8b7d10196d2bf898a0889448dc498cffc1')
    class Field(torch.nn.Module):
        def __init__(self, modes):
            super().__init__(); self.modes = modes
        def forward(self, inp):
            z, xy = inp[:, :self.modes], inp[:, self.modes:]
            x, y = xy[:, 0], xy[:, 1]
            phi = torch.stack([torch.sin(torch.pi*x)*torch.sin(torch.pi*y),
                               torch.sin(2*torch.pi*x)*torch.sin(torch.pi*y)], 1)[:, :self.modes]
            return torch.sum(z*phi, dim=1, keepdim=True)
    network = Field(2); decoder = Decoder(network, None, None)
    z = torch.tensor([.7, .1]).reshape(1, 1, 2)
    def initial(points):
        return network(torch.cat([z.reshape(1, 2).expand(len(points), -1), points], 1))
    fixed = torch.tensor([[.25, .375], [.5, .5], [.75, .625]], dtype=torch.float64)
    continuous_errors = []
    for intervals in [32, 64, 128]:
        nodes = (fixed*intervals).to(torch.int64)
        problem = ScalarBurgers2D(decoder, 2, initial, nodes, intervals=intervals)
        rhs = problem.getResidualSample(z)[:, 0]
        x, y = fixed[:, 0], fixed[:, 1]; pi = torch.pi
        u = .7*torch.sin(pi*x)*torch.sin(pi*y)+.1*torch.sin(2*pi*x)*torch.sin(pi*y)
        ux = .7*pi*torch.cos(pi*x)*torch.sin(pi*y)+.2*pi*torch.cos(2*pi*x)*torch.sin(pi*y)
        uy = .7*pi*torch.sin(pi*x)*torch.cos(pi*y)+.1*pi*torch.sin(2*pi*x)*torch.cos(pi*y)
        lap = -2*pi*pi*.7*torch.sin(pi*x)*torch.sin(pi*y)-5*pi*pi*.1*torch.sin(2*pi*x)*torch.sin(pi*y)
        continuum = .005*lap-u*(ux+uy)
        continuous_errors.append(float(torch.linalg.vector_norm(rhs-continuum)))
    assert continuous_errors[2] < .3*continuous_errors[0] and continuous_errors[2] < continuous_errors[1]
    intervals = 16; nodes = structured_nodes(intervals-1, intervals)
    problem = ScalarBurgers2D(decoder, 2, initial, nodes, intervals=intervals)
    axis = torch.linspace(0, 1, intervals+1)
    xy = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    dense_error = 0.
    for code in [z, -z]:
        dense = problem.field(code, xy).reshape(intervals+1, intervals+1)
        expected = dense_rhs(dense, .005, intervals).reshape(-1, 1)
        actual = problem.getResidualSample(code)
        dense_error = max(dense_error, float(torch.max(torch.abs(actual-expected))))
    assert dense_error < 1e-13
    boundary_points = torch.tensor([[0., .5], [1., .5], [.5, 0.], [.5, 1.], [-.01, .5], [.5, 1.01]])
    assert torch.count_nonzero(problem.field(z, boundary_points)) == 0
    assert torch.count_nonzero(problem.code_jacobian(z, boundary_points)) == 0
    direction = torch.tensor([.2, -.3]).reshape_as(z); eps = 1e-6
    jac = problem.getJacobianSample(z)
    field_fd = (problem.field(z+eps*direction, problem.points)-problem.field(z-eps*direction, problem.points))/(2*eps)
    jac_error = float(torch.max(torch.abs(field_fd[:, 0]-jac@direction.flatten())))
    rhs_fd = (problem.getResidualSample(z+eps*direction)-problem.getResidualSample(z-eps*direction))/(2*eps)
    rhs_jac_error = float(torch.max(torch.abs(rhs_fd[:, 0]-problem.rhs_code_jacobian(z)@direction.flatten())))
    assert jac_error < 1e-8 and rhs_jac_error < 1e-7
    # Native GN exactly projects this linear decoder, then repeat multiple PDE
    # steps against an independent dense-grid predictor/least-squares recurrence.
    single = Decoder(Field(1), None, None)
    code = torch.tensor([.5]).reshape(1, 1, 1)
    scalar = ScalarBurgers2D(single, 1, lambda p: .5*torch.sin(torch.pi*p[:, :1])*torch.sin(torch.pi*p[:, 1:]), nodes, intervals=intervals)
    solver = Solver(scalar, single, 1e-15, 1e-15)
    phi = torch.sin(torch.pi*scalar.points[:, 0])*torch.sin(torch.pi*scalar.points[:, 1])
    phi_full = (torch.sin(torch.pi*xy[:, 0])*torch.sin(torch.pi*xy[:, 1])).reshape(intervals+1, intervals+1)
    phi_full[[0, -1], :] = 0.; phi_full[:, [0, -1]] = 0.
    expected_code = .5; dt = .01; code_errors = []; diagnostics = []
    for iteration in range(3):
        previous_code = code.clone()
        reference_target = expected_code*phi+dt*dense_rhs(expected_code*phi_full, .005, intervals).reshape(-1)
        expected_code = float(torch.dot(phi, reference_target)/torch.dot(phi, phi))
        target = scalar.explicit_target(code, dt)[:, 0]
        assert torch.max(torch.abs(target-reference_target)) < 1e-13
        code, diag = scalar.advance(code, dt, solver, max_iters=50)
        diagnostics.append(diag); code_errors.append(abs(float(code)-expected_code))
        assert torch.max(torch.abs(scalar.q_sample.reshape(-1)-scalar.field(code, scalar.points)[:, 0])) < 1e-14
        if iteration == 1:
            stale_target = .5*phi+dt*scalar.getResidualSample(previous_code)[:, 0]
            assert torch.linalg.vector_norm(target-stale_target) > 1e-5
    assert max(code_errors) < 1e-12
    fine = ScalarBurgers2D(single, 1, lambda p: torch.zeros((len(p), 1)), structured_nodes(32), intervals=16384)
    assert fine.h == 1/16384 and len(fine.points) == 32**2
    assert len(fine.stencil_points) <= 5*32**2
    print(json.dumps(dict(status='passed', native_decoder_used=True, native_gauss_newton_used=True,
                          dense_rhs_max_error=dense_error, continuum_rhs_errors=continuous_errors,
                          code_jacobian_fd_max_error=jac_error, rhs_code_jacobian_fd_max_error=rhs_jac_error,
                          three_step_code_errors=code_errors, final_projection_diagnostics=diagnostics[-1],
                          stale_sample_regression_passed=True, physical_intervals=fine.intervals,
                          sample_points=len(fine.points), stencil_union_points=len(fine.stencil_points),
                          pde_comparator_admitted=False)))


if __name__ == '__main__':
    main()
