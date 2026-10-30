"""Exact analytic geometry envelopes. All coordinates and gradients are physical.

These functions do not certify a curved FE geometry approximation or incompressibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Geometry:
    kind: str
    inlet: float
    outlet: float
    center: tuple[float, ...]
    obstacle_radius: float
    height: float | None = None
    pipe_radius: float | None = None

    def __post_init__(self):
        if self.kind not in ("channel2d", "pipe3d"):
            raise ValueError("unknown geometry")
        if len(self.center) != self.dimension or self.outlet <= self.inlet:
            raise ValueError("invalid geometry dimension or axial interval")
        if not self.obstacle_radius > 0:
            raise ValueError("positive obstacle radius required")
        if not (self.inlet < self.center[0] - self.obstacle_radius and
                self.center[0] + self.obstacle_radius < self.outlet):
            raise ValueError("obstacle must be strictly inside axial interval")
        if self.kind == "channel2d":
            if self.height is None or not (self.obstacle_radius < self.center[1] <
                                          self.height - self.obstacle_radius):
                raise ValueError("obstacle must lie inside channel")
        elif self.pipe_radius is None or not (np.linalg.norm(self.center[1:]) +
                                              self.obstacle_radius < self.pipe_radius):
            raise ValueError("sphere must lie inside pipe")

    @property
    def dimension(self):
        return 2 if self.kind == "channel2d" else 3

    def as_dict(self):
        return asdict(self)

    def coordinate_scaling(self):
        length = self.outlet - self.inlet
        if self.dimension == 2:
            return np.array([(self.inlet + self.outlet) / 2, self.height / 2]), np.array([length / 2, self.height / 2])
        return np.array([(self.inlet + self.outlet) / 2, 0., 0.]), np.array([length / 2, self.pipe_radius, self.pipe_radius])

    def envelope(self, points):
        """Return b[n], db[n,coordinate]; no division by b at its zeros."""
        x = np.asarray(points, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError("points must be finite (n,dimension)")
        length = self.outlet - self.inlet
        axial = (x[:, 0] - self.inlet) / length
        da = np.zeros_like(x)
        da[:, 0] = 1 / length
        if self.dimension == 2:
            wall = x[:, 1] * (self.height - x[:, 1]) / self.height**2
            dw = np.zeros_like(x)
            dw[:, 1] = (self.height - 2*x[:, 1]) / self.height**2
        else:
            wall = 1 - np.sum(x[:, 1:]**2, axis=1) / self.pipe_radius**2
            dw = np.zeros_like(x)
            dw[:, 1:] = -2*x[:, 1:] / self.pipe_radius**2
        offset = x - np.asarray(self.center)
        obstacle = (np.sum(offset**2, axis=1) - self.obstacle_radius**2) / length**2
        do = 2*offset / length**2
        value = axial * wall * obstacle
        gradient = (da * (wall*obstacle)[:, None] +
                    dw * (axial*obstacle)[:, None] +
                    do * (axial*wall)[:, None])
        return value, gradient

    def fluid_mask(self, points, tolerance=0.):
        x = np.asarray(points, dtype=np.float64)
        axial = (x[:, 0] >= self.inlet-tolerance) & (x[:, 0] <= self.outlet+tolerance)
        if self.dimension == 2:
            wall = (x[:, 1] >= -tolerance) & (x[:, 1] <= self.height+tolerance)
        else:
            wall = np.linalg.norm(x[:, 1:], axis=1) <= self.pipe_radius+tolerance
        obstacle = np.linalg.norm(x-np.asarray(self.center), axis=1) >= self.obstacle_radius-tolerance
        return axial & wall & obstacle


CHANNEL = Geometry("channel2d", 0., 2.2, (0.2, 0.2), 0.05, height=0.41)
PIPE = Geometry("pipe3d", -5., 15., (0., 0., 0.), 0.5, pipe_radius=2.5)
