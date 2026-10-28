"""Independent analytic array checks for the A6 FE adapter. No FE import."""
import numpy as np
from spaces import positive_weights

ANALYTIC_TOLERANCE = 1e-10
OBSTACLE_RADIAL_COSINE_MAX = -.99


def boundary_control(points, weights, normals, labels, center):
    x = np.asarray(points); n = np.asarray(normals); labels = np.asarray(labels)
    w = positive_weights(weights, len(x))
    directions = {'inlet': [-1., 0.], 'outlet': [1., 0.],
                  'wall_bottom': [0., -1.], 'wall_top': [0., 1.]}
    if (x.shape != n.shape or x.shape[1:] != (2,) or not np.isfinite(x).all() or
        not np.isfinite(n).all() or set(labels) != set(directions)|{'obstacle'}):
        raise ValueError('invalid boundary samples or labels')
    unit_error = float(np.max(np.abs(np.linalg.norm(n, axis=1)-1)))
    side_errors = {name: float(np.max(np.linalg.norm(n[labels == name]-direction, axis=1)))
                   for name, direction in directions.items()}
    obstacle = labels == 'obstacle'; radial = x[obstacle]-np.asarray(center)
    lengths = np.linalg.norm(radial, axis=1)
    if np.any(lengths == 0): raise ValueError('undefined radial normal at obstacle center')
    cosines = np.einsum('ij,ij->i', n[obstacle], radial/lengths[:, None])
    closures = {}
    for name, selected in [('obstacle', obstacle), ('outer_rectangle', ~obstacle)]:
        length = float(np.sum(w[selected]))  # Positive full closed-component measure.
        integral = np.sum(w[selected, None]*n[selected], axis=0)
        closures[name] = {'integral_normal_ds': integral.tolist(), 'perimeter': length,
                          'normalized_closure': float(np.linalg.norm(integral)/length)}
    report = {'unit_error': unit_error, 'box_direction_errors': side_errors,
              'obstacle_radial_cosine_range': [float(np.min(cosines)), float(np.max(cosines))],
              'closed_components': closures, 'tolerance': ANALYTIC_TOLERANCE,
              'obstacle_radial_cosine_max': OBSTACLE_RADIAL_COSINE_MAX}
    if (unit_error > ANALYTIC_TOLERANCE or max(side_errors.values()) > ANALYTIC_TOLERANCE or
        np.max(cosines) > OBSTACLE_RADIAL_COSINE_MAX or
        max(c['normalized_closure'] for c in closures.values()) > ANALYTIC_TOLERANCE):
        raise ArithmeticError('analytic boundary normal/closure control failed: '+str(report))
    return report


def gradient_control(points, values, gradients, component):
    x = np.asarray(points); u = np.asarray(values); g = np.asarray(gradients)
    if x.shape[1:] != (2,) or u.shape != x.shape or g.shape != (len(x), 2, 2):
        raise ValueError('gradient control shape mismatch')
    exact = np.zeros_like(u); expected_gradient = np.zeros_like(g)
    if component == 0:
        exact[:, 0] = x[:, 1]; expected_gradient[:, 0, 1] = 1.
    elif component == 1:
        exact[:, 1] = x[:, 0]; expected_gradient[:, 1, 0] = 1.
    else: raise ValueError('unknown analytic control component')
    value_error = float(np.max(np.abs(u-exact))/max(1., float(np.max(np.abs(exact)))))
    gradient_error = float(np.max(np.abs(g-expected_gradient)))
    if not np.isfinite(value_error+gradient_error) or max(value_error, gradient_error) > ANALYTIC_TOLERANCE:
        raise ArithmeticError('off-diagonal derivative convention/control failed')
    return {'field': '(y,0)' if component == 0 else '(0,x)',
            'value_relative_max_error': value_error, 'gradient_absolute_max_error': gradient_error,
            'tolerance': ANALYTIC_TOLERANCE, 'gradient_axes': 'component,derivative_coordinate'}
