"""Prospective supervised derivative weighting; not a pointwise PDE residual.

No optimizer, parent model, lambda selection, or execution release is supplied here.
Inputs are dimensionless values and gradients d(u/U)/d(x/D). Frozen training
norms/weights must be reused by a separately reviewed future launch wrapper.
"""
import math

SCHEMA = 'ns-obstacle-supervised-divergence-component-v1'


def validate_weight(weight):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('explicit finite nonnegative divergence weight required')
    return float(weight)


def block_objective(values, gradients, targets, target_gradients, weights,
                    value_norms, gradient_norms, mode_weights, divergence_weight, xp):
    """Additive block contributions, with global training-only denominators.

    Gradient layout: (point, velocity component, spatial derivative, mode).
    The trace error matches the actual FE teacher, which need not be solenoidal.
    Absolute predicted/teacher trace norms are diagnostics, never substitutes for
    the supervised error. All returned entries sum over blocks without averaging.
    """
    dv = values - targets
    dg = gradients - target_gradients
    ev = xp.einsum('n,ncr,ncr->r', weights, dv, dv) / value_norms
    eg = xp.einsum('n,ncdr,ncdr->r', weights, dg, dg) / gradient_norms
    predicted_trace = gradients[:, 0, 0, :] + gradients[:, 1, 1, :]
    teacher_trace = target_gradients[:, 0, 0, :] + target_gradients[:, 1, 1, :]
    trace_error = predicted_trace - teacher_trace
    error_squared = xp.einsum('n,nr,nr->r', weights, trace_error, trace_error)
    predicted_squared = xp.einsum('n,nr,nr->r', weights, predicted_trace, predicted_trace)
    teacher_squared = xp.einsum('n,nr,nr->r', weights, teacher_trace, teacher_trace)
    relative_trace_error = error_squared / gradient_norms
    base = xp.sum(mode_weights * (ev + eg))
    divergence = xp.sum(mode_weights * relative_trace_error)
    combined = base + divergence_weight * divergence
    return combined, {
        'base_objective': base,
        'supervised_divergence_objective_unweighted': divergence,
        'supervised_divergence_objective_weighted': divergence_weight * divergence,
        'combined_objective': combined,
        'value_relative_squared_by_mode': ev,
        'gradient_relative_squared_by_mode': eg,
        'divergence_error_relative_squared_by_mode': relative_trace_error,
        'divergence_error_norm_squared_by_mode': error_squared,
        'predicted_divergence_norm_squared_by_mode': predicted_squared,
        'teacher_divergence_norm_squared_by_mode': teacher_squared,
    }
