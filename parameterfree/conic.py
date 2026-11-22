# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import math

import torch
from .base import BatchedAlgorithm, BatchedOptimizer


def _solve_scale(theta_norm, m, a, b, epsilon, gamma):
    """Solve (52) via its dimensionless cubic and an exact constrained root."""
    if theta_norm == 0:
        return 0.0, epsilon / b
    cap_p = b / (0.5 + 0.5 * math.hypot(
        1, (math.sqrt(gamma) / math.sqrt(2)) * math.sqrt(b) / m))
    if theta_norm / (a + gamma * cap_p) >= 0.5 / m:
        return (cap_p / (2 * m)) / theta_norm, epsilon / cap_p

    A = a / gamma
    c = ((theta_norm / math.sqrt(a)) / math.sqrt(2)) ** 2
    beta = b / A
    d = 2 + c - beta
    if beta <= 1 + c / 4:
        upper = beta / (0.5 + 0.5 * math.hypot(1, math.sqrt(c) * math.sqrt(beta)))
        lower = upper / 2
    else:
        disc = math.hypot(d, (2 * math.sqrt(2)) * math.sqrt(beta))
        r = 4 * beta / (disc + d) if d >= 0 else (disc - d) / 2
        lower, upper = max(1, r), min(beta, r + 0.5)
    padding = 8 * math.ulp(1.0) if hasattr(math, 'ulp') else 8 * 2.220446049250313e-16
    lower = max(lower * (1 - padding), beta - c)
    upper = min(beta, upper * (1 + padding))
    z = lower + (upper - lower) / 2
    e = 1 - 2 * beta
    for _ in range(3):
        f = ((z + d) * z + e) * z - beta
        derivative = (3 * z + 2 * d) * z + e
        second = 6 * z + 2 * d
        if f < 0:
            lower = z
        elif f > 0:
            upper = z
        denominator = derivative * derivative - 0.5 * f * second
        proposal = z - f * derivative / denominator if denominator != 0 else float('nan')
        next_z = min(max(proposal, lower), upper) if math.isfinite(proposal) else lower + (upper - lower) / 2
        if next_z == z:
            break
        z = next_z
    p = min(A * z, b)
    scale = min(p / (a + gamma * p), (p / (2 * m)) / theta_norm)
    return scale, epsilon / p


def _solve_scale_tensor(theta_norm, m, a, b, epsilon, gamma):
    """Solve the same stationarity equation with three bracketed Halley steps.

    With A=a/gamma, z=p/A, c=theta_norm**2/(2*gamma*A), and beta=b/A,
    the unconstrained equation is z + c*(z/(1+z))**2 = beta. Multiplying
    by (1+z)**2 yields z**3 + (2+c-beta)*z**2 + (1-2*beta)*z - beta = 0.
    Horner evaluation needs one division per refinement, rather than repeatedly
    evaluating a rational function. Existing quadratic bounds supply a tight
    bracket. Finite proposals are clipped to handle rounded endpoint roots.
    The constrained branch still uses its exact quadratic solution.
    """
    safe_m = torch.where(m > 0, m, torch.ones_like(m))
    safe_norm = torch.where(theta_norm > 0, theta_norm, torch.ones_like(theta_norm))
    A = a / gamma
    inverse_sqrt_a = a.rsqrt()
    sqrt_c = (theta_norm * inverse_sqrt_a) / math.sqrt(2)
    c = sqrt_c.square()
    beta = b / A
    ones = torch.ones_like(b)
    sqrt_beta = beta.sqrt()
    small_upper = beta / (0.5 + 0.5 * torch.hypot(ones, sqrt_c * sqrt_beta))
    d = 2 + c - beta
    disc = torch.hypot(d, (2 * math.sqrt(2)) * sqrt_beta)
    safe_sum = torch.where(d >= 0, disc + d, ones)
    r = torch.where(d >= 0, 4 * beta / safe_sum, (disc - d) / 2)
    small = beta <= 1 + c / 4
    lower = torch.where(small, small_upper / 2, torch.maximum(ones, r))
    upper = torch.where(small, small_upper, torch.minimum(beta, r + 0.5))
    padding = 8 * torch.finfo(b.dtype).eps
    # The nonlinear term is <= c, so z >= beta-c. This also makes the
    # zero-direction root exactly beta and tightens nearly linear cases.
    lower = torch.maximum(lower * (1 - padding), beta - c)
    upper = torch.minimum(beta, upper * (1 + padding))
    z = lower + (upper - lower) / 2
    e = 1 - 2 * beta
    for _ in range(3):
        f = ((z + d) * z + e) * z - beta
        derivative = (3 * z + 2 * d) * z + e
        second = 6 * z + 2 * d
        lower = torch.where(f < 0, z, lower)
        upper = torch.where(f > 0, z, upper)
        proposal = z - f * derivative / (derivative.square() - 0.5 * f * second)
        clipped = torch.minimum(torch.maximum(proposal, lower), upper)
        z = torch.where(torch.isfinite(proposal), clipped, lower + (upper - lower) / 2)
    p = torch.minimum(A * z, b)
    # gamma*b = a*beta: reuse rsqrt(a) and sqrt(beta) instead of another sqrt.
    cap_p = b / (0.5 + 0.5 * torch.hypot(
        ones, sqrt_beta / ((math.sqrt(2) * safe_m) * inverse_sqrt_a)))
    capped = theta_norm / (a + gamma * cap_p) >= 0.5 / safe_m
    p = torch.where(capped, cap_p, p)
    scale = torch.minimum(p / (a + gamma * p), (p / (2 * safe_m)) / safe_norm)
    scale = torch.where(theta_norm > 0, scale, torch.zeros_like(scale))
    return scale, epsilon / p


class ConicAlgorithm(BatchedAlgorithm):
    """Algorithm 4, with independent lifted state per leading batch entry."""

    skip_zero_gradients = True

    def __init__(self, epsilon=1.0, gamma=1.0):
        for name, value in (('epsilon', epsilon), ('gamma', gamma)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid {} value: {}'.format(name, value))
        self.config = dict(epsilon=float(epsilon), gamma=float(gamma))

    def init_state(self, initial_point):
        zeros = torch.zeros(initial_point.shape[0], dtype=initial_point.dtype, device=initial_point.device)
        return dict(anchor=initial_point.clone(), theta=torch.zeros_like(initial_point),
                    m=zeros.clone(), v=zeros.clone(), delta=zeros.clone(),
                    lifted_loss=zeros.clone(), height=torch.ones_like(zeros))

    def update(self, state, point, gradient):
        epsilon, gamma = self.config['epsilon'], self.config['gamma']
        theta = state['theta'] + gradient
        displacement = point - state['anchor']
        if point.shape[-1] == 1:
            # Independent scalar learners need no reductions or sqrt(theta**2).
            grad_sq = gradient.squeeze(-1).square()
            gain = gradient.squeeze(-1) * displacement.squeeze(-1)
            theta_norm = theta.squeeze(-1).abs()
        else:
            grad_sq = torch.linalg.vector_norm(gradient, dim=-1).square()
            gain = (gradient * displacement).sum(dim=-1)
            theta_norm = torch.linalg.vector_norm(theta, dim=-1)
        compiling = hasattr(torch, 'compiler') and torch.compiler.is_compiling()
        if point.shape[0] == 1 and not compiling:
            # Keep the vector optimizer's scalar solve cheap: one batched host
            # transfer rather than launching dozens of tiny GPU root kernels.
            values = torch.stack((grad_sq, gain, theta_norm, state['m'], state['v'],
                                  state['delta'], state['lifted_loss'])).reshape(-1).tolist()
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError('ConicAlgorithm received non-finite gradients or state')
            g2, gain_value, norm, old_m, old_v, old_delta, old_loss = values
            if g2 == 0:
                return state, point
            m = max(old_m, math.sqrt(g2))
            v = old_v + g2
            delta = old_delta + (m - old_m) ** 2
            lifted_loss = old_loss - gain_value
            a = gamma * epsilon + 4 * (v + m * m)
            b = epsilon + delta / (2 * gamma) + lifted_loss
            if not math.isfinite(a) or not math.isfinite(b) or b <= 0:
                raise RuntimeError('ConicAlgorithm scalar state lost numerical precision')
            scale, height = _solve_scale(norm, m, a, b, epsilon, gamma)
            scalars = torch.tensor([m, v, delta, lifted_loss, height],
                                   dtype=point.dtype, device=point.device)
            candidate = dict(anchor=state['anchor'], theta=theta)
            candidate.update({key: scalars[index:index + 1] for index, key in enumerate(
                ('m', 'v', 'delta', 'lifted_loss', 'height'))})
            return candidate, state['anchor'] - theta * scale

        m = torch.maximum(state['m'], grad_sq.sqrt())
        v = state['v'] + grad_sq
        delta = state['delta'] + (m - state['m']).square()
        lifted_loss = state['lifted_loss'] - gain
        a = gamma * epsilon + 4 * (v + m.square())
        b = epsilon + delta / (2 * gamma) + lifted_loss
        scale, height = _solve_scale_tensor(theta_norm, m, a, b, epsilon, gamma)
        candidate = dict(anchor=state['anchor'], theta=theta, m=m, v=v,
                         delta=delta, lifted_loss=lifted_loss, height=height)
        prediction = state['anchor'] - theta * scale.unsqueeze(-1)
        return candidate, prediction

    def is_valid(self, state):
        epsilon, gamma = self.config['epsilon'], self.config['gamma']
        return epsilon + state['delta'] / (2 * gamma) + state['lifted_loss'] > 0


class ConicFTRL(BatchedOptimizer):
    """Euclidean Conic FTRL with unknown gradient bound (Algorithm 4).

    Section 8 of Grand-Clement, Kwon, and Orabona,
    https://arxiv.org/abs/2610.39531. The initial network weights are a fixed
    anchor, and all parameters share one lifted scalar coordinate. epsilon is
    the initial loss budget (not numerical stabilization); gamma is the
    regularization tradeoff. Both default to 1. No clipping is required.

    Zero-gradient rounds leave state and predictions unchanged. Missing
    components are zero feedback in this joint vector learner. Schedulers scale
    gradients before all updates; L2 weight decay never changes p.grad.
    CoordinateWise(ConicFTRL, ...) uses independent scalar copies of this rule.
    """

    algorithm_type = ConicAlgorithm

    def __init__(self, params, epsilon=1.0, gamma=1.0, weight_decay=0.0,
                 error_if_nonfinite=True, compile=False):
        super().__init__(params, ConicAlgorithm(epsilon, gamma), weight_decay=weight_decay,
                         error_if_nonfinite=error_if_nonfinite, compile=compile)
