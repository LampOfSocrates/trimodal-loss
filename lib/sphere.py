"""Directional-statistics primitives on the unit hypersphere S^{d-1}.

Everything here is coordinate-free (dot products only) — we deliberately avoid
literal spherical coordinates, which are singular/ill-conditioned in high d.
All functions are batched torch ops and differentiable unless noted.
"""

import math

import torch

_EPS = 1e-7


def normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=dim)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Angle between unit vectors, arccos clamped away from +-1 so the
    gradient (which blows up like 1/sqrt(1-c^2)) stays finite."""
    c = (x * y).sum(-1).clamp(-1 + _EPS, 1 - _EPS)
    return torch.acos(c)


def log_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Log map at p: tangent vector at p pointing to q with norm d_g(p, q)."""
    c = (p * q).sum(-1, keepdim=True).clamp(-1 + _EPS, 1 - _EPS)
    theta = torch.acos(c)
    # q's component orthogonal to p, rescaled to length theta
    perp = q - c * p
    # sin(theta) = |perp| up to rounding; guard the tiny-angle case
    return perp * (theta / torch.sin(theta).clamp_min(_EPS))


def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exp map at p for tangent vector v (v assumed orthogonal to p)."""
    theta = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cos(theta) * p + torch.sin(theta) * (v / theta)


def karcher_mean(points: torch.Tensor, iters: int = 10, tol: float = 1e-6,
                 exact: bool = True) -> torch.Tensor:
    """Frechet/Karcher mean of points (..., M, D) on the sphere.

    Init = normalized arithmetic mean (already a 2nd-order-accurate
    approximation for tight clusters); with exact=True refine by Riemannian
    gradient descent (average of log maps, then exp map).
    """
    m = normalize(points.mean(dim=-2))
    if not exact:
        return m
    for _ in range(iters):
        v = log_map(m.unsqueeze(-2), points).mean(dim=-2)
        if v.norm(dim=-1).max() < tol:
            break
        m = normalize(exp_map(m, v))  # renormalize to kill drift
    return m


def spherical_triangle_excess(a: torch.Tensor, b: torch.Tensor,
                              c: torch.Tensor) -> torch.Tensor:
    """Area (spherical excess E, Girard) of the geodesic triangle a,b,c on
    S^{d-1}, via L'Huilier's theorem — numerically stable, uses only the three
    side lengths, exact for any d (three points span a <=3-dim subspace).

    E -> 0 for near-degenerate (collinear-on-a-great-circle) triangles.
    """
    la = geodesic_distance(b, c)
    lb = geodesic_distance(a, c)
    lc = geodesic_distance(a, b)
    s = (la + lb + lc) / 2
    # clamp: rounding can make (s - side) slightly negative for degenerate
    # triangles, and tan explodes at pi/2 (giant triangles; not our regime)
    prod = (torch.tan(s / 2).clamp_min(0)
            * torch.tan(((s - la) / 2).clamp_min(0))
            * torch.tan(((s - lb) / 2).clamp_min(0))
            * torch.tan(((s - lc) / 2).clamp_min(0)))
    return 4 * torch.atan(torch.sqrt(prod.clamp_min(0) + _EPS**2) - _EPS)


# ---------------------------------------------------------------- vMF ------

def log_bessel_iv(nu: float, kappa: torch.Tensor) -> torch.Tensor:
    """log I_nu(kappa) via the uniform asymptotic expansion (Abramowitz &
    Stegun 9.7.7, leading term). Accurate for large order nu — our regime is
    nu = d/2 - 1 (e.g. 255 for d=512). Differentiable in kappa.

    CAVEAT (paper.md A4): for small nu AND small kappa use the series instead.
    Validated against scipy.special.ive in 03_vmf_loss.ipynb.
    """
    z = kappa / nu
    t = torch.sqrt(1 + z ** 2)
    eta = t + torch.log(z.clamp_min(_EPS)) - torch.log(1 + t)
    return nu * eta - 0.5 * math.log(2 * math.pi * nu) - 0.5 * torch.log(t)


def log_vmf_normalizer(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """log C_d(kappa) for the vMF density C_d(k) exp(k mu.x) on S^{d-1}."""
    nu = d / 2 - 1
    return (nu * torch.log(kappa.clamp_min(_EPS))
            - (d / 2) * math.log(2 * math.pi)
            - log_bessel_iv(nu, kappa))


def vmf_nll(x: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor,
            d: int | None = None) -> torch.Tensor:
    """Per-sample vMF negative log-likelihood, -[kappa mu.x + log C_d(kappa)].

    d(NLL)/d(kappa) = A_d(kappa) - mu.x, so kappa converges to the
    concentration whose expected alignment A_d(kappa) matches the observed
    mean alignment — the learned per-modality uncertainty weight.
    """
    d = d or x.shape[-1]
    return -(kappa * (x * mu).sum(-1) + log_vmf_normalizer(d, kappa))


@torch.no_grad()
def kappa_mle(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Banerjee et al. (2005) closed-form kappa estimate from unit vectors
    x (N, D): kappa ~= Rbar (d - Rbar^2) / (1 - Rbar^2). Returns (kappa_hat,
    mean_direction). Diagnostic only (not differentiable path)."""
    m = x.mean(dim=0)
    rbar = m.norm().clamp(1e-6, 1 - 1e-6)
    d = x.shape[-1]
    return rbar * (d - rbar ** 2) / (1 - rbar ** 2), m / m.norm()
