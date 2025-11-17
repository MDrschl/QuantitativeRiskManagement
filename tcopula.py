# tcopula.py
# =============================================================
# t-Copula Parameter Estimation using Maximum Likelihood
# (see Patton's Copula toolbox for Matlab (2008))
# =============================================================

import numpy as np
from scipy.stats import t as t_dist, norm
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import kendalltau


def t_copula_pdf(u1, u2, rho, nu):
    """
    Compute the bivariate t-copula density c(u1, u2; rho, nu).

    Parameters
    ----------
    u1, u2 : array-like
        Uniform marginals in (0,1)
    rho : float
        Correlation parameter in (-1, 1)
    nu : float
        Degrees of freedom parameter (> 2)

    Returns
    -------
    pdf : array-like
        t-copula density values
    """
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)

    # Avoid 0 and 1 to keep finite quantiles
    u1 = np.clip(u1, 1e-10, 1 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1 - 1e-10)

    # Transform uniforms to t_nu variables
    x1 = t_dist.ppf(u1, df=nu)
    x2 = t_dist.ppf(u2, df=nu)

    # Bivariate t density with df = nu and correlation rho
    detR = 1.0 - rho**2
    q = (x1**2 - 2 * rho * x1 * x2 + x2**2) / detR

    const = gamma((nu + 2) / 2.0) / (
        gamma(nu / 2.0) * (nu * np.pi) * np.sqrt(detR)
    )
    joint_t = const * (1.0 + q / nu) ** (-(nu + 2) / 2.0)

    # Marginal t densities with df = nu
    marginal1 = t_dist.pdf(x1, df=nu)
    marginal2 = t_dist.pdf(x2, df=nu)

    # Copula density = joint t density / (product of marginals)
    copula_density = joint_t / (marginal1 * marginal2)

    return copula_density


def t_copula_neg_loglik(params, u1, u2):
    """
    Negative log-likelihood for t-copula.

    Parameters
    ----------
    params : array-like
        [rho, nu] where rho in (-1,1) and nu > 2
    u1, u2 : array-like
        Uniform marginals

    Returns
    -------
    neg_ll : float
        Negative log-likelihood
    """
    rho, nu = params

    # Parameter constraints
    if rho <= -0.99 or rho >= 0.99:
        return 1e10
    if nu <= 2.01 or nu > 100:  # cap nu for numerical stability
        return 1e10

    u1 = np.asarray(u1)
    u2 = np.asarray(u2)

    # Handle edge cases
    u1 = np.clip(u1, 1e-10, 1 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1 - 1e-10)

    try:
        pdf_vals = t_copula_pdf(u1, u2, rho, nu)

        # Avoid log of zero or negative
        pdf_vals = np.clip(pdf_vals, 1e-12, None)

        log_likelihood = np.sum(np.log(pdf_vals))

        if not np.isfinite(log_likelihood):
            return 1e10

        return -log_likelihood

    except Exception:
        return 1e10


def fit_t_copula(u1, u2, init_rho=None, init_nu=None):
    """
    Estimate t-copula parameters (rho, nu) using maximum likelihood.

    Parameters
    ----------
    u1, u2 : array-like
        Uniform marginals in (0,1), typically obtained via
        probability integral transform of Gaussian marginals.
    init_rho : float, optional
        Initial guess for correlation. If None, uses Kendall's tau.
    init_nu : float, optional
        Initial guess for degrees of freedom. Default is 10.

    Returns
    -------
    rho : float
        Estimated correlation parameter
    nu : float
        Estimated degrees of freedom
    convergence : bool
        Whether optimization converged successfully
    """
    u1 = np.asarray(u1).flatten()
    u2 = np.asarray(u2).flatten()

    # Remove any NaN or invalid values
    valid = (
        np.isfinite(u1)
        & np.isfinite(u2)
        & (u1 > 0) & (u1 < 1)
        & (u2 > 0) & (u2 < 1)
    )
    u1 = u1[valid]
    u2 = u2[valid]

    if len(u1) < 20:
        print("Not enough data.")
        return np.nan, np.nan, False

    # Initial parameter guesses
    if init_rho is None:
        # Use Kendall's tau as initial estimate for rho (same as Gaussian/t copula)
        tau, _ = kendalltau(u1, u2)
        init_rho = np.sin(tau * np.pi / 2.0)
        init_rho = np.clip(init_rho, -0.9, 0.9)

    if init_nu is None:
        init_nu = 10.0

    # Bounds: rho in (-0.99, 0.99), nu in (2.1, 100)
    bounds = [(-0.99, 0.99), (2.1, 100.0)]

    # Multiple starting points for robustness
    best_result = None
    best_ll = np.inf

    starting_points = [
        [init_rho, init_nu],
        [init_rho * 0.8, 8.0],
        [init_rho * 1.2, 15.0],
        [0.5 * init_rho, 5.0],
    ]

    for start in starting_points:
        # Ensure starting point is within bounds
        start[0] = np.clip(start[0], -0.98, 0.98)
        start[1] = np.clip(start[1], 2.1, 99.0)

        try:
            result = minimize(
                t_copula_neg_loglik,
                x0=start,
                args=(u1, u2),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12},
            )

            if result.fun < best_ll:
                best_ll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None or not best_result.success:
        # Fallback to simple correlation if optimization fails
        print("Fallback")
        return np.corrcoef(u1, u2)[0, 1], 10.0, False

    rho_opt, nu_opt = best_result.x
    return rho_opt, nu_opt, best_result.success
