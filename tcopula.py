# tcopula.py
# =============================================================
# t-Copula Parameter Estimation using Maximum Likelihood (for reference see )
# =============================================================

import numpy as np
from scipy.stats import t as t_dist, norm
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import kendalltau

def t_copula_pdf(u1, u2, rho, nu):
    """
    Compute the bivariate t-copula density.
    
    Parameters
    ----------
    u1, u2 : array-like
        Uniform marginals in [0,1]
    rho : float
        Correlation parameter in (-1, 1)
    nu : float
        Degrees of freedom parameter (> 2)
    
    Returns
    -------
    pdf : array-like
        t-copula density values
    """
    # Transform uniforms to standard t variables
    x1 = t_dist.ppf(u1, df=nu)
    x2 = t_dist.ppf(u2, df=nu)
    
    # Bivariate t density (standardized)
    numerator = gamma((nu + 2) / 2) * gamma(nu / 2)
    denominator = gamma((nu + 1) / 2)**2 * np.pi * np.sqrt(1 - rho**2)
    
    const = numerator / denominator
    
    inner = 1 + (x1**2 + x2**2 - 2*rho*x1*x2) / (nu * (1 - rho**2))
    bivariate_t = const * inner**(-(nu + 2) / 2)
    
    # Marginal t densities
    marginal1 = t_dist.pdf(x1, df=nu)
    marginal2 = t_dist.pdf(x2, df=nu)
    
    # Copula density = joint density / (marginal1 * marginal2)
    copula_density = bivariate_t / (marginal1 * marginal2)
    
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
    if nu <= 2.01 or nu > 100: # Cap at 100 for numerical stability
        return 1e10
    
    # Handle edge cases
    u1 = np.clip(u1, 1e-10, 1 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1 - 1e-10)
    
    try:
        pdf_vals = t_copula_pdf(u1, u2, rho, nu)
        
        # Avoid log of zero or negative
        pdf_vals = np.clip(pdf_vals, 1e-10, None)
        
        log_likelihood = np.sum(np.log(pdf_vals))
        
        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            return 1e10
        
        return -log_likelihood
    
    except:
        return 1e10


def fit_t_copula(u1, u2, init_rho=None, init_nu=None):
    """
    Estimate t-copula parameters (rho, nu) using maximum likelihood.
    
    Parameters
    ----------
    u1, u2 : array-like
        Uniform marginals in [0,1], typically obtained via 
        probability integral transform of Gaussian marginals
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
    
    Examples
    --------
    >>> # Transform Gaussian returns to uniforms
    >>> mu1, sigma1 = np.mean(returns1), np.std(returns1, ddof=1)
    >>> mu2, sigma2 = np.mean(returns2), np.std(returns2, ddof=1)
    >>> u1 = norm.cdf(returns1, mu1, sigma1)
    >>> u2 = norm.cdf(returns2, mu2, sigma2)
    >>> rho, nu, converged = fit_t_copula(u1, u2)
    """
    u1 = np.asarray(u1).flatten()
    u2 = np.asarray(u2).flatten()
    
    # Remove any NaN or invalid values
    valid = np.isfinite(u1) & np.isfinite(u2) & (u1 > 0) & (u1 < 1) & (u2 > 0) & (u2 < 1)
    u1 = u1[valid]
    u2 = u2[valid]
    
    if len(u1) < 20:
        # Not enough data
        print("Not enough data.")
        return np.nan, np.nan, False
        
    
    # Initial parameter guesses
    if init_rho is None:
        # Use Kendall's tau as initial estimate for rho
        
        tau, _ = kendalltau(u1, u2)
        init_rho = np.sin(tau * np.pi / 2)
        init_rho = np.clip(init_rho, -0.9, 0.9)
    
    if init_nu is None:
        init_nu = 10.0
    
    # Bounds: rho in (-0.99, 0.99), nu in (2.1, 100)
    bounds = [(-0.99, 0.99), (2.1, 100)]
    
    # Multiple starting points for robustness
    best_result = None
    best_ll = np.inf
    
    starting_points = [
        [init_rho, init_nu],
        [init_rho * 0.8, 8.0],
        [init_rho * 1.2, 15.0],
        [0.5 * init_rho, 5.0]
    ]
    
    for start in starting_points:
        # Ensure starting point is within bounds
        start[0] = np.clip(start[0], -0.98, 0.98)
        start[1] = np.clip(start[1], 2.1, 99)
        
        try:
            result = minimize(
                t_copula_neg_loglik,
                x0=start,
                args=(u1, u2),
                method= "L-BFGS-B",
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if result.fun < best_ll:
                best_ll = result.fun
                best_result = result
        except:
            continue

    
    if best_result is None or not best_result.success:
        # Fallback to simple correlation if optimization fails
        print("Fallback")
        return np.corrcoef(u1, u2)[0, 1], 10.0, False
    
    rho_opt = best_result.x[0]
    nu_opt = best_result.x[1]
    
    return rho_opt, nu_opt, best_result.success


def fit_gaussian_marginals_and_t_copula(returns1, returns2):
    """
    Complete estimation pipeline: fit Gaussian marginals and t-copula.
    
    This function combines marginal fitting and copula estimation in one call,
    which is convenient for rolling window applications.
    
    Parameters
    ----------
    returns1, returns2 : array-like
        Raw return series (e.g., log returns)
    
    Returns
    -------
    params : dict
        Dictionary containing:
        - 'mu1', 'mu2': marginal means
        - 'sigma1', 'sigma2': marginal standard deviations
        - 'rho': t-copula correlation
        - 'nu': t-copula degrees of freedom
        - 'converged': whether estimation succeeded
    
    Examples
    --------
    >>> # For rolling window application
    >>> window_returns1 = Theta1[i-window:i]
    >>> window_returns2 = Theta2[i-window:i]
    >>> params = fit_gaussian_marginals_and_t_copula(window_returns1, window_returns2)
    >>> if params['converged']:
    >>>     rho, nu = params['rho'], params['nu']
    """
    returns1 = np.asarray(returns1).flatten()
    returns2 = np.asarray(returns2).flatten()
    
    # Remove NaN values
    valid = np.isfinite(returns1) & np.isfinite(returns2)
    returns1 = returns1[valid]
    returns2 = returns2[valid]
    
    if len(returns1) < 20:
        return {
            'mu1': np.nan, 'mu2': np.nan,
            'sigma1': np.nan, 'sigma2': np.nan,
            'rho': np.nan, 'nu': np.nan,
            'converged': False
        }
    
    # Fit Gaussian marginals
    mu1, sigma1 = np.mean(returns1), np.std(returns1, ddof=1)
    mu2, sigma2 = np.mean(returns2), np.std(returns2, ddof=1)
    
    # Transform to uniform marginals
    u1 = norm.cdf(returns1, mu1, sigma1)
    u2 = norm.cdf(returns2, mu2, sigma2)
    
    # Fit t-copula
    rho, nu, converged = fit_t_copula(u1, u2)
    
    return {
        'mu1': mu1, 'mu2': mu2,
        'sigma1': sigma1, 'sigma2': sigma2,
        'rho': rho, 'nu': nu,
        'converged': converged
    }