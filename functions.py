# functions.py
# =============================================================
# Utility and Simulation Functions for QRM Assignment
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
from copulae import StudentCopula


# -------------------------------------------------------------
# (1) Data Preprocessing
# -------------------------------------------------------------
def preprocess_indices(indices_df):
    """
    Cleans and merges index data (SPI and SPX) and computes weekly log returns.
    """
    indices_clean = indices_df.copy()
    indices_clean.columns = ['SPI_Date', 'SPI', 'SPX_Date', 'SPX']
    indices_clean = indices_clean.iloc[1:]
    
    indices_clean['SPI_Date'] = pd.to_datetime(indices_clean['SPI_Date'])
    indices_clean['SPX_Date'] = pd.to_datetime(indices_clean['SPX_Date'])
    indices_clean['SPI'] = pd.to_numeric(indices_clean['SPI'], errors='coerce')
    indices_clean['SPX'] = pd.to_numeric(indices_clean['SPX'], errors='coerce')
    
    spi_df = indices_clean[['SPI_Date', 'SPI']].rename(columns={'SPI_Date': 'Date'})
    spx_df = indices_clean[['SPX_Date', 'SPX']].rename(columns={'SPX_Date': 'Date'})
    indices_merged = (
        pd.merge(spi_df, spx_df, on='Date', how='outer')
        .sort_values('Date')
        .dropna()
    )
    
    indices_merged['SPI_logret'] = np.log(indices_merged['SPI'] / indices_merged['SPI'].shift(1))
    indices_merged['SPX_logret'] = np.log(indices_merged['SPX'] / indices_merged['SPX'].shift(1))
    indices_merged = indices_merged.dropna()
    
    Theta1 = indices_merged['SPI_logret'].values
    Theta2 = indices_merged['SPX_logret'].values
    
    return indices_merged, Theta1, Theta2


# -------------------------------------------------------------
# (2) Simulation Function
# -------------------------------------------------------------
def simulation(portfolio, Theta1, Theta2, n_simulations, model='M1', seed=42):
    """
    Simulate Y_k for all counterparties using models M1, M2, or M3.
    """
    np.random.seed(seed)
    n_counterparties = len(portfolio)

    # Portfolio parameters
    lambda_k = portfolio['lambda_k'].values
    a_k1 = portfolio['a_k1'].values
    a_k2 = portfolio['a_k2'].values
    pi_k = portfolio['pi_k'].values

    # --- Model Specification ---
    if model == 'M1':
        indices = np.random.choice(len(Theta1), size=n_simulations, replace=True)
        Theta_samples = np.column_stack([Theta1[indices], Theta2[indices]])
        Sigma_Theta = np.cov(np.column_stack([Theta1, Theta2]).T)

    elif model == 'M2':
        mu = np.array([np.mean(Theta1), np.mean(Theta2)])
        sigma1, sigma2 = np.std(Theta1, ddof=1), np.std(Theta2, ddof=1)
        rho = np.corrcoef(Theta1, Theta2)[0, 1]
        Sigma_Theta = np.array([
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2]
        ])
        Theta_samples = multivariate_normal.rvs(mean=mu, cov=Sigma_Theta, size=n_simulations)

    elif model == 'M3':
        mu1, mu2 = np.mean(Theta1), np.mean(Theta2)
        sigma1, sigma2 = np.std(Theta1, ddof=1), np.std(Theta2, ddof=1)
        u1 = norm.cdf(Theta1, mu1, sigma1)
        u2 = norm.cdf(Theta2, mu2, sigma2)
        U = np.column_stack([u1, u2])

        copula = StudentCopula(dim=2)
        copula.fit(U, method='ml')
        rho_m3 = float(np.ravel(copula.params.rho)[0])

        U_samples = copula.random(n_simulations)
        Theta_samples = np.zeros((n_simulations, 2))
        Theta_samples[:, 0] = norm.ppf(U_samples[:, 0], loc=mu1, scale=sigma1)
        Theta_samples[:, 1] = norm.ppf(U_samples[:, 1], loc=mu2, scale=sigma2)

        Sigma_Theta = np.array([
            [sigma1**2, rho_m3 * sigma1 * sigma2],
            [rho_m3 * sigma1 * sigma2, sigma2**2]
        ])

    else:
        raise ValueError("Model must be one of 'M1', 'M2', or 'M3'.")

    # --- Compute s_k ---
    s_k = np.array([np.sqrt(np.array([a_k1[k], a_k2[k]]).T @ Sigma_Theta @ np.array([a_k1[k], a_k2[k]]))
                    for k in range(n_counterparties)])

    # --- Simulate Y_k ---
    Y_k_simulations = np.zeros((n_simulations, n_counterparties))
    for sim in range(n_simulations):
        Theta_vec = Theta_samples[sim, :]
        epsilons = np.random.randn(n_counterparties)
        Y_k_simulations[sim, :] = (
            np.sqrt(lambda_k) * (a_k1 * Theta_vec[0] + a_k2 * Theta_vec[1])
            + np.sqrt(1 - lambda_k) * s_k * epsilons
        )

    # --- Default thresholds ---
    d_k = np.array([np.quantile(Y_k_simulations[:, k], pi_k[k]) for k in range(n_counterparties)])
    return Y_k_simulations, d_k, Sigma_Theta


# -------------------------------------------------------------
# (3) Portfolio Loss and Risk Measures
# -------------------------------------------------------------
def portfolio_loss(Y_k, d_k, E_k, R_k):
    """Compute total (unnormalized) portfolio loss."""
    I_k = (Y_k <= d_k).astype(int)
    return np.sum(E_k * (1 - R_k) * I_k, axis=1)


def risk_measures(L, alpha=0.95):
    """Compute VaR and ES for given loss distribution."""
    var = np.quantile(L, alpha)
    es = L[L >= var].mean()
    return var, es
