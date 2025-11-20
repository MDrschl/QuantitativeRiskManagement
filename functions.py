# functions.py
# =============================================================
# Utility and Simulation Functions for QRM Assignment
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal, t as t_dist, skew, kurtosis
from tqdm import tqdm
from tcopula import fit_t_copula


# -------------------------------------------------------------
# (1) Data Preprocessing
# -------------------------------------------------------------
def preprocess_indices(indices_df, frequency="daily"):
    """
    Cleans and merges index data (SPI and SPX) and computes log returns.
    
    Parameters
    ----------
    indices_df : pd.DataFrame
        Raw dataframe with SPI and SPX columns.
    frequency : str, optional
        Frequency of returns to compute. Options:
        - "daily"  (default)
        - "weekly" (uses Friday or last business day of week)
    
    Returns
    -------
    indices_merged : pd.DataFrame
        Cleaned dataframe with log returns.
    Theta1 : np.ndarray
        SPI log returns.
    Theta2 : np.ndarray
        SPX log returns.
    """
    # Clean the dataframe
    indices_clean = indices_df.copy()
    indices_clean.columns = ['SPI_Date', 'SPI', 'SPX_Date', 'SPX']
    indices_clean = indices_clean.iloc[1:]  # remove header row if duplicated

    # Convert to datetime and numeric
    indices_clean['SPI_Date'] = pd.to_datetime(indices_clean['SPI_Date'])
    indices_clean['SPX_Date'] = pd.to_datetime(indices_clean['SPX_Date'])
    indices_clean['SPI'] = pd.to_numeric(indices_clean['SPI'], errors='coerce')
    indices_clean['SPX'] = pd.to_numeric(indices_clean['SPX'], errors='coerce')

    # Merge
    spi_df = indices_clean[['SPI_Date', 'SPI']].rename(columns={'SPI_Date': 'Date'})
    spx_df = indices_clean[['SPX_Date', 'SPX']].rename(columns={'SPX_Date': 'Date'})
    indices_merged = (
        pd.merge(spi_df, spx_df, on='Date', how='outer')
        .sort_values('Date')
        .dropna()
    )

    if frequency == "weekly":
        indices_merged = (
            indices_merged
            .set_index('Date')
            .resample('W-FRI')
            .last()
            .dropna()
            .reset_index()
        )
    elif frequency != "daily":
        raise ValueError("frequency must be either 'daily' or 'weekly'")

    # Compute log returns
    indices_merged['SPI_logret'] = np.log(
        indices_merged['SPI'] / indices_merged['SPI'].shift(1)
    )
    indices_merged['SPX_logret'] = np.log(
        indices_merged['SPX'] / indices_merged['SPX'].shift(1)
    )
    indices_merged = indices_merged.dropna()

    Theta1 = indices_merged['SPI_logret'].values
    Theta2 = indices_merged['SPX_logret'].values

    return indices_merged, Theta1, Theta2


# -------------------------------------------------------------
# (2) Simulation Function with ML t-Copula
# -------------------------------------------------------------
def simulation(portfolio, Theta1, Theta2, n_simulations, model='M1', seed=42):
    """
    Simulate Y_k for all counterparties using models M1, M2, or M3.
    M3 uses maximum likelihood estimation for a t-copula with Gaussian marginals.
    
    Parameters
    ----------
    portfolio : pd.DataFrame
        Portfolio with columns: lambda_k, a_k1, a_k2, pi_k, Exposure USD, R_k
    Theta1, Theta2 : array-like
        Historical return series (e.g., SPI and SPX log returns)
    n_simulations : int
        Number of Monte Carlo simulations
    model : str
        'M1' (empirical resampling), 'M2' (Gaussian), 'M3' (t-copula with ML)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    Y_k_simulations : np.ndarray
        Simulated credit factors (n_simulations × n_counterparties)
    d_k : np.ndarray
        Default thresholds for each counterparty
    Sigma_Theta : np.ndarray
        Covariance matrix of the systemic factors
    copula_params : dict or None
        For M3: {'rho': correlation, 'nu': degrees of freedom, 'converged': bool}
    """
    np.random.seed(seed)
    n_counterparties = len(portfolio)

    # Portfolio parameters
    lambda_k = portfolio['lambda_k'].values
    a_k1 = portfolio['a_k1'].values
    a_k2 = portfolio['a_k2'].values
    pi_k = portfolio['pi_k'].values

    copula_params = None

    # --- Model Specification ---
    if model == 'M1':
        # Empirical resampling
        indices = np.random.choice(len(Theta1), size=n_simulations, replace=True)
        Theta_samples = np.column_stack([Theta1[indices], Theta2[indices]])
        Sigma_Theta = np.cov(np.column_stack([Theta1, Theta2]).T)

    elif model == 'M2':
        # Gaussian copula (bivariate normal)
        mu = np.array([np.mean(Theta1), np.mean(Theta2)])
        sigma1, sigma2 = np.std(Theta1, ddof=0), np.std(Theta2, ddof=0)
        rho = np.corrcoef(Theta1, Theta2)[0, 1]
        Sigma_Theta = np.array([
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2]
        ])
        Theta_samples = multivariate_normal.rvs(
            mean=mu, cov=Sigma_Theta, size=n_simulations
        )

    elif model == 'M3':
        # t-Copula with Gaussian marginals using ML estimation
        # 1) Estimate Gaussian marginals
        mu1, sigma1 = np.mean(Theta1), np.std(Theta1, ddof=0)
        mu2, sigma2 = np.mean(Theta2), np.std(Theta2, ddof=0)

        # 2) Transform historical returns to uniforms via normal CDF
        z1 = (Theta1 - mu1) / sigma1
        z2 = (Theta2 - mu2) / sigma2
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)

        # 3) Fit t-copula to uniforms
        rho_copula, nu_copula, converged = fit_t_copula(u1, u2)

        # Fallback conditions:
        #   - optimisation failed to converge
        #   - rho or nu are NaN
        if ((not converged) or
            np.isnan(rho_copula) or
            np.isnan(nu_copula)):
            print("Warning: t-copula estimation did not converge. "
                  "Falling back to Gaussian copula.")
            # Fallback: Gaussian with same marginals
            rho = np.corrcoef(Theta1, Theta2)[0, 1]
            Sigma_Theta = np.array([
                [sigma1**2, rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2**2]
            ])
            mu = np.array([mu1, mu2])
            Theta_samples = multivariate_normal.rvs(
                mean=mu, cov=Sigma_Theta, size=n_simulations
            )
            copula_params = {'rho': rho, 'nu': np.nan, 'converged': False}
        else:
            # Successful t-copula fit
            copula_params = {
                'rho': rho_copula,
                'nu': nu_copula,
                'converged': True
            }

            # 4) Generate samples from bivariate t with correlation rho_copula, df = nu_copula
            mean_t = np.zeros(2)
            cov_t = np.array([[1.0, rho_copula],
                              [rho_copula, 1.0]])

            # 5) Generate multivariate t: X = sqrt(nu/W) * Z, Z ~ N(0, cov_t), W ~ chi^2_nu
            Z = multivariate_normal.rvs(mean=mean_t, cov=cov_t, size=n_simulations)
            W = np.random.chisquare(nu_copula, size=n_simulations)
            t_samples = np.sqrt(nu_copula / W)[:, np.newaxis] * Z

            # 6) Transform to uniforms using t CDF
            U_samples = np.clip(t_dist.cdf(t_samples, df=nu_copula), 1e-10, 1-1e-10)

            # 7) Map uniforms to Gaussian marginals with (mu_i, sigma_i)
            Theta_samples = np.zeros((n_simulations, 2))
            Theta_samples[:, 0] = norm.ppf(U_samples[:, 0], loc=mu1, scale=sigma1)
            Theta_samples[:, 1] = norm.ppf(U_samples[:, 1], loc=mu2, scale=sigma2)

            # 8) Covariance matrix computed from simulated theta
            Sigma_Theta = np.cov(Theta_samples.T, ddof=0)

    else:
        raise ValueError("Model must be one of 'M1', 'M2', or 'M3'.")

    # --- Compute s_k ---
    s_k = np.array([
        np.sqrt(np.array([a_k1[k], a_k2[k]]).T @ Sigma_Theta @ np.array([a_k1[k], a_k2[k]]))
        for k in range(n_counterparties)
    ])

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
    d_k = np.array([
        np.quantile(Y_k_simulations[:, k], pi_k[k]) 
        for k in range(n_counterparties)
    ])
    
    return Y_k_simulations, d_k, Sigma_Theta, copula_params


# -------------------------------------------------------------
# (3) Portfolio Loss and Risk Measures
# -------------------------------------------------------------
def portfolio_loss(Y_k, d_k, E_k, R_k):
    """
    Compute total (unnormalized) portfolio loss.
    """
    I_k = (Y_k <= d_k).astype(int)
    return np.sum(E_k * (1 - R_k) * I_k, axis=1)


def risk_measures(L, alpha=0.95):
    """
    Compute VaR and ES for given loss distribution.
    """
    var = np.quantile(L, alpha)
    es = L[L >= var].mean()
    return var, es


def loss_statistics(L, model_name=""):
    """
    Compute comprehensive statistics for a loss distribution.
    
    Parameters
    ----------
    L : np.ndarray
        Loss distribution
    model_name : str
        Name of the model (for display purposes)
    
    Returns
    -------
    dict
        Dictionary containing various statistics
    """
    stats = {
        'Model': model_name,
        'Mean': np.mean(L),
        'Median': np.median(L),
        'Std Dev': np.std(L),
        'Variance': np.var(L),
        'Min': np.min(L),
        'Max': np.max(L),
        'Range': np.max(L) - np.min(L),
        'Q1 (25%)': np.percentile(L, 25),
        'Q3 (75%)': np.percentile(L, 75),
        'IQR': np.percentile(L, 75) - np.percentile(L, 25),
        'Skewness': skew(L),
        'Kurtosis': kurtosis(L),
        'CV (Coef. of Variation)': np.std(L) / np.mean(L) if np.mean(L) != 0 else np.nan,
        'P(Loss > 0)': np.mean(L > 0),
        'P(Loss = 0)': np.mean(L == 0),
        'P90': np.percentile(L, 90),
        'P95': np.percentile(L, 95),
        'P99': np.percentile(L, 99),
        'P99.5': np.percentile(L, 99.5),
        'P99.9': np.percentile(L, 99.9),
    }
    return stats


def default_statistics(Y_k, d_k, portfolio, model_name=""):
    """
    Compute default-related statistics across simulations.
    
    Parameters
    ----------
    Y_k : np.ndarray
        Simulated Y_k values (n_simulations × n_counterparties)
    d_k : np.ndarray
        Default thresholds
    portfolio : pd.DataFrame
        Portfolio data
    model_name : str
        Name of the model
    
    Returns
    -------
    dict
        Dictionary containing default statistics
    """
    I_k = (Y_k <= d_k).astype(int)
    n_defaults_per_sim = I_k.sum(axis=1)
    
    # Exposure-weighted default rate
    E_k = portfolio['Exposure USD'].values
    exposure_weighted_defaults = (I_k * E_k).sum(axis=1) / E_k.sum()
    
    stats = {
        'Model': model_name,
        'Mean # Defaults': np.mean(n_defaults_per_sim),
        'Median # Defaults': np.median(n_defaults_per_sim),
        'Std # Defaults': np.std(n_defaults_per_sim),
        'Max # Defaults': np.max(n_defaults_per_sim),
        'Min # Defaults': np.min(n_defaults_per_sim),
        'P(No Defaults)': np.mean(n_defaults_per_sim == 0),
        'P(≥1 Default)': np.mean(n_defaults_per_sim >= 1),
        'P(≥5 Defaults)': np.mean(n_defaults_per_sim >= 5),
        'P(≥10 Defaults)': np.mean(n_defaults_per_sim >= 10),
        'Mean Exposure-Weighted Default Rate': np.mean(exposure_weighted_defaults),
        'P95 # Defaults': np.percentile(n_defaults_per_sim, 95),
        'P99 # Defaults': np.percentile(n_defaults_per_sim, 99),
    }
    
    return stats


# -------------------------------------------------------------
# (4) Convert Daily to Weekly Log Returns
# -------------------------------------------------------------
def convert_weekly(indices_daily):
    """
    Aggregates daily index levels to weekly frequency and computes weekly log returns.
    """
    df = indices_daily.copy()
    df = df.sort_values("Date").set_index("Date")

    df_weekly = df.resample("W-FRI").last().dropna()

    df_weekly["SPI_logret"] = np.log(df_weekly["SPI"] / df_weekly["SPI"].shift(1))
    df_weekly["SPX_logret"] = np.log(df_weekly["SPX"] / df_weekly["SPX"].shift(1))
    df_weekly = df_weekly.dropna().reset_index()

    Theta1_weekly = df_weekly["SPI_logret"].values
    Theta2_weekly = df_weekly["SPX_logret"].values

    return df_weekly, Theta1_weekly, Theta2_weekly


# -------------------------------------------------------------
# (5) Dynamic VaR and ES with Rolling Window
# -------------------------------------------------------------
def dynamic_var_es_weekly_window(portfolio, indices_df, window=500, n_simulations=5000, alpha=0.95):
    """
    Compute dynamic VaR and ES over time using models M1 to M3 with ML t-copula.
    """
    # Clean and merge full daily data once
    daily_data, _, _ = preprocess_indices(indices_df, frequency="daily")
    
    E_k = portfolio['Exposure USD'].values
    R_k = portfolio['R_k'].values

    results = []

    # Rolling loop
    for i in tqdm(range(window, len(daily_data)), desc="Rolling window analysis"):
        sample = daily_data.iloc[i - window:i].copy()

        weekly = (
            sample.set_index('Date')
            .resample('W-FRI')
            .last()
            .dropna()
            .reset_index()
        )
        weekly['SPI_logret'] = np.log(weekly['SPI'] / weekly['SPI'].shift(1))
        weekly['SPX_logret'] = np.log(weekly['SPX'] / weekly['SPX'].shift(1))
        weekly = weekly.dropna()
        
        if len(weekly) < 20:
            continue
            
        Theta1 = weekly['SPI_logret'].values
        Theta2 = weekly['SPX_logret'].values

        var_es = {'Date': daily_data['Date'].iloc[i]}
        
        for model in ['M1', 'M2', 'M3']:
            try:
                Y_k, d_k, Sigma, copula_params = simulation(
                    portfolio, Theta1, Theta2, 
                    n_simulations=n_simulations, 
                    model=model,
                    seed=42
                )
                losses = portfolio_loss(Y_k, d_k, E_k, R_k)
                var_t, es_t = risk_measures(losses, alpha=alpha)
                
                var_es[f'VaR_{model}'] = var_t
                var_es[f'ES_{model}'] = es_t
                
                if model == 'M3' and copula_params is not None:
                    var_es['rho_M3'] = copula_params['rho']
                    var_es['nu_M3'] = copula_params['nu']
                    var_es['converged_M3'] = copula_params['converged']
                    
            except Exception as e:
                print(f"Warning at date {daily_data['Date'].iloc[i]}: {str(e)}")
                var_es[f'VaR_{model}'] = np.nan
                var_es[f'ES_{model}'] = np.nan
                if model == 'M3':
                    var_es['rho_M3'] = np.nan
                    var_es['nu_M3'] = np.nan
                    var_es['converged_M3'] = False

        results.append(var_es)

    results_df = pd.DataFrame(results)
    return results_df