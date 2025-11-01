# functions.py
# =============================================================
# Utility and Simulation Functions for QRM Assignment
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal, t as t_dist
from tqdm import tqdm
from tcopula import fit_gaussian_marginals_and_t_copula


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
    indices_merged['SPI_logret'] = np.log(indices_merged['SPI'] / indices_merged['SPI'].shift(1))
    indices_merged['SPX_logret'] = np.log(indices_merged['SPX'] / indices_merged['SPX'].shift(1))
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
    M3 uses maximum likelihood estimation for t-copula parameters.
    
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
        sigma1, sigma2 = np.std(Theta1, ddof=1), np.std(Theta2, ddof=1)
        rho = np.corrcoef(Theta1, Theta2)[0, 1]
        Sigma_Theta = np.array([
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2]
        ])
        Theta_samples = multivariate_normal.rvs(mean=mu, cov=Sigma_Theta, size=n_simulations)

    elif model == 'M3':
        # t-Copula with Gaussian marginals using ML estimation
        params = fit_gaussian_marginals_and_t_copula(Theta1, Theta2)
        
        if not params['converged']:
            print("Warning: t-copula estimation did not converge. Falling back to Gaussian copula.")
            # Fallback to M2
            mu = np.array([params['mu1'], params['mu2']])
            sigma1, sigma2 = params['sigma1'], params['sigma2']
            rho = params['rho'] if not np.isnan(params['rho']) else np.corrcoef(Theta1, Theta2)[0, 1]
            Sigma_Theta = np.array([
                [sigma1**2, rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2**2]
            ])
            Theta_samples = multivariate_normal.rvs(mean=mu, cov=Sigma_Theta, size=n_simulations)
            copula_params = {'rho': rho, 'nu': np.nan, 'converged': False}
        else:
            mu1, sigma1 = params['mu1'], params['sigma1']
            mu2, sigma2 = params['mu2'], params['sigma2']
            rho_copula = params['rho']
            nu_copula = params['nu']
            
            copula_params = {'rho': rho_copula, 'nu': nu_copula, 'converged': True}
            
            # Generate samples from t-copula
            # 1. Sample from bivariate t with correlation rho
            mean_t = np.zeros(2)
            cov_t = np.array([[1, rho_copula], [rho_copula, 1]])
            
            # Generate multivariate t samples
            # X ~ t_nu(0, Sigma) can be generated as X = Z / sqrt(S/nu)
            # where Z ~ N(0, Sigma) and S ~ chi^2(nu)
            Z = multivariate_normal.rvs(mean=mean_t, cov=cov_t, size=n_simulations)
            chi2_samples = np.random.chisquare(nu_copula, size=n_simulations)
            t_samples = Z / np.sqrt(chi2_samples / nu_copula)[:, np.newaxis]
            
            # 2. Transform to uniforms using t-cdf
            U_samples = t_dist.cdf(t_samples, df=nu_copula)
            
            # 3. Transform to Gaussian marginals
            Theta_samples = np.zeros((n_simulations, 2))
            Theta_samples[:, 0] = norm.ppf(U_samples[:, 0], loc=mu1, scale=sigma1)
            Theta_samples[:, 1] = norm.ppf(U_samples[:, 1], loc=mu2, scale=sigma2)
            
            # Covariance matrix (using copula correlation)
            Sigma_Theta = np.array([
                [sigma1**2, rho_copula * sigma1 * sigma2],
                [rho_copula * sigma1 * sigma2, sigma2**2]
            ])

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
    
    Parameters
    ----------
    Y_k : np.ndarray
        Credit factors (n_simulations × n_counterparties)
    d_k : np.ndarray
        Default thresholds (n_counterparties,)
    E_k : np.ndarray
        Exposures (n_counterparties,)
    R_k : np.ndarray
        Recovery rates (n_counterparties,)
    
    Returns
    -------
    L : np.ndarray
        Portfolio losses (n_simulations,)
    """
    I_k = (Y_k <= d_k).astype(int)
    return np.sum(E_k * (1 - R_k) * I_k, axis=1)


def risk_measures(L, alpha=0.95):
    """
    Compute VaR and ES for given loss distribution.
    
    Parameters
    ----------
    L : np.ndarray
        Loss distribution
    alpha : float
        Confidence level (default: 0.95)
    
    Returns
    -------
    var : float
        Value-at-Risk at level alpha
    es : float
        Expected Shortfall at level alpha
    """
    var = np.quantile(L, alpha)
    es = L[L >= var].mean()
    return var, es


# -------------------------------------------------------------
# (4) Convert Daily to Weekly Log Returns
# -------------------------------------------------------------
def convert_weekly(indices_daily):
    """
    Aggregates daily index levels to weekly frequency and computes weekly log returns.

    Parameters
    ----------
    indices_daily : pd.DataFrame
        DataFrame with columns ['Date', 'SPI', 'SPX'] at daily frequency.

    Returns
    -------
    indices_weekly : pd.DataFrame
        DataFrame with weekly SPI and SPX levels and weekly log returns.
    Theta1_weekly : np.ndarray
        Weekly log returns of SPI (for Θ₁).
    Theta2_weekly : np.ndarray
        Weekly log returns of SPX (for Θ₂).
    """
    df = indices_daily.copy()
    df = df.sort_values("Date").set_index("Date")

    # --- Resample to weekly frequency using last available trading day ---
    df_weekly = df.resample("W-FRI").last().dropna()

    # --- Compute weekly log returns ---
    df_weekly["SPI_logret"] = np.log(df_weekly["SPI"] / df_weekly["SPI"].shift(1))
    df_weekly["SPX_logret"] = np.log(df_weekly["SPX"] / df_weekly["SPX"].shift(1))
    df_weekly = df_weekly.dropna().reset_index()

    # --- Extract as numpy arrays for simulation ---
    Theta1_weekly = df_weekly["SPI_logret"].values
    Theta2_weekly = df_weekly["SPX_logret"].values

    return df_weekly, Theta1_weekly, Theta2_weekly


# -------------------------------------------------------------
# (5) Dynamic VaR and ES with Rolling Window
# -------------------------------------------------------------
def dynamic_var_es_weekly_window(portfolio, indices_df, window=500, n_simulations=5000, alpha=0.95):
    """
    Compute dynamic VaR and ES over time using models M1 to M3 with ML t-copula.
    At each day, use the last 'window' daily observations to compute weekly returns,
    fit the model, and simulate losses.

    Parameters
    ----------
    portfolio : pd.DataFrame
        Portfolio data.
    indices_df : pd.DataFrame
        Raw indices dataframe with SPI and SPX columns.
    window : int, default=500
        Rolling window length in trading days.
    n_simulations : int, default=5000
        Number of simulations per window.
    alpha : float, default=0.95
        Confidence level for VaR and ES.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, VaR_M1, ES_M1, VaR_M2, ES_M2, VaR_M3, ES_M3,
        rho_M3, nu_M3, converged_M3
    """
    # Clean and merge full daily data once
    daily_data, _, _ = preprocess_indices(indices_df, frequency="daily")
    
    E_k = portfolio['Exposure USD'].values
    R_k = portfolio['R_k'].values

    results = []

    # Rolling loop
    for i in tqdm(range(window, len(daily_data)), desc="Rolling window analysis"):
        # 1. Select the rolling window
        sample = daily_data.iloc[i - window:i].copy()

        # 2. Convert that window into weekly returns
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
        
        if len(weekly) < 20:  # Need minimum data
            continue
            
        Theta1 = weekly['SPI_logret'].values
        Theta2 = weekly['SPX_logret'].values

        # 3. Run simulations for each model
        var_es = {'Date': daily_data['Date'].iloc[i]}
        
        for model in ['M1', 'M2', 'M3']:
            try:
                Y_k, d_k, Sigma, copula_params = simulation(
                    portfolio, Theta1, Theta2, 
                    n_simulations=n_simulations, 
                    model=model,
                    seed=42  # Use same seed for reproducibility
                )
                losses = portfolio_loss(Y_k, d_k, E_k, R_k)
                var_t, es_t = risk_measures(losses, alpha=alpha)
                
                var_es[f'VaR_{model}'] = var_t
                var_es[f'ES_{model}'] = es_t
                
                # Store copula parameters for M3
                if model == 'M3' and copula_params is not None:
                    var_es['rho_M3'] = copula_params['rho']
                    var_es['nu_M3'] = copula_params['nu']
                    var_es['converged_M3'] = copula_params['converged']
                    
            except Exception as e:
                # Handle convergence/numerical issues gracefully
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