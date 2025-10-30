# functions.py
# =============================================================
# Utility and Simulation Functions for QRM Assignment
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
from copulae import StudentCopula
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    import pandas as pd
    import numpy as np

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


def dynamic_var_es_weekly_window(portfolio, indices_df, window=500, n_simulations=5000, alpha=0.99):
    """
    Compute dynamic VaR and ES over time using models M1 to M3.
    At each day, use the last 500 daily observations to compute weekly returns,
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
    alpha : float, default=0.99
        Confidence level for VaR and ES.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, VaR_M1, ES_M1, VaR_M2, ES_M2, VaR_M3, ES_M3
    """

    # Clean and merge full daily data once
    daily_data, _, _ = preprocess_indices(indices_df, frequency="daily")

    results = []
    dates = daily_data['Date'].iloc[window:].reset_index(drop=True)

    # Rolling loop
    for i in tqdm(range(window, len(daily_data))):
        # 1. Select the rolling 500-day window
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
        Theta1 = weekly['SPI_logret'].values
        Theta2 = weekly['SPX_logret'].values

        # 3. Run simulations for each model
        var_es = {}
        for model in ['M1', 'M2', 'M3']:
            try:
                Y_k, d_k, Sigma = simulation(portfolio, Theta1, Theta2, n_simulations=n_simulations, model=model)
                losses = portfolio_loss(Y_k, d_k)
                var_t, es_t = risk_measures(losses, alpha=alpha)
                var_es[f'VaR_{model}'] = var_t
                var_es[f'ES_{model}'] = es_t
            except Exception as e:
                # Handle convergence/numerical issues gracefully
                var_es[f'VaR_{model}'] = np.nan
                var_es[f'ES_{model}'] = np.nan

        var_es['Date'] = daily_data['Date'].iloc[i]
        results.append(var_es)

    results_df = pd.DataFrame(results)
    return results_df
